"""
Сборщик датасета для прогнозирования паводков в Кыргызстане.

Источники данных:
    1. Open-Meteo Historical API — реальные почасовые погодные данные 2019–2024
    2. МЧС Кыргызстана (data/open_data/) — каталог опасных зон, калибровка порогов
    3. DFO FloodArchive + EM-DAT — подтверждённые паводковые события

Выходной файл: data/kyrgyzstan_floods_dataset.csv

Запуск:
    cd /home/ksw-pc/claude_docs/diploma
    source venv/bin/activate
    python scripts/build_dataset.py

Опции:
    --region chui       # только один регион (быстро)
    --start 2022-01-01  # начало периода
    --end 2024-12-31    # конец периода
    --freq D            # частота: H (часы) или D (дни)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REGIONS
from src.data.mchs_parser import get_region_stats, get_flood_threshold_multiplier

# ===================== КОНФИГУРАЦИЯ =====================

OPEN_METEO_ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'

# Переменные, запрашиваемые у Open-Meteo
HOURLY_VARS = [
    'precipitation',
    'temperature_2m',
    'relative_humidity_2m',
    'wind_speed_10m',
    'snowfall',
    'snow_depth',
]

# Подтверждённые паводковые события из DFO/EM-DAT
CONFIRMED_FLOOD_EVENTS = [
    # (start_date, end_date, affected_regions, severity)
    ('2024-04-21', '2024-04-22',
     ['batken', 'jalal_abad', 'naryn', 'osh', 'talas'], 'critical'),
    ('2024-05-05', '2024-05-07', ['jalal_abad'], 'moderate'),
    ('2024-07-14', '2024-07-16', ['osh'], 'high'),
]

# Пороги расхода по умолчанию (м³/с) из src/config.py
DEFAULT_FLOOD_THRESHOLDS = {
    'chui': 150.0,
    'issyk_kul': 200.0,
    'naryn': 180.0,
    'osh': 160.0,
    'jalal_abad': 170.0,
    'talas': 120.0,
    'batken': 130.0,
}

OUTPUT_PATH = Path('data/kyrgyzstan_floods_dataset.csv')


# ===================== ЗАГРУЗКА ДАННЫХ =====================

def fetch_open_meteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Загружает исторические данные Open-Meteo для точки (lat, lon)."""
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(HOURLY_VARS),
        'timezone': 'Asia/Bishkek',
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get('hourly', {})
            if not hourly or 'time' not in hourly:
                return None

            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly['time']),
                'precipitation_mm': hourly.get('precipitation', [np.nan] * len(hourly['time'])),
                'temperature_c': hourly.get('temperature_2m', [np.nan] * len(hourly['time'])),
                'humidity': hourly.get('relative_humidity_2m', [np.nan] * len(hourly['time'])),
                'wind_speed': hourly.get('wind_speed_10m', [np.nan] * len(hourly['time'])),
                'snowfall_cm': hourly.get('snowfall', [np.nan] * len(hourly['time'])),
                'snow_depth_m': hourly.get('snow_depth', [np.nan] * len(hourly['time'])),
            })
            df = df.fillna(0.0)
            return df

        except requests.Timeout:
            print(f'    Timeout (попытка {attempt + 1}/{max_retries})')
            time.sleep(5 * (attempt + 1))
        except requests.HTTPError as e:
            print(f'    HTTP ошибка: {e}')
            if e.response.status_code == 429:  # rate limit
                time.sleep(30)
            else:
                return None
        except Exception as e:
            print(f'    Ошибка: {e}')
            return None

    return None


def simulate_discharge(
    weather_df: pd.DataFrame,
    region: str,
    cn: float = 75.0,
    ddf: float = 4.5,
) -> pd.Series:
    """
    Упрощённое гидрологическое моделирование.

    SCS-CN метод для стока + Degree-Day для снеготаяния.
    Возвращает почасовой расход в м³/с.
    """
    from src.config import REGIONS
    config = REGIONS.get(region)
    area_km2 = config.area_km2 if config else 20000.0

    precip = weather_df['precipitation_mm'].values.copy()
    temp = weather_df['temperature_c'].values.copy()
    snowfall = weather_df['snowfall_cm'].values.copy() * 10  # → мм

    n = len(precip)
    discharge = np.zeros(n)
    snow_pack = 0.0
    api = 0.0  # Antecedent Precipitation Index

    s_param = 25400.0 / cn - 254.0
    ia = 0.2 * s_param

    for i in range(n):
        # Обновление API (индекс предшествующих осадков)
        api = api * 0.85 + precip[i]

        # Накопление снега
        if temp[i] < 0:
            snow_pack += snowfall[i]

        # Снеготаяние
        snowmelt = 0.0
        if temp[i] > 0:
            snowmelt = min(ddf * temp[i] / 24.0, snow_pack)
            snow_pack = max(0.0, snow_pack - snowmelt)

        # Эффективные осадки = дождь + снеготаяние
        p_eff = precip[i] + snowmelt

        # SCS-CN: поверхностный сток
        if p_eff > ia:
            runoff = (p_eff - ia) ** 2 / (p_eff - ia + s_param)
        else:
            runoff = 0.0

        # Коррекция AMC по API
        if api > 53:
            runoff *= 1.4
        elif api < 35:
            runoff *= 0.7

        # Перевод мм → м³/с (площадь в км², часовой шаг)
        # Q = runoff[мм] * area[км²] * 1000 / 3600 [м³/с]
        discharge[i] = runoff * area_km2 * 1000.0 / 3600.0

    # Маршрутизация стока: Мускингум (k_steps дней, x=0.2)
    k_steps = 3  # 3 дня задержки для горных рек
    x = 0.2
    c0 = (1 - 2 * k_steps * x) / (2 * k_steps * (1 - x) + 1)
    c1 = (1 + 2 * k_steps * x) / (2 * k_steps * (1 - x) + 1)
    c2 = (2 * k_steps * (1 - x) - 1) / (2 * k_steps * (1 - x) + 1)

    routed = np.zeros(n)
    routed[0] = discharge[0]
    for i in range(1, n):
        routed[i] = c0 * discharge[i] + c1 * discharge[i - 1] + c2 * routed[i - 1]
        routed[i] = max(0.0, routed[i])

    # Ограничение физически возможным максимумом (площадь × 10 мм/день)
    area_km2 = config.area_km2 if config else 20000.0
    max_possible = area_km2 * 1000.0 * 0.01 / 86400.0 * 1000.0  # ~2.3 м³/с на км²
    routed = np.clip(routed, 0, max_possible)

    return pd.Series(routed, index=weather_df.index)


# ===================== МЕТКИ =====================

def assign_risk_labels(
    df: pd.DataFrame,
    region: str,
    flood_threshold: float,
) -> pd.DataFrame:
    """
    Присваивает метки риска каждой строке.

    Приоритеты (по убыванию):
    1. Подтверждённые события DFO/EM-DAT → метка severity напрямую
    2. Расход > порог × коэф. → высокий/критический
    3. 90-й перцентиль расхода → умеренный
    4. Остальное → низкий
    """
    df = df.copy()

    # Перцентили расхода для этого региона
    q50 = df['discharge_m3s'].quantile(0.50)
    q85 = df['discharge_m3s'].quantile(0.85)
    q95 = df['discharge_m3s'].quantile(0.95)

    # Базовые метки по расходу
    conditions = [
        df['discharge_m3s'] >= flood_threshold * 1.5,
        df['discharge_m3s'] >= flood_threshold,
        df['discharge_m3s'] >= q85,
        df['discharge_m3s'] >= q50,
    ]
    choices = ['critical', 'high', 'moderate', 'low']
    df['risk_level'] = np.select(conditions, choices, default='low')

    # Бинарный флаг паводка
    df['flood'] = (df['risk_level'].isin(['high', 'critical'])).astype(int)

    # Переопределяем подтверждёнными событиями
    for start, end, regions, severity in CONFIRMED_FLOOD_EVENTS:
        if region not in regions:
            continue
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        if mask.any():
            df.loc[mask, 'risk_level'] = severity
            df.loc[mask, 'flood'] = 1

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет временные признаки."""
    df = df.copy()
    ts = df['timestamp']
    df['month'] = ts.dt.month
    df['day_of_year'] = ts.dt.dayofyear
    df['hour'] = ts.dt.hour
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_spring'] = ts.dt.month.isin([3, 4, 5]).astype(int)
    df['is_summer'] = ts.dt.month.isin([6, 7, 8]).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет скользящие признаки осадков и температуры."""
    df = df.copy()
    precip = df['precipitation_mm']
    temp = df['temperature_c']

    for window in [6, 24, 72]:
        df[f'precip_{window}h'] = precip.rolling(window, min_periods=1).sum()
        df[f'temp_mean_{window}h'] = temp.rolling(window, min_periods=1).mean()

    # API — Antecedent Precipitation Index
    df['api'] = precip.ewm(span=72).mean()

    # Изменение осадков
    df['precip_change_6h'] = precip.diff(6).fillna(0)

    return df


def add_mchs_features(df: pd.DataFrame, region: str, stats: pd.DataFrame) -> pd.DataFrame:
    """Добавляет региональные признаки из каталога МЧС."""
    df = df.copy()
    row = stats[stats['region'] == region]
    if row.empty:
        df['mchs_flood_zones'] = 0
        df['mchs_lake_danger'] = 1
        df['mchs_density_norm'] = 0.5
    else:
        df['mchs_flood_zones'] = int(row['flood_zones'].iloc[0])
        df['mchs_lake_danger'] = int(row['max_danger_score'].iloc[0])
        df['mchs_density_norm'] = float(row['flood_density_norm'].iloc[0])
    return df


# ===================== ОСНОВНАЯ ЛОГИКА =====================

def build_region_dataset(
    region: str,
    start_date: str,
    end_date: str,
    mchs_stats: pd.DataFrame,
    freq: str = 'D',
) -> pd.DataFrame | None:
    """Строит датасет для одного региона."""
    config = REGIONS.get(region)
    if config is None:
        print(f'  Регион {region} не найден в конфигурации')
        return None

    lat = (config.lat_min + config.lat_max) / 2
    lon = (config.lon_min + config.lon_max) / 2

    print(f'  Загружаем Open-Meteo для {region} ({lat:.2f}°N {lon:.2f}°E)...')
    weather_hourly = fetch_open_meteo(lat, lon, start_date, end_date)

    if weather_hourly is None or len(weather_hourly) == 0:
        print(f'  Не удалось загрузить данные для {region}')
        return None

    print(f'  Получено {len(weather_hourly)} почасовых записей')

    # ── Агрегируем к суткам (SCS-CN требует суточных сумм осадков) ──────────
    weather_hourly = weather_hourly.set_index('timestamp')
    daily = weather_hourly.resample('D').agg({
        'precipitation_mm': 'sum',
        'temperature_c': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'snowfall_cm': 'sum',
        'snow_depth_m': 'mean',
    }).reset_index()
    daily = daily.rename(columns={'timestamp': 'timestamp'})

    # ── Симуляция расхода на суточных данных ────────────────────────────────
    daily['discharge_m3s'] = simulate_discharge(daily, region)

    # ── Скользящие признаки (по суткам: окна 3, 7, 30 дней) ─────────────────
    precip = daily['precipitation_mm']
    temp = daily['temperature_c']
    for window, label in [(3, '3d'), (7, '7d'), (30, '30d')]:
        daily[f'precip_{label}'] = precip.rolling(window, min_periods=1).sum()
        daily[f'temp_mean_{label}'] = temp.rolling(window, min_periods=1).mean()

    daily['api'] = precip.ewm(span=30).mean()
    daily['precip_change_3d'] = precip.diff(3).fillna(0)

    # ── Временные признаки ────────────────────────────────────────────────────
    ts = daily['timestamp']
    daily['month'] = ts.dt.month
    daily['day_of_year'] = ts.dt.dayofyear
    daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
    daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
    daily['is_spring'] = ts.dt.month.isin([3, 4, 5]).astype(int)
    daily['is_summer'] = ts.dt.month.isin([6, 7, 8]).astype(int)

    # ── Признаки МЧС ─────────────────────────────────────────────────────────
    daily = add_mchs_features(daily, region, mchs_stats)

    # ── Порог паводка (скорректированный по МЧС) ─────────────────────────────
    base_threshold = DEFAULT_FLOOD_THRESHOLDS.get(region, 150.0)
    flood_threshold = get_flood_threshold_multiplier(region, base_threshold)

    # ── Метки риска ───────────────────────────────────────────────────────────
    daily = assign_risk_labels(daily, region, flood_threshold)

    # ── Идентификатор региона ─────────────────────────────────────────────────
    daily.insert(1, 'region', region)

    print(f'  Паводков: {daily["flood"].sum()} / {len(daily)} '
          f'({daily["flood"].mean() * 100:.1f}%)')
    print(f'  Расход: min={daily["discharge_m3s"].min():.1f}, '
          f'mean={daily["discharge_m3s"].mean():.1f}, '
          f'max={daily["discharge_m3s"].max():.1f} м³/с')
    print(f'  Порог паводка: {flood_threshold:.1f} м³/с')

    return daily


def main():
    parser = argparse.ArgumentParser(description='Сборщик датасета паводков Кыргызстана')
    parser.add_argument('--region', default=None, help='Один регион (all = все)')
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--freq', default='D', choices=['H', 'D'])
    parser.add_argument('--output', default=str(OUTPUT_PATH))
    args = parser.parse_args()

    regions_to_process = (
        [args.region] if args.region and args.region != 'all'
        else list(REGIONS.keys())
    )

    print('=' * 60)
    print('Сборка датасета паводков Кыргызстана')
    print(f'Период: {args.start} — {args.end}')
    print(f'Регионы: {regions_to_process}')
    print(f'Частота: {"почасовая" if args.freq == "H" else "ежедневная"}')
    print('=' * 60)

    # Загружаем статистику МЧС
    print('\n[1/3] Парсинг данных МЧС...')
    mchs_stats = get_region_stats()
    print(f'Загружено {len(mchs_stats)} регионов из каталога МЧС')
    print(mchs_stats[['region', 'flood_zones', 'lake_zones',
                       'lake_danger_i', 'max_danger_score']].to_string(index=False))

    # Строим датасет по регионам
    print('\n[2/3] Загрузка погодных данных и симуляция...')
    frames = []
    for i, region in enumerate(regions_to_process, 1):
        print(f'\n  [{i}/{len(regions_to_process)}] {region}')
        df = build_region_dataset(region, args.start, args.end, mchs_stats, args.freq)
        if df is not None:
            frames.append(df)
        time.sleep(1)  # пауза между запросами к API

    if not frames:
        print('\nОшибка: нет данных ни по одному региону')
        sys.exit(1)

    # Объединяем и сохраняем
    print('\n[3/3] Сохранение датасета...')
    dataset = pd.concat(frames, ignore_index=True)

    # Базовая статистика
    print(f'\nИтого строк: {len(dataset):,}')
    print(f'Регионы: {dataset["region"].unique().tolist()}')
    print(f'Паводковых записей: {dataset["flood"].sum():,} '
          f'({dataset["flood"].mean() * 100:.1f}%)')
    print('\nРаспределение уровней риска:')
    print(dataset['risk_level'].value_counts().to_string())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding='utf-8')
    print(f'\nДатасет сохранён: {output_path}')
    print(f'Размер файла: {output_path.stat().st_size / 1024:.1f} KB')


if __name__ == '__main__':
    main()
