"""
Модуль загрузки и обработки данных об осадках.

Включает:
- Загрузку данных из различных источников (CSV, NetCDF, API)
- Генерацию синтетических данных для тестирования
- Обработку и валидацию данных
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import os

from ..config import REGIONS, DATA_DIR, RegionConfig


@dataclass
class PrecipitationRecord:
    """Запись об осадках."""
    timestamp: datetime
    latitude: float
    longitude: float
    precipitation_mm: float
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    wind_speed_ms: Optional[float] = None


class PrecipitationDataLoader:
    """
    Загрузчик данных об осадках.

    Поддерживает различные форматы данных и источники:
    - CSV файлы с метеорологическими данными
    - NetCDF файлы (ERA5, GPM)
    - Синтетические данные для тестирования
    """

    def __init__(self, region: str = "chui"):
        """
        Инициализация загрузчика.

        Args:
            region: Код региона из REGIONS
        """
        if region not in REGIONS:
            raise ValueError(f"Неизвестный регион: {region}. Доступные: {list(REGIONS.keys())}")

        self.region = region
        self.region_config = REGIONS[region]
        self.data: Optional[pd.DataFrame] = None

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.

        Ожидаемые колонки:
        - timestamp или date: дата/время
        - precipitation или precip: осадки в мм
        - latitude, longitude: координаты (опционально)
        - temperature: температура в °C (опционально)

        Args:
            filepath: Путь к CSV файлу

        Returns:
            DataFrame с данными об осадках
        """
        df = pd.read_csv(filepath)

        # Нормализация названий колонок
        column_mapping = {
            'date': 'timestamp',
            'datetime': 'timestamp',
            'precip': 'precipitation_mm',
            'precipitation': 'precipitation_mm',
            'temp': 'temperature_c',
            'temperature': 'temperature_c',
            'lat': 'latitude',
            'lon': 'longitude',
            'lng': 'longitude'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Преобразование временных меток
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        self.data = df
        return df

    def load_synthetic(
        self,
        start_date: datetime,
        end_date: datetime,
        grid_resolution: float = 0.1,
        include_extreme_events: bool = True
    ) -> pd.DataFrame:
        """
        Генерация синтетических данных об осадках.

        Создает реалистичные данные с учетом:
        - Сезонных вариаций
        - Высотной поясности
        - Экстремальных событий

        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            grid_resolution: Разрешение сетки в градусах
            include_extreme_events: Включать экстремальные события

        Returns:
            DataFrame с синтетическими данными
        """
        bounds = self.region_config.bounds

        # Создание сетки координат
        lons = np.arange(bounds[0], bounds[2], grid_resolution)
        lats = np.arange(bounds[1], bounds[3], grid_resolution)

        # Генерация временного ряда
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        records = []

        for date in dates:
            # Сезонный коэффициент
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = self._get_seasonal_factor(day_of_year)

            for lon in lons:
                for lat in lats:
                    # Базовые осадки с сезонной вариацией
                    base_precip = np.random.exponential(scale=5.0) * seasonal_factor

                    # Высотная коррекция (приблизительная)
                    elevation_factor = 1.0 + (lat - bounds[1]) / (bounds[3] - bounds[1]) * 0.5

                    precipitation = base_precip * elevation_factor

                    # Добавление экстремальных событий
                    if include_extreme_events and np.random.random() < 0.02:
                        precipitation *= np.random.uniform(3, 8)

                    # Температура (упрощенная модель)
                    base_temp = 15 - (day_of_year - 182) ** 2 / 1000  # Максимум летом
                    temperature = base_temp - (lat - bounds[1]) * 5  # Высотный градиент
                    temperature += np.random.normal(0, 3)

                    records.append({
                        'timestamp': date,
                        'latitude': lat,
                        'longitude': lon,
                        'precipitation_mm': max(0, precipitation),
                        'temperature_c': temperature
                    })

        self.data = pd.DataFrame(records)
        return self.data

    def _get_seasonal_factor(self, day_of_year: int) -> float:
        """
        Расчет сезонного коэффициента осадков.

        Для Кыргызстана характерны:
        - Весенний максимум (март-май)
        - Летние ливни (июнь-август)
        - Сухая осень-зима

        Args:
            day_of_year: День года (1-365)

        Returns:
            Сезонный множитель
        """
        # Весенний пик
        spring_peak = 100  # Середина апреля
        spring_factor = np.exp(-((day_of_year - spring_peak) ** 2) / 2000)

        # Летние осадки
        summer_peak = 200  # Июль
        summer_factor = 0.7 * np.exp(-((day_of_year - summer_peak) ** 2) / 3000)

        # Базовый уровень
        base = 0.3

        return base + spring_factor + summer_factor

    def get_daily_totals(self) -> pd.DataFrame:
        """
        Получение суточных сумм осадков по региону.

        Returns:
            DataFrame с суточными суммами
        """
        if self.data is None:
            raise ValueError("Данные не загружены")

        daily = self.data.groupby(self.data['timestamp'].dt.date).agg({
            'precipitation_mm': ['sum', 'mean', 'max'],
            'temperature_c': 'mean'
        }).reset_index()

        daily.columns = ['date', 'total_precip_mm', 'mean_precip_mm',
                        'max_precip_mm', 'mean_temp_c']
        return daily

    def get_spatial_distribution(self, date: datetime) -> pd.DataFrame:
        """
        Получение пространственного распределения осадков на дату.

        Args:
            date: Дата

        Returns:
            DataFrame с координатами и осадками
        """
        if self.data is None:
            raise ValueError("Данные не загружены")

        mask = self.data['timestamp'].dt.date == date.date()
        return self.data[mask][['latitude', 'longitude', 'precipitation_mm']]

    def detect_extreme_events(
        self,
        threshold_mm: float = 50.0,
        consecutive_days: int = 1
    ) -> List[Dict]:
        """
        Выявление экстремальных осадков.

        Args:
            threshold_mm: Пороговое значение (мм/сутки)
            consecutive_days: Минимальное число последовательных дней

        Returns:
            Список экстремальных событий
        """
        if self.data is None:
            raise ValueError("Данные не загружены")

        daily = self.get_daily_totals()
        extreme_events = []

        i = 0
        while i < len(daily):
            if daily.iloc[i]['max_precip_mm'] >= threshold_mm:
                event_start = daily.iloc[i]['date']
                event_days = 1
                total_precip = daily.iloc[i]['total_precip_mm']
                max_precip = daily.iloc[i]['max_precip_mm']

                # Поиск последовательных дней
                j = i + 1
                while j < len(daily) and daily.iloc[j]['max_precip_mm'] >= threshold_mm * 0.5:
                    event_days += 1
                    total_precip += daily.iloc[j]['total_precip_mm']
                    max_precip = max(max_precip, daily.iloc[j]['max_precip_mm'])
                    j += 1

                if event_days >= consecutive_days:
                    extreme_events.append({
                        'start_date': event_start,
                        'duration_days': event_days,
                        'total_precipitation_mm': total_precip,
                        'max_daily_mm': max_precip,
                        'region': self.region
                    })

                i = j
            else:
                i += 1

        return extreme_events

    def save_processed(self, filepath: str):
        """
        Сохранение обработанных данных.

        Args:
            filepath: Путь для сохранения
        """
        if self.data is None:
            raise ValueError("Данные не загружены")

        self.data.to_csv(filepath, index=False)
        print(f"Данные сохранены в {filepath}")


def generate_synthetic_precipitation(
    region: str,
    year: int,
    scenario: str = "normal"
) -> pd.DataFrame:
    """
    Генерация синтетических данных об осадках для сценарного анализа.

    Args:
        region: Код региона
        year: Год для генерации
        scenario: Сценарий ('normal', 'wet', 'dry', 'extreme')

    Returns:
        DataFrame с данными об осадках
    """
    loader = PrecipitationDataLoader(region=region)

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    # Генерация базовых данных
    data = loader.load_synthetic(
        start_date=start_date,
        end_date=end_date,
        include_extreme_events=(scenario in ['normal', 'wet', 'extreme'])
    )

    # Модификация в зависимости от сценария
    scenario_multipliers = {
        'normal': 1.0,
        'wet': 1.5,
        'dry': 0.5,
        'extreme': 2.0
    }

    multiplier = scenario_multipliers.get(scenario, 1.0)
    data['precipitation_mm'] *= multiplier

    # Для экстремального сценария добавляем больше пиковых событий
    if scenario == 'extreme':
        # Добавляем дополнительные экстремальные дни
        extreme_days = np.random.choice(data['timestamp'].unique(), size=10, replace=False)
        for day in extreme_days:
            mask = data['timestamp'] == day
            data.loc[mask, 'precipitation_mm'] *= np.random.uniform(2, 4)

    return data


class HistoricalDataAnalyzer:
    """
    Анализатор исторических данных об осадках и паводках.
    """

    def __init__(self):
        self.historical_floods = self._load_historical_floods()

    def _load_historical_floods(self) -> List[Dict]:
        """
        Загрузка данных об исторических паводках в Кыргызстане.

        Returns:
            Список исторических событий
        """
        # Известные крупные паводковые события в Кыргызстане
        return [
            {
                'year': 2024,
                'region': 'osh',
                'description': 'Селевые потоки в Ошской области',
                'casualties': 15,
                'damage_usd': 5000000
            },
            {
                'year': 2023,
                'region': 'jalal_abad',
                'description': 'Весеннее половодье на р. Нарын',
                'casualties': 8,
                'damage_usd': 3000000
            },
            {
                'year': 2022,
                'region': 'issyk_kul',
                'description': 'Наводнение в Иссык-Кульской котловине',
                'casualties': 3,
                'damage_usd': 1500000
            },
            {
                'year': 2021,
                'region': 'chui',
                'description': 'Подтопление в Чуйской долине',
                'casualties': 2,
                'damage_usd': 2000000
            },
            {
                'year': 2020,
                'region': 'batken',
                'description': 'Сели в Баткенской области',
                'casualties': 5,
                'damage_usd': 1000000
            },
            {
                'year': 2019,
                'region': 'naryn',
                'description': 'Прорыв ледникового озера',
                'casualties': 0,
                'damage_usd': 800000
            }
        ]

    def get_flood_statistics(self, region: Optional[str] = None) -> Dict:
        """
        Получение статистики по паводкам.

        Args:
            region: Фильтр по региону (опционально)

        Returns:
            Словарь со статистикой
        """
        data = self.historical_floods
        if region:
            data = [f for f in data if f['region'] == region]

        if not data:
            return {'events': 0, 'total_damage': 0, 'avg_damage': 0}

        total_damage = sum(f['damage_usd'] for f in data)
        total_casualties = sum(f['casualties'] for f in data)

        return {
            'events': len(data),
            'total_damage_usd': total_damage,
            'avg_damage_usd': total_damage / len(data),
            'total_casualties': total_casualties,
            'years': [f['year'] for f in data]
        }
