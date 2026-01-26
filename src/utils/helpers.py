"""
Вспомогательные функции для проекта.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

from ..config import RESULTS_DIR, REGIONS


def create_output_directory(name: str = None) -> str:
    """
    Создание директории для результатов.

    Args:
        name: Название директории (по умолчанию - текущая дата/время)

    Returns:
        Путь к созданной директории
    """
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    # Создание поддиректорий
    subdirs = ['figures', 'data', 'maps', 'reports']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    return output_dir


def save_results(
    results: pd.DataFrame,
    output_dir: str,
    filename: str = "simulation_results",
    formats: List[str] = ['csv', 'json']
):
    """
    Сохранение результатов моделирования.

    Args:
        results: DataFrame с результатами
        output_dir: Директория для сохранения
        filename: Базовое имя файла
        formats: Список форматов для сохранения
    """
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if 'csv' in formats:
        path = os.path.join(data_dir, f"{filename}.csv")
        results.to_csv(path, index=False)
        print(f"Сохранено: {path}")

    if 'json' in formats:
        path = os.path.join(data_dir, f"{filename}.json")
        # Преобразование datetime для JSON
        results_copy = results.copy()
        for col in results_copy.columns:
            if results_copy[col].dtype == 'datetime64[ns]':
                results_copy[col] = results_copy[col].astype(str)
        results_copy.to_json(path, orient='records', indent=2)
        print(f"Сохранено: {path}")

    if 'excel' in formats:
        path = os.path.join(data_dir, f"{filename}.xlsx")
        results.to_excel(path, index=False)
        print(f"Сохранено: {path}")


def load_results(filepath: str) -> pd.DataFrame:
    """
    Загрузка сохраненных результатов.

    Args:
        filepath: Путь к файлу

    Returns:
        DataFrame с результатами
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    # Преобразование временных меток
    for col in df.columns:
        if 'timestamp' in col.lower() or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col])

    return df


def calculate_statistics(
    results: pd.DataFrame,
    variables: List[str] = None
) -> Dict:
    """
    Расчет статистики по результатам.

    Args:
        results: DataFrame с результатами
        variables: Список переменных для анализа

    Returns:
        Словарь со статистикой
    """
    if variables is None:
        # Выбираем числовые колонки
        variables = results.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for var in variables:
        if var not in results.columns:
            continue

        data = results[var].dropna()

        stats[var] = {
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(data.median()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'sum': float(data.sum())
        }

    # Общая статистика
    if 'timestamp' in results.columns:
        stats['period'] = {
            'start': str(results['timestamp'].min()),
            'end': str(results['timestamp'].max()),
            'days': len(results)
        }

    return stats


def format_report(
    stats: Dict,
    events: List = None,
    region: str = None
) -> str:
    """
    Форматирование отчета.

    Args:
        stats: Статистика
        events: Паводковые события
        region: Код региона

    Returns:
        Текст отчета
    """
    report = []
    report.append("=" * 60)
    report.append("ОТЧЕТ О РЕЗУЛЬТАТАХ МОДЕЛИРОВАНИЯ ПАВОДКОВ")
    report.append("=" * 60)
    report.append("")

    # Информация о регионе
    if region and region in REGIONS:
        config = REGIONS[region]
        report.append(f"Регион: {config.name}")
        report.append(f"Площадь: {config.area_km2:,} км²")
        report.append(f"Средняя высота: {config.avg_elevation} м")
        report.append("")

    # Период моделирования
    if 'period' in stats:
        report.append("ПЕРИОД МОДЕЛИРОВАНИЯ")
        report.append("-" * 40)
        report.append(f"  Начало: {stats['period']['start']}")
        report.append(f"  Конец: {stats['period']['end']}")
        report.append(f"  Дней: {stats['period']['days']}")
        report.append("")

    # Статистика по осадкам
    if 'precipitation_mm' in stats:
        p = stats['precipitation_mm']
        report.append("ОСАДКИ")
        report.append("-" * 40)
        report.append(f"  Сумма: {p['sum']:.1f} мм")
        report.append(f"  Среднее: {p['mean']:.2f} мм/день")
        report.append(f"  Максимум: {p['max']:.1f} мм/день")
        report.append(f"  Стд. откл.: {p['std']:.2f} мм")
        report.append("")

    # Статистика по расходу
    if 'discharge_m3s' in stats:
        d = stats['discharge_m3s']
        report.append("РАСХОД ВОДЫ")
        report.append("-" * 40)
        report.append(f"  Максимум: {d['max']:.1f} м³/с")
        report.append(f"  Среднее: {d['mean']:.1f} м³/с")
        report.append(f"  Медиана: {d['median']:.1f} м³/с")
        report.append("")

    # Паводковые события
    if events:
        report.append("ПАВОДКОВЫЕ СОБЫТИЯ")
        report.append("-" * 40)
        report.append(f"  Всего событий: {len(events)}")

        if events:
            peak_discharges = [e.peak_discharge_m3s for e in events]
            durations = [e.duration_hours for e in events]

            report.append(f"  Макс. пиковый расход: {max(peak_discharges):.1f} м³/с")
            report.append(f"  Средняя продолжительность: {np.mean(durations):.1f} ч")

            # Распределение по уровням риска
            risk_counts = {}
            for e in events:
                risk = e.risk_level.value
                risk_counts[risk] = risk_counts.get(risk, 0) + 1

            report.append("  Распределение по уровням риска:")
            for risk, count in risk_counts.items():
                report.append(f"    - {risk}: {count}")
        report.append("")

    report.append("=" * 60)
    report.append(f"Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    return "\n".join(report)


def validate_input_data(
    precipitation: pd.DataFrame,
    required_columns: List[str] = None
) -> Dict:
    """
    Валидация входных данных.

    Args:
        precipitation: DataFrame с данными
        required_columns: Обязательные колонки

    Returns:
        Результаты валидации
    """
    if required_columns is None:
        required_columns = ['timestamp', 'precipitation_mm']

    result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    # Проверка колонок
    missing = [c for c in required_columns if c not in precipitation.columns]
    if missing:
        result['is_valid'] = False
        result['errors'].append(f"Отсутствуют колонки: {missing}")

    # Проверка пропусков
    for col in required_columns:
        if col in precipitation.columns:
            missing_ratio = precipitation[col].isna().mean()
            if missing_ratio > 0.1:
                result['warnings'].append(
                    f"Колонка {col}: {missing_ratio:.1%} пропущенных значений"
                )

    # Проверка отрицательных значений
    if 'precipitation_mm' in precipitation.columns:
        neg_count = (precipitation['precipitation_mm'] < 0).sum()
        if neg_count > 0:
            result['is_valid'] = False
            result['errors'].append(
                f"Найдены отрицательные значения осадков: {neg_count}"
            )

    # Проверка хронологии
    if 'timestamp' in precipitation.columns:
        timestamps = pd.to_datetime(precipitation['timestamp'])
        if not timestamps.is_monotonic_increasing:
            result['warnings'].append("Временной ряд не отсортирован")

    return result


def generate_sample_data(
    region: str = "chui",
    days: int = 365,
    scenario: str = "normal"
) -> pd.DataFrame:
    """
    Генерация тестовых данных для демонстрации.

    Args:
        region: Код региона
        days: Число дней
        scenario: Сценарий ('normal', 'wet', 'dry', 'extreme')

    Returns:
        DataFrame с тестовыми данными
    """
    from datetime import timedelta

    np.random.seed(42)

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # Базовые осадки с сезонной вариацией
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])

    # Сезонный паттерн для Кыргызстана
    seasonal = (
        np.sin((day_of_year - 60) * 2 * np.pi / 365) * 5 +  # Весенний максимум
        np.sin((day_of_year - 200) * 2 * np.pi / 365) * 3    # Летний пик
    )
    seasonal = np.maximum(seasonal, 0)

    # Случайные осадки
    precip = np.random.exponential(scale=3, size=days) + seasonal
    precip = np.maximum(precip, 0)

    # Добавление экстремальных событий
    if scenario in ['normal', 'wet', 'extreme']:
        extreme_days = np.random.choice(days, size=int(days * 0.02), replace=False)
        precip[extreme_days] *= np.random.uniform(3, 6, size=len(extreme_days))

    # Модификация по сценарию
    multipliers = {'normal': 1.0, 'wet': 1.5, 'dry': 0.5, 'extreme': 2.0}
    precip *= multipliers.get(scenario, 1.0)

    # Температура
    base_temp = 10 + 15 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
    temp = base_temp + np.random.normal(0, 3, size=days)

    # Создание DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'precipitation_mm': precip,
        'temperature_c': temp
    })

    return df
