"""
Модуль предобработки данных для гидрологического моделирования.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from scipy import interpolate
from scipy.ndimage import gaussian_filter

from ..config import REGIONS


class DataPreprocessor:
    """
    Класс для предобработки гидрометеорологических данных.
    """

    def __init__(self):
        self.missing_value_threshold = 0.1  # 10% пропусков

    def validate_precipitation_data(self, df: pd.DataFrame) -> Dict:
        """
        Валидация данных об осадках.

        Args:
            df: DataFrame с данными

        Returns:
            Словарь с результатами валидации
        """
        results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }

        # Проверка обязательных колонок
        required_cols = ['timestamp', 'precipitation_mm']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            results['is_valid'] = False
            results['issues'].append(f"Отсутствуют колонки: {missing_cols}")

        # Проверка пропущенных значений
        missing_ratio = df['precipitation_mm'].isna().mean()
        results['statistics']['missing_ratio'] = missing_ratio
        if missing_ratio > self.missing_value_threshold:
            results['issues'].append(f"Слишком много пропусков: {missing_ratio:.1%}")

        # Проверка отрицательных значений
        negative_count = (df['precipitation_mm'] < 0).sum()
        if negative_count > 0:
            results['is_valid'] = False
            results['issues'].append(f"Найдены отрицательные значения: {negative_count}")

        # Проверка аномально высоких значений
        max_precip = df['precipitation_mm'].max()
        if max_precip > 500:  # мм/сутки
            results['issues'].append(f"Аномально высокие значения: {max_precip} мм")

        # Статистика
        results['statistics'].update({
            'count': len(df),
            'mean': df['precipitation_mm'].mean(),
            'max': max_precip,
            'std': df['precipitation_mm'].std()
        })

        return results

    def fill_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Заполнение пропущенных значений.

        Args:
            df: DataFrame с данными
            method: Метод заполнения ('interpolate', 'mean', 'zero')

        Returns:
            DataFrame с заполненными значениями
        """
        df = df.copy()

        if method == 'interpolate':
            # Линейная интерполяция
            df['precipitation_mm'] = df['precipitation_mm'].interpolate(
                method='linear',
                limit_direction='both'
            )
        elif method == 'mean':
            # Заполнение средним
            mean_val = df['precipitation_mm'].mean()
            df['precipitation_mm'] = df['precipitation_mm'].fillna(mean_val)
        elif method == 'zero':
            # Заполнение нулями
            df['precipitation_mm'] = df['precipitation_mm'].fillna(0)

        # Заполнение температуры если есть
        if 'temperature_c' in df.columns:
            df['temperature_c'] = df['temperature_c'].interpolate(
                method='linear',
                limit_direction='both'
            )

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str = 'precipitation_mm',
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Удаление выбросов из данных.

        Args:
            df: DataFrame с данными
            column: Колонка для обработки
            method: Метод ('iqr', 'zscore')
            threshold: Порог для определения выбросов

        Returns:
            DataFrame без выбросов
        """
        df = df.copy()

        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (df[column] >= lower) & (df[column] <= upper)
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            mask = z_scores < threshold
        else:
            return df

        return df[mask]

    def resample_temporal(
        self,
        df: pd.DataFrame,
        frequency: str = 'D',
        agg_method: str = 'sum'
    ) -> pd.DataFrame:
        """
        Ресемплирование временного ряда.

        Args:
            df: DataFrame с данными
            frequency: Частота ('H' - часы, 'D' - дни, 'W' - недели, 'M' - месяцы)
            agg_method: Метод агрегации ('sum', 'mean', 'max')

        Returns:
            Ресемплированный DataFrame
        """
        df = df.copy()
        df = df.set_index('timestamp')

        agg_funcs = {
            'precipitation_mm': agg_method,
        }

        if 'temperature_c' in df.columns:
            agg_funcs['temperature_c'] = 'mean'

        if 'latitude' in df.columns:
            agg_funcs['latitude'] = 'first'
            agg_funcs['longitude'] = 'first'

        resampled = df.resample(frequency).agg(agg_funcs)
        resampled = resampled.reset_index()

        return resampled

    def interpolate_spatial(
        self,
        df: pd.DataFrame,
        grid_resolution: float = 0.05,
        method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Пространственная интерполяция данных об осадках.

        Args:
            df: DataFrame с колонками latitude, longitude, precipitation_mm
            grid_resolution: Разрешение сетки в градусах
            method: Метод интерполяции ('linear', 'cubic', 'nearest')

        Returns:
            Кортеж (lon_grid, lat_grid, precip_grid)
        """
        # Извлечение координат и значений
        lons = df['longitude'].values
        lats = df['latitude'].values
        values = df['precipitation_mm'].values

        # Создание регулярной сетки
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()

        lon_grid = np.arange(lon_min, lon_max, grid_resolution)
        lat_grid = np.arange(lat_min, lat_max, grid_resolution)
        LON, LAT = np.meshgrid(lon_grid, lat_grid)

        # Интерполяция
        points = np.column_stack((lons, lats))
        precip_grid = interpolate.griddata(
            points,
            values,
            (LON, LAT),
            method=method,
            fill_value=0
        )

        return lon_grid, lat_grid, precip_grid

    def smooth_spatial(
        self,
        data: np.ndarray,
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Пространственное сглаживание данных.

        Args:
            data: 2D массив данных
            sigma: Параметр сглаживания

        Returns:
            Сглаженный массив
        """
        return gaussian_filter(data, sigma=sigma)

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Нормализация данных.

        Args:
            df: DataFrame с данными
            columns: Список колонок для нормализации
            method: Метод ('minmax', 'zscore')

        Returns:
            Нормализованный DataFrame
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-10)

        return df

    def aggregate_by_region(
        self,
        df: pd.DataFrame,
        region: str
    ) -> pd.DataFrame:
        """
        Агрегация данных по региону.

        Args:
            df: DataFrame с данными
            region: Код региона

        Returns:
            Агрегированный DataFrame
        """
        if region not in REGIONS:
            raise ValueError(f"Неизвестный регион: {region}")

        bounds = REGIONS[region].bounds

        # Фильтрация по границам
        mask = (
            (df['longitude'] >= bounds[0]) &
            (df['longitude'] <= bounds[2]) &
            (df['latitude'] >= bounds[1]) &
            (df['latitude'] <= bounds[3])
        )

        filtered = df[mask].copy()

        # Агрегация по времени
        if 'timestamp' in filtered.columns:
            aggregated = filtered.groupby('timestamp').agg({
                'precipitation_mm': ['sum', 'mean', 'max', 'min', 'std'],
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()

            aggregated.columns = [
                'timestamp', 'total_precip', 'mean_precip',
                'max_precip', 'min_precip', 'std_precip',
                'center_lat', 'center_lon'
            ]

            aggregated['region'] = region
            return aggregated

        return filtered

    def create_lag_features(
        self,
        df: pd.DataFrame,
        column: str = 'precipitation_mm',
        lags: List[int] = [1, 2, 3, 7]
    ) -> pd.DataFrame:
        """
        Создание лаговых признаков для временного ряда.

        Args:
            df: DataFrame с данными
            column: Колонка для создания лагов
            lags: Список лагов (дни)

        Returns:
            DataFrame с лаговыми признаками
        """
        df = df.copy()

        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)

        # Скользящие статистики
        for window in [3, 7, 14]:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window).max()
            df[f'{column}_rolling_sum_{window}'] = df[column].rolling(window).sum()

        return df

    def calculate_antecedent_precipitation(
        self,
        df: pd.DataFrame,
        days: int = 5,
        decay_factor: float = 0.85
    ) -> pd.DataFrame:
        """
        Расчет предшествующих осадков (API - Antecedent Precipitation Index).

        API_t = P_t + k * API_{t-1}

        Args:
            df: DataFrame с данными
            days: Число дней для расчета
            decay_factor: Коэффициент затухания

        Returns:
            DataFrame с индексом API
        """
        df = df.copy()

        # Расчет API
        api = np.zeros(len(df))
        precip = df['precipitation_mm'].values

        for i in range(len(df)):
            if i == 0:
                api[i] = precip[i]
            else:
                api[i] = precip[i] + decay_factor * api[i-1]

        df['antecedent_precipitation_index'] = api

        return df
