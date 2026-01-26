"""
Модуль интеграции с Open-Meteo API для получения реальных погодных данных.

Open-Meteo - бесплатный API без необходимости ключа.
Документация: https://open-meteo.com/

Функциональность:
- Получение прогноза погоды (до 16 дней)
- Получение исторических данных
- Кэширование запросов
- Валидация данных
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from ..config import REGIONS, RegionConfig, BASE_DIR


# Конфигурация API
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Директория для файлового кэша
CACHE_DIR = Path(BASE_DIR) / ".weather_cache"

# Время жизни кэша (в секундах)
CACHE_TTL_FORECAST = 3600  # 1 час для прогноза
CACHE_TTL_HISTORICAL = 86400 * 7  # 7 дней для исторических данных


@dataclass
class WeatherDataPoint:
    """Точка данных погоды."""
    timestamp: datetime
    precipitation_mm: float
    temperature_c: float
    snowfall_cm: Optional[float] = None
    snow_depth_m: Optional[float] = None
    humidity_percent: Optional[float] = None
    wind_speed_ms: Optional[float] = None


class WeatherAPIError(Exception):
    """Ошибка при работе с Weather API."""
    pass


class WeatherDataCache:
    """Файловый кэш для погодных данных."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, params: Dict) -> str:
        """Генерация ключа кэша из параметров запроса."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Путь к файлу кэша."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, params: Dict, ttl: int) -> Optional[Dict]:
        """
        Получение данных из кэша.

        Args:
            params: Параметры запроса
            ttl: Время жизни кэша в секундах

        Returns:
            Данные из кэша или None
        """
        cache_key = self._get_cache_key(params)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached['cached_at'])
            if (datetime.now() - cached_time).total_seconds() > ttl:
                cache_path.unlink()  # Удаление устаревшего кэша
                return None

            return cached['data']
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def set(self, params: Dict, data: Dict):
        """
        Сохранение данных в кэш.

        Args:
            params: Параметры запроса
            data: Данные для кэширования
        """
        cache_key = self._get_cache_key(params)
        cache_path = self._get_cache_path(cache_key)

        cached = {
            'cached_at': datetime.now().isoformat(),
            'params': params,
            'data': data
        }

        with open(cache_path, 'w') as f:
            json.dump(cached, f)

    def clear(self):
        """Очистка всего кэша."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


class OpenMeteoClient:
    """
    Клиент для работы с Open-Meteo API.

    Поддерживает:
    - Прогноз погоды (до 16 дней)
    - Исторические данные
    - Автоматическое кэширование
    """

    def __init__(self, use_cache: bool = True):
        """
        Инициализация клиента.

        Args:
            use_cache: Использовать кэширование
        """
        self.use_cache = use_cache
        self.cache = WeatherDataCache() if use_cache else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FloodPredictionSystem/1.0'
        })

    def _make_request(
        self,
        url: str,
        params: Dict,
        cache_ttl: int = CACHE_TTL_FORECAST
    ) -> Dict:
        """
        Выполнение HTTP запроса с кэшированием.

        Args:
            url: URL API
            params: Параметры запроса
            cache_ttl: Время жизни кэша

        Returns:
            JSON ответ API

        Raises:
            WeatherAPIError: При ошибке запроса
        """
        # Проверка кэша
        if self.use_cache and self.cache:
            cache_key_params = {'url': url, **params}
            cached = self.cache.get(cache_key_params, cache_ttl)
            if cached is not None:
                return cached

        # Выполнение запроса
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Проверка на ошибки API
            if 'error' in data:
                raise WeatherAPIError(f"API error: {data.get('reason', 'Unknown error')}")

            # Сохранение в кэш
            if self.use_cache and self.cache:
                self.cache.set(cache_key_params, data)

            return data

        except requests.exceptions.Timeout:
            raise WeatherAPIError("Request timeout - API not responding")
        except requests.exceptions.ConnectionError:
            raise WeatherAPIError("Connection error - check internet connection")
        except requests.exceptions.HTTPError as e:
            raise WeatherAPIError(f"HTTP error: {e}")
        except json.JSONDecodeError:
            raise WeatherAPIError("Invalid JSON response from API")

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7,
        hourly: bool = True
    ) -> pd.DataFrame:
        """
        Получение прогноза погоды.

        Args:
            latitude: Широта
            longitude: Долгота
            days: Число дней прогноза (1-16)
            hourly: Почасовые данные (True) или суточные (False)

        Returns:
            DataFrame с прогнозом
        """
        days = min(max(1, days), 16)  # Ограничение 1-16 дней

        params = {
            'latitude': latitude,
            'longitude': longitude,
            'forecast_days': days,
            'timezone': 'auto'
        }

        if hourly:
            params['hourly'] = ','.join([
                'temperature_2m',
                'precipitation',
                'snowfall',
                'snow_depth',
                'relative_humidity_2m',
                'wind_speed_10m'
            ])
        else:
            params['daily'] = ','.join([
                'temperature_2m_mean',
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'snowfall_sum'
            ])

        data = self._make_request(
            OPEN_METEO_FORECAST_URL,
            params,
            CACHE_TTL_FORECAST
        )

        return self._parse_response(data, hourly)

    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        hourly: bool = True
    ) -> pd.DataFrame:
        """
        Получение исторических данных.

        Args:
            latitude: Широта
            longitude: Долгота
            start_date: Начальная дата
            end_date: Конечная дата
            hourly: Почасовые данные (True) или суточные (False)

        Returns:
            DataFrame с историческими данными
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')

        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'timezone': 'auto'
        }

        if hourly:
            params['hourly'] = ','.join([
                'temperature_2m',
                'precipitation',
                'snowfall',
                'snow_depth',
                'relative_humidity_2m',
                'wind_speed_10m'
            ])
        else:
            params['daily'] = ','.join([
                'temperature_2m_mean',
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'snowfall_sum'
            ])

        data = self._make_request(
            OPEN_METEO_ARCHIVE_URL,
            params,
            CACHE_TTL_HISTORICAL
        )

        return self._parse_response(data, hourly)

    def _parse_response(self, data: Dict, hourly: bool) -> pd.DataFrame:
        """
        Парсинг ответа API в DataFrame.

        Args:
            data: JSON ответ API
            hourly: Почасовые данные

        Returns:
            DataFrame с данными
        """
        if hourly:
            hourly_data = data.get('hourly', {})

            if not hourly_data or 'time' not in hourly_data:
                return pd.DataFrame()

            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly_data['time']),
                'precipitation_mm': hourly_data.get('precipitation', []),
                'temperature_c': hourly_data.get('temperature_2m', []),
                'snowfall_cm': hourly_data.get('snowfall', []),
                'snow_depth_m': hourly_data.get('snow_depth', []),
                'humidity_percent': hourly_data.get('relative_humidity_2m', []),
                'wind_speed_ms': hourly_data.get('wind_speed_10m', [])
            })
        else:
            daily_data = data.get('daily', {})

            if not daily_data or 'time' not in daily_data:
                return pd.DataFrame()

            df = pd.DataFrame({
                'timestamp': pd.to_datetime(daily_data['time']),
                'precipitation_mm': daily_data.get('precipitation_sum', []),
                'temperature_c': daily_data.get('temperature_2m_mean', []),
                'temperature_max_c': daily_data.get('temperature_2m_max', []),
                'temperature_min_c': daily_data.get('temperature_2m_min', []),
                'snowfall_cm': daily_data.get('snowfall_sum', [])
            })

        return df


class KyrgyzstanWeatherLoader:
    """
    Загрузчик погодных данных для регионов Кыргызстана.

    Использует Open-Meteo API для получения реальных данных
    с автоматическим определением координат по региону.
    """

    def __init__(self, region: str = "chui", use_cache: bool = True):
        """
        Инициализация загрузчика.

        Args:
            region: Код региона из REGIONS
            use_cache: Использовать кэширование
        """
        if region not in REGIONS:
            raise ValueError(f"Неизвестный регион: {region}. Доступные: {list(REGIONS.keys())}")

        self.region = region
        self.region_config = REGIONS[region]
        self.client = OpenMeteoClient(use_cache=use_cache)

        # Координаты центра региона
        bounds = self.region_config.bounds
        self.latitude = (bounds[1] + bounds[3]) / 2
        self.longitude = (bounds[0] + bounds[2]) / 2

    def get_forecast(self, days: int = 7, hourly: bool = True) -> pd.DataFrame:
        """
        Получение прогноза погоды для региона.

        Args:
            days: Число дней прогноза (1-16)
            hourly: Почасовые данные

        Returns:
            DataFrame с прогнозом
        """
        return self.client.get_forecast(
            latitude=self.latitude,
            longitude=self.longitude,
            days=days,
            hourly=hourly
        )

    def get_historical(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        hourly: bool = True
    ) -> pd.DataFrame:
        """
        Получение исторических данных для региона.

        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            hourly: Почасовые данные

        Returns:
            DataFrame с историческими данными
        """
        return self.client.get_historical(
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=start_date,
            end_date=end_date,
            hourly=hourly
        )

    def get_current_conditions(self) -> Dict:
        """
        Получение текущих погодных условий.

        Returns:
            Словарь с текущими условиями
        """
        df = self.get_forecast(days=1, hourly=True)

        if df.empty:
            return {}

        # Найти ближайший час к текущему времени
        now = datetime.now()
        df['time_diff'] = abs(df['timestamp'] - now)
        current = df.loc[df['time_diff'].idxmin()]

        return {
            'timestamp': current['timestamp'],
            'temperature_c': current['temperature_c'],
            'precipitation_mm': current['precipitation_mm'],
            'humidity_percent': current.get('humidity_percent'),
            'wind_speed_ms': current.get('wind_speed_ms'),
            'region': self.region_config.name
        }


def validate_weather_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Валидация и очистка погодных данных.

    Args:
        df: DataFrame с погодными данными

    Returns:
        Кортеж (очищенный DataFrame, статистика валидации)
    """
    if df.empty:
        return df, {'is_valid': False, 'error': 'Empty DataFrame'}

    stats = {
        'is_valid': True,
        'total_rows': len(df),
        'missing_values': {},
        'filled_values': {},
        'warnings': []
    }

    df = df.copy()

    # Проверка обязательных колонок
    required_cols = ['timestamp', 'precipitation_mm', 'temperature_c']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        stats['is_valid'] = False
        stats['error'] = f"Missing required columns: {missing_cols}"
        return df, stats

    # Обработка пропущенных значений
    for col in df.columns:
        if col == 'timestamp':
            continue

        missing_count = df[col].isna().sum()
        if missing_count > 0:
            stats['missing_values'][col] = missing_count

            # Заполнение пропусков
            if col == 'precipitation_mm':
                df[col] = df[col].fillna(0)  # Нет данных = нет осадков
            elif col == 'temperature_c':
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].bfill().ffill()
            else:
                df[col] = df[col].fillna(0)

            stats['filled_values'][col] = missing_count

    # Проверка на отрицательные осадки
    if (df['precipitation_mm'] < 0).any():
        neg_count = (df['precipitation_mm'] < 0).sum()
        df.loc[df['precipitation_mm'] < 0, 'precipitation_mm'] = 0
        stats['warnings'].append(f"Fixed {neg_count} negative precipitation values")

    # Проверка экстремальных значений
    if df['precipitation_mm'].max() > 200:  # > 200 мм/час маловероятно
        stats['warnings'].append(
            f"Extreme precipitation detected: {df['precipitation_mm'].max():.1f} mm"
        )

    if df['temperature_c'].max() > 50 or df['temperature_c'].min() < -50:
        stats['warnings'].append(
            f"Extreme temperature detected: {df['temperature_c'].min():.1f} to {df['temperature_c'].max():.1f} C"
        )

    return df, stats


def get_weather_data_for_model(
    region: str = "chui",
    days: int = 30,
    use_real_data: bool = True,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Получение погодных данных для модели паводков.

    Функция для удобной интеграции с FloodModel.
    Возвращает DataFrame в формате, ожидаемом моделью.

    Args:
        region: Код региона
        days: Число дней данных
        use_real_data: Использовать реальные данные (True) или синтетические (False)
        use_cache: Использовать кэширование

    Returns:
        DataFrame с колонками: timestamp, precipitation_mm, temperature_c
    """
    if not use_real_data:
        # Используем существующий генератор синтетических данных
        from .precipitation import PrecipitationDataLoader
        loader = PrecipitationDataLoader(region=region)
        return loader.load_synthetic(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=days),
            grid_resolution=0.1
        )

    try:
        weather_loader = KyrgyzstanWeatherLoader(region=region, use_cache=use_cache)

        if days <= 16:
            # Используем прогноз
            df = weather_loader.get_forecast(days=days, hourly=True)
        else:
            # Комбинируем исторические данные и прогноз
            historical_days = days - 16
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=historical_days)

            df_historical = weather_loader.get_historical(
                start_date=start_date,
                end_date=end_date,
                hourly=True
            )

            df_forecast = weather_loader.get_forecast(days=16, hourly=True)

            df = pd.concat([df_historical, df_forecast], ignore_index=True)

        # Валидация
        df, validation_stats = validate_weather_data(df)

        if not validation_stats['is_valid']:
            raise WeatherAPIError(f"Data validation failed: {validation_stats.get('error')}")

        # Оставляем только необходимые колонки
        result_cols = ['timestamp', 'precipitation_mm', 'temperature_c']
        available_cols = [c for c in result_cols if c in df.columns]

        return df[available_cols]

    except WeatherAPIError as e:
        # При ошибке API возвращаем синтетические данные
        print(f"Warning: Weather API error ({e}), falling back to synthetic data")
        from .precipitation import PrecipitationDataLoader
        loader = PrecipitationDataLoader(region=region)
        return loader.load_synthetic(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=days),
            grid_resolution=0.1
        )


def get_multi_region_weather(
    regions: List[str] = None,
    days: int = 7,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Получение погодных данных для нескольких регионов.

    Args:
        regions: Список кодов регионов (по умолчанию - все)
        days: Число дней
        use_cache: Использовать кэширование

    Returns:
        Словарь {регион: DataFrame}
    """
    if regions is None:
        regions = list(REGIONS.keys())

    results = {}

    for region in regions:
        try:
            loader = KyrgyzstanWeatherLoader(region=region, use_cache=use_cache)
            df = loader.get_forecast(days=days, hourly=True)
            df, _ = validate_weather_data(df)
            results[region] = df
        except Exception as e:
            print(f"Warning: Failed to get data for {region}: {e}")
            results[region] = pd.DataFrame()

    return results


# Быстрые функции с кэшированием через lru_cache
@lru_cache(maxsize=32)
def _cached_region_center(region: str) -> Tuple[float, float]:
    """Кэшированное получение центра региона."""
    config = REGIONS[region]
    bounds = config.bounds
    return (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2


def clear_weather_cache():
    """Очистка кэша погодных данных."""
    cache = WeatherDataCache()
    cache.clear()
    _cached_region_center.cache_clear()
    print("Weather cache cleared")


if __name__ == "__main__":
    # Тестирование
    print("Testing Open-Meteo API integration...")

    # Тест прогноза
    loader = KyrgyzstanWeatherLoader(region="chui")

    print("\n1. Current conditions:")
    current = loader.get_current_conditions()
    for key, value in current.items():
        print(f"   {key}: {value}")

    print("\n2. 7-day forecast:")
    forecast = loader.get_forecast(days=7)
    print(forecast.head(10))

    print("\n3. Validation stats:")
    df, stats = validate_weather_data(forecast)
    print(f"   Valid: {stats['is_valid']}")
    print(f"   Rows: {stats['total_rows']}")
    print(f"   Warnings: {stats['warnings']}")

    print("\n4. Model-ready data:")
    model_data = get_weather_data_for_model(region="chui", days=7)
    print(model_data.head())
    print(f"   Columns: {list(model_data.columns)}")
