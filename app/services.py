"""
Сервисный слой: все кэшированные функции и бизнес-логика.
Все @st.cache_data / @st.cache_resource функции должны быть на уровне модуля.
"""

import threading
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import REGIONS
from src.models.flood_model import FloodModel
from src.models.ml_predictor import FloodMLPredictor, generate_training_data
from src.utils.helpers import generate_sample_data
from src.data.weather_api import (
    get_weather_data_for_model,
    KyrgyzstanWeatherLoader,
    WeatherAPIError,
)


# ===================== КЭШИРОВАННЫЕ ФУНКЦИИ =====================

@st.cache_data(ttl=3600)
def load_region_data(
    region: str, days: int, scenario: str, use_real_data: bool = False
) -> pd.DataFrame:
    """
    Загрузка/генерация данных для региона.

    Args:
        region: Код региона
        days: Количество дней
        scenario: Сценарий осадков (для синтетических данных)
        use_real_data: Использовать реальные данные из Open-Meteo API

    Returns:
        DataFrame с погодными данными
    """
    if use_real_data:
        try:
            df = get_weather_data_for_model(
                region=region,
                days=min(days, 16),  # API ограничивает прогноз до 16 дней
                use_real_data=True,
                use_cache=True,
            )
            # Если запрошено больше дней чем доступно в API, дополняем синтетическими
            if days > 16 and len(df) < days * 24:
                synthetic = generate_sample_data(region=region, days=days - 16, scenario=scenario)
                if not df.empty:
                    last_real_time = df['timestamp'].max()
                    time_offset = last_real_time - synthetic['timestamp'].min() + timedelta(hours=1)
                    synthetic['timestamp'] = synthetic['timestamp'] + time_offset
                df = pd.concat([df, synthetic], ignore_index=True)
            return df
        except WeatherAPIError as e:
            st.warning(f"Не удалось получить реальные данные: {e}. Используются синтетические данные.")
            return generate_sample_data(region=region, days=days, scenario=scenario)
        except Exception as e:
            st.warning(f"Ошибка загрузки данных: {e}. Используются синтетические данные.")
            return generate_sample_data(region=region, days=days, scenario=scenario)
    else:
        return generate_sample_data(region=region, days=days, scenario=scenario)


@st.cache_data(ttl=1800)
def get_current_weather(region: str) -> dict:
    """Получение текущей погоды для региона (кэш 30 мин)."""
    try:
        loader = KyrgyzstanWeatherLoader(region=region, use_cache=True)
        return loader.get_current_conditions()
    except Exception as e:
        return {'error': str(e)}


@st.cache_resource
def get_flood_model(region: str) -> FloodModel:
    """Получение экземпляра гидрологической модели (singleton per region)."""
    return FloodModel(region=region)


@st.cache_data(ttl=7200)
def generate_default_simulation_data(
    region: str = 'chui', days: int = 90, scenario: str = 'wet'
) -> dict:
    """
    Генерация демонстрационных данных симуляции при запуске.
    Используется для быстрого отображения результатов без ожидания.
    Кэшируется на 2 часа.
    """
    input_data = generate_sample_data(region=region, days=days, scenario=scenario, ensure_events=True)
    model = FloodModel(region=region)
    model.initialize_state(input_data['timestamp'].iloc[0])
    results = model.run_simulation(input_data)
    events = model.detect_flood_events(results, flood_threshold=100.0)
    return {
        'input_data': input_data,
        'results': results,
        'events': events,
        'model_params': model.params,
    }


@st.cache_resource
def get_ml_predictor(region: str) -> FloodMLPredictor:
    """
    Получение ML предиктора (singleton per region).

    Приоритет загрузки:
    1. models/flood_predictor_all_real.joblib  — обучен на всех регионах (scripts/train_model.py)
    2. models/flood_predictor_{region}_real.joblib — обучен на конкретном регионе
    3. models/flood_predictor_{region}.joblib  — старая синтетическая модель
    4. Обучение на лету (синтетические данные)
    """
    predictor = FloodMLPredictor(model_type='ensemble')
    try:
        candidates = [
            Path('models/flood_predictor_all_real.joblib'),
            Path(f'models/flood_predictor_{region}_real.joblib'),
            Path(f'models/flood_predictor_{region}.joblib'),
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is not None:
            predictor.load_model(str(model_path))
        else:
            # Обучение на лету если ни один файл не найден
            data = generate_training_data(region=region, days=365, include_floods=True)
            predictor.train(data, flood_threshold=100.0)
    except Exception as e:
        st.warning(f"Ошибка загрузки ML модели: {e}")
    return predictor


# ===================== ОБЫЧНЫЕ ФУНКЦИИ =====================

def run_simulation(model: FloodModel, data: pd.DataFrame) -> pd.DataFrame:
    """Запуск гидрологической симуляции."""
    model.initialize_state(data['timestamp'].iloc[0])
    results = model.run_simulation(data, dt_hours=1)
    return results


def train_ml_model_background(region: str = 'chui', days: int = 365) -> bool:
    """
    Фоновое обучение ML модели.
    Запускается в отдельном потоке при старте приложения.
    Сохраняет модель в models/flood_predictor_{region}.joblib.
    """
    try:
        predictor = FloodMLPredictor(model_type='ensemble')
        model_path = Path(f'models/flood_predictor_{region}.joblib')
        if not model_path.exists():
            training_data = generate_training_data(region=region, days=days, include_floods=True)
            predictor.train(training_data, flood_threshold=100.0)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            predictor.save_model(str(model_path))
            return True
        return False  # Модель уже существует
    except Exception as e:
        print(f"Ошибка фонового обучения ML модели: {e}")
        return False


def init_session_state() -> None:
    """
    Инициализация session_state для симуляции и ML.
    Вызывается из web_app.py ПОСЛЕ импорта этого модуля,
    поэтому все функции уже определены — NameError невозможен.
    """
    if 'simulation_results' not in st.session_state:
        try:
            default_data = generate_default_simulation_data(region='chui', days=91, scenario='wet')
            st.session_state.simulation_results = default_data['results']
            st.session_state.input_data = default_data['input_data']
            st.session_state.flood_events = default_data['events']
            st.session_state.data_loaded = True
        except Exception:
            st.session_state.simulation_results = None
            st.session_state.input_data = None
            st.session_state.flood_events = []
            st.session_state.data_loaded = False

    if 'ml_training_started' not in st.session_state:
        st.session_state.ml_training_started = False
        try:
            training_thread = threading.Thread(
                target=train_ml_model_background,
                args=('chui', 365),
                daemon=True,
            )
            training_thread.start()
            st.session_state.ml_training_started = True
        except Exception:
            pass  # Тихо игнорируем ошибки фонового обучения
