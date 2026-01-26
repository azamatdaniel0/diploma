"""
Веб-интерфейс для системы прогнозирования паводков в Кыргызстане.

Запуск: streamlit run web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import json
from pathlib import Path

# Импорт модулей проекта
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import REGIONS, RegionConfig
from src.models.flood_model import FloodModel, FloodRiskLevel
from src.models.ml_predictor import FloodMLPredictor, generate_training_data
from src.utils.helpers import generate_sample_data
from src.data.weather_api import (
    get_weather_data_for_model,
    KyrgyzstanWeatherLoader,
    WeatherAPIError,
    clear_weather_cache
)
from src.visualization.enhanced_charts import (
    create_animated_hydrograph,
    create_calendar_heatmap,
    create_scenario_comparison,
    create_enhanced_hydrograph,
    create_risk_distribution_chart,
    create_correlation_matrix,
    get_chart_download_config,
    create_gauge_chart
)


# Конфигурация страницы
st.set_page_config(
    page_title="Прогноз паводков - Кыргызстан",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== ТЕМА И СОСТОЯНИЕ СЕССИИ =====================
# Инициализация темы в session_state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    """Переключение темы."""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Определение цветов темы
THEMES = {
    'light': {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8f9fa',
        'bg_card': '#ffffff',
        'text_primary': '#1a1a2e',
        'text_secondary': '#4a4a6a',
        'accent': '#1E88E5',
        'accent_hover': '#1565C0',
        'border': '#e0e0e0',
        'shadow': 'rgba(0, 0, 0, 0.08)',
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2',
        'text_on_accent': '#ffffff',
    },
    'dark': {
        'bg_primary': '#0e1117',
        'bg_secondary': '#1a1a2e',
        'bg_card': '#16213e',
        'text_primary': '#e8e8e8',
        'text_secondary': '#a0a0b0',
        'accent': '#4fc3f7',
        'accent_hover': '#29b6f6',
        'border': '#2d2d44',
        'shadow': 'rgba(0, 0, 0, 0.3)',
        'gradient_start': '#667eea',
        'gradient_end': '#764ba2',
        'text_on_accent': '#ffffff',
    }
}

current_theme = THEMES[st.session_state.theme]

# CSS стили с поддержкой тем и адаптивности
st.markdown(f"""
<style>
    /* ===================== БАЗОВЫЕ ПЕРЕМЕННЫЕ ===================== */
    :root {{
        --bg-primary: {current_theme['bg_primary']};
        --bg-secondary: {current_theme['bg_secondary']};
        --bg-card: {current_theme['bg_card']};
        --text-primary: {current_theme['text_primary']};
        --text-secondary: {current_theme['text_secondary']};
        --accent: {current_theme['accent']};
        --accent-hover: {current_theme['accent_hover']};
        --border: {current_theme['border']};
        --shadow: {current_theme['shadow']};
        --gradient-start: {current_theme['gradient_start']};
        --gradient-end: {current_theme['gradient_end']};
        --text-on-accent: {current_theme['text_on_accent']};
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* ===================== ОСНОВНЫЕ СТИЛИ ===================== */
    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: var(--transition);
    }}

    /* ===================== ЗАГОЛОВКИ ===================== */
    .main-header {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }}

    .section-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent);
        display: inline-block;
    }}

    /* ===================== КАРТОЧКИ ===================== */
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px var(--shadow);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }}

    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
    }}

    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 30px var(--shadow);
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0.5rem 0;
    }}

    .metric-label {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* ===================== ИНДИКАТОРЫ РИСКА ===================== */
    .risk-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: var(--transition);
    }}

    .risk-low {{
        background: linear-gradient(135deg, #4CAF50, #66BB6A);
        color: var(--text-on-accent);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }}
    .risk-moderate {{
        background: linear-gradient(135deg, #FFC107, #FFD54F);
        color: #1a1a2e;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.4);
    }}
    .risk-high {{
        background: linear-gradient(135deg, #FF9800, #FFB74D);
        color: var(--text-on-accent);
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.4);
    }}
    .risk-critical {{
        background: linear-gradient(135deg, #F44336, #E57373);
        color: var(--text-on-accent);
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.4);
        animation: pulse-critical 2s infinite;
    }}

    @keyframes pulse-critical {{
        0%, 100% {{ box-shadow: 0 4px 15px rgba(244, 67, 54, 0.4); }}
        50% {{ box-shadow: 0 4px 25px rgba(244, 67, 54, 0.7); }}
    }}

    /* ===================== ВКЛАДКИ ===================== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 48px;
        padding: 0 24px;
        border-radius: 8px;
        font-weight: 500;
        color: var(--text-secondary);
        background: transparent;
        transition: var(--transition);
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: var(--bg-card);
        color: var(--text-primary);
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)) !important;
        color: var(--text-on-accent) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }}

    /* ===================== БОКОВАЯ ПАНЕЛЬ ===================== */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-primary);
    }}

    .sidebar-header {{
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent);
    }}

    .sidebar-section {{
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
        transition: var(--transition);
    }}

    .sidebar-section:hover {{
        border-color: var(--accent);
    }}

    /* ===================== КНОПКИ ===================== */
    .stButton > button {{
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        color: var(--text-on-accent);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: var(--transition);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Кнопка переключения темы */
    .theme-toggle {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 50px;
        padding: 0.5rem 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
        transition: var(--transition);
        font-size: 0.9rem;
        color: var(--text-primary);
    }}

    .theme-toggle:hover {{
        background: var(--accent);
        color: var(--text-on-accent);
        border-color: var(--accent);
    }}

    /* ===================== ПОЛЯ ВВОДА ===================== */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        transition: var(--transition);
    }}

    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {{
        border-color: var(--accent);
    }}

    /* ===================== МЕТРИКИ STREAMLIT ===================== */
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent);
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.9rem;
        color: var(--text-secondary);
    }}

    [data-testid="stMetricDelta"] svg {{
        display: none;
    }}

    /* ===================== EXPANDER ===================== */
    .streamlit-expanderHeader {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        transition: var(--transition);
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--accent);
        background: var(--bg-secondary);
    }}

    .streamlit-expanderContent {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 1rem;
    }}

    /* ===================== DATAFRAMES ===================== */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px var(--shadow);
    }}

    /* ===================== СПИННЕРЫ И ПРОГРЕСС ===================== */
    .stSpinner > div {{
        border-color: var(--accent);
    }}

    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
    }}

    /* ===================== ПОДСКАЗКИ (TOOLTIPS) ===================== */
    .tooltip {{
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
    }}

    .tooltip .tooltip-text {{
        visibility: hidden;
        opacity: 0;
        width: 250px;
        background: var(--bg-card);
        color: var(--text-primary);
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        box-shadow: 0 8px 30px var(--shadow);
        border: 1px solid var(--border);
        font-size: 0.85rem;
        line-height: 1.4;
        transition: opacity 0.3s, visibility 0.3s;
    }}

    .tooltip .tooltip-text::after {{
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -8px;
        border-width: 8px;
        border-style: solid;
        border-color: var(--bg-card) transparent transparent transparent;
    }}

    .tooltip:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}

    .tooltip-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 50%;
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin-left: 6px;
        transition: var(--transition);
    }}

    .tooltip:hover .tooltip-icon {{
        background: var(--accent);
        color: var(--text-on-accent);
        border-color: var(--accent);
    }}

    /* ===================== ФУТЕР ===================== */
    .footer {{
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }}

    .footer a {{
        color: var(--accent);
        text-decoration: none;
        transition: var(--transition);
    }}

    .footer a:hover {{
        color: var(--accent-hover);
    }}

    /* ===================== АЛЕРТЫ И СООБЩЕНИЯ ===================== */
    .stAlert {{
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px var(--shadow);
    }}

    /* ===================== АНИМАЦИИ ЗАГРУЗКИ ===================== */
    .loading-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        gap: 1rem;
    }}

    .loading-spinner {{
        width: 48px;
        height: 48px;
        border: 4px solid var(--border);
        border-top-color: var(--accent);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}

    .loading-text {{
        color: var(--text-secondary);
        font-size: 0.95rem;
        animation: pulse 1.5s ease-in-out infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    /* ===================== АДАПТИВНЫЕ СТИЛИ ===================== */

    /* Планшеты */
    @media (max-width: 992px) {{
        .main-header {{
            font-size: 2rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            padding: 0 16px;
            font-size: 0.9rem;
        }}

        .metric-card {{
            padding: 1rem;
        }}

        .metric-value {{
            font-size: 1.6rem;
        }}
    }}

    /* Мобильные устройства */
    @media (max-width: 768px) {{
        .main-header {{
            font-size: 1.6rem;
            padding: 0.75rem 0;
        }}

        .section-header {{
            font-size: 1.2rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            flex-wrap: wrap;
            gap: 4px;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 40px;
            padding: 0 12px;
            font-size: 0.8rem;
            flex: 1 1 auto;
            min-width: 80px;
            justify-content: center;
        }}

        .metric-card {{
            padding: 0.75rem;
            border-radius: 12px;
        }}

        .metric-value {{
            font-size: 1.4rem;
        }}

        .metric-label {{
            font-size: 0.75rem;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.4rem;
        }}

        .tooltip .tooltip-text {{
            width: 200px;
            font-size: 0.8rem;
        }}

        .sidebar-section {{
            padding: 0.75rem;
        }}

        /* Скрыть некоторые элементы на мобильных */
        .hide-mobile {{
            display: none !important;
        }}
    }}

    /* Маленькие мобильные */
    @media (max-width: 480px) {{
        .main-header {{
            font-size: 1.3rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 36px;
            padding: 0 8px;
            font-size: 0.75rem;
        }}

        .metric-value {{
            font-size: 1.2rem;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1.2rem;
        }}

        .risk-badge {{
            padding: 0.4rem 0.8rem;
            font-size: 0.75rem;
        }}
    }}

    /* Поддержка тёмной темы системы */
    @media (prefers-color-scheme: dark) {{
        .auto-theme {{
            --bg-primary: #0e1117;
            --bg-secondary: #1a1a2e;
            --text-primary: #e8e8e8;
        }}
    }}

    /* Анимация появления элементов */
    .fade-in {{
        animation: fadeIn 0.5s ease-out;
    }}

    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* Улучшенная прокрутка */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent);
    }}
</style>
""", unsafe_allow_html=True)


# Вспомогательные функции для UI компонентов
def render_tooltip(text: str, tooltip: str) -> str:
    """Рендеринг текста с подсказкой."""
    return f'''
    <span class="tooltip">
        {text}
        <span class="tooltip-icon">?</span>
        <span class="tooltip-text">{tooltip}</span>
    </span>
    '''

def render_metric_card(icon: str, label: str, value: str, tooltip: str = None) -> str:
    """Рендеринг метрики в виде карточки."""
    tooltip_html = f'<span class="tooltip-icon" title="{tooltip}">?</span>' if tooltip else ''
    return f'''
    <div class="metric-card fade-in">
        <div class="metric-label">{icon} {label} {tooltip_html}</div>
        <div class="metric-value">{value}</div>
    </div>
    '''

def render_risk_badge(risk: str) -> str:
    """Рендеринг индикатора риска."""
    labels = {
        'low': ('Низкий', ''),
        'moderate': ('Умеренный', ''),
        'high': ('Высокий', ''),
        'critical': ('Критический', '')
    }
    label, icon = labels.get(risk, (risk, ''))
    return f'<span class="risk-badge risk-{risk}">{icon} {label}</span>'

def show_loading_animation(message: str = "Загрузка..."):
    """Показать анимацию загрузки."""
    return f'''
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
    </div>
    '''


# Кэширование данных
@st.cache_data(ttl=3600)
def load_region_data(region: str, days: int, scenario: str, use_real_data: bool = False) -> pd.DataFrame:
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
            # Используем реальные данные из Open-Meteo API
            df = get_weather_data_for_model(
                region=region,
                days=min(days, 16),  # API ограничивает прогноз до 16 дней
                use_real_data=True,
                use_cache=True
            )

            # Если запрошено больше дней чем доступно в API, дополняем синтетическими
            if days > 16 and len(df) < days * 24:
                synthetic = generate_sample_data(region=region, days=days-16, scenario=scenario)
                # Сдвигаем даты синтетических данных
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
    """Получение текущей погоды для региона."""
    try:
        loader = KyrgyzstanWeatherLoader(region=region, use_cache=True)
        return loader.get_current_conditions()
    except Exception as e:
        return {'error': str(e)}


@st.cache_resource
def get_flood_model(region: str) -> FloodModel:
    """Получение модели паводков."""
    return FloodModel(region=region)


@st.cache_resource
def get_ml_predictor(region: str) -> FloodMLPredictor:
    """Получение ML предиктора."""
    predictor = FloodMLPredictor(model_type='ensemble')
    try:
        model_path = Path(f'models/flood_predictor_{region}.joblib')
        if model_path.exists():
            predictor.load_model(str(model_path))
        else:
            # Обучение на лету
            data = generate_training_data(region=region, days=365, include_floods=True)
            predictor.train(data, flood_threshold=100.0)
    except Exception as e:
        st.warning(f"Ошибка загрузки ML модели: {e}")
    return predictor


def run_simulation(model: FloodModel, data: pd.DataFrame) -> pd.DataFrame:
    """Запуск симуляции."""
    model.initialize_state(data['timestamp'].iloc[0])
    results = model.run_simulation(data, dt_hours=1)
    return results


def get_risk_color(risk: str) -> str:
    """Цвет для уровня риска."""
    colors = {
        'low': '#4CAF50',
        'moderate': '#FFC107',
        'high': '#FF9800',
        'critical': '#F44336'
    }
    return colors.get(risk, '#9E9E9E')


def get_risk_label(risk: str) -> str:
    """Русский текст для уровня риска."""
    labels = {
        'low': 'Низкий',
        'moderate': 'Умеренный',
        'high': 'Высокий',
        'critical': 'Критический'
    }
    return labels.get(risk, risk)


# ===================== БОКОВАЯ ПАНЕЛЬ =====================

# Логотип и заголовок
st.sidebar.markdown('''
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌊</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary);">FloodPredict KG</div>
    <div style="font-size: 0.75rem; color: var(--text-secondary);">Система раннего предупреждения</div>
</div>
''', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Переключатель темы
theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
theme_label = "Тёмная тема" if st.session_state.theme == 'light' else "Светлая тема"

col_theme1, col_theme2 = st.sidebar.columns([1, 3])
with col_theme1:
    st.markdown(f"<div style='font-size: 1.5rem; text-align: center;'>{theme_icon}</div>", unsafe_allow_html=True)
with col_theme2:
    if st.button(theme_label, key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()

st.sidebar.markdown("---")

# Секция выбора региона
st.sidebar.markdown('''
<div class="sidebar-section">
    <div class="sidebar-header">🌍 Регион мониторинга</div>
</div>
''', unsafe_allow_html=True)

region_names = {code: cfg.name for code, cfg in REGIONS.items()}
selected_region = st.sidebar.selectbox(
    "Выберите регион",
    options=list(region_names.keys()),
    format_func=lambda x: region_names[x],
    index=0,
    help="Выберите регион Кыргызстана для анализа паводковой обстановки"
)

# Информация о выбранном регионе
region_info = REGIONS[selected_region]
st.sidebar.markdown(f'''
<div style="background: var(--bg-card); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; border: 1px solid var(--border);">
    <div style="font-size: 0.8rem; color: var(--text-secondary);">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>Площадь:</span>
            <span style="color: var(--accent); font-weight: 600;">{region_info.area_km2:,} км²</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>Высота:</span>
            <span style="color: var(--accent); font-weight: 600;">{region_info.avg_elevation} м</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Реки:</span>
            <span style="color: var(--accent); font-weight: 600;">{len(region_info.main_rivers)}</span>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Параметры симуляции
st.sidebar.markdown('''
<div class="sidebar-section">
    <div class="sidebar-header">📊 Параметры симуляции</div>
</div>
''', unsafe_allow_html=True)

scenario = st.sidebar.selectbox(
    "Сценарий осадков",
    options=['normal', 'wet', 'dry', 'extreme'],
    format_func=lambda x: {
        'normal': '🌤️ Нормальный',
        'wet': '🌧️ Влажный (+50%)',
        'dry': '☀️ Засушливый (-50%)',
        'extreme': '⛈️ Экстремальный (+100%)'
    }[x],
    help="Сценарий определяет интенсивность осадков в симуляции"
)

simulation_days = st.sidebar.slider(
    "Период симуляции (дней)",
    min_value=7,
    max_value=365,
    value=30,
    step=7,
    help="Количество дней для моделирования гидрологической обстановки"
)

# Визуальный индикатор периода
period_percent = (simulation_days - 7) / (365 - 7) * 100
st.sidebar.markdown(f'''
<div style="background: var(--bg-secondary); border-radius: 4px; height: 6px; margin: 0.5rem 0;">
    <div style="background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
                border-radius: 4px; height: 100%; width: {period_percent}%; transition: width 0.3s;"></div>
</div>
''', unsafe_allow_html=True)

# Порог паводка с подсказкой
st.sidebar.markdown(f'''
{render_tooltip("🌊 Порог паводка", "Критический уровень расхода воды, при превышении которого объявляется паводковая опасность")}
''', unsafe_allow_html=True)

flood_threshold = st.sidebar.number_input(
    "Порог (м³/с)",
    min_value=10.0,
    max_value=500.0,
    value=100.0,
    step=10.0,
    help="Пороговое значение расхода воды для определения паводковой ситуации"
)

st.sidebar.markdown("---")

# Источник данных
st.sidebar.markdown('''
<div class="sidebar-section">
    <div class="sidebar-header">📡 Источник данных</div>
</div>
''', unsafe_allow_html=True)

use_real_data = st.sidebar.toggle(
    "Реальные данные (Open-Meteo)",
    value=False,
    help="Использовать реальные метеоданные из Open-Meteo API. При отключении используются синтетические данные."
)

if use_real_data:
    st.sidebar.markdown('''
    <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
                border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; border: 1px solid var(--border);">
        <div style="font-size: 0.8rem; color: var(--text-secondary);">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">✓</span>
                <span>Прогноз до 16 дней</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">✓</span>
                <span>Почасовые данные</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #2196F3;">ℹ</span>
                <span>API: Open-Meteo (бесплатно)</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Показать текущую погоду
    current_weather = get_current_weather(selected_region)
    if current_weather and 'error' not in current_weather:
        temp_val = current_weather.get('temperature_c', 'N/A')
        precip_val = current_weather.get('precipitation_mm', 0)
        temp_str = f"{temp_val:.1f}" if isinstance(temp_val, (int, float)) else "N/A"
        precip_str = f"{precip_val:.1f}" if isinstance(precip_val, (int, float)) else "0.0"
        st.sidebar.markdown(f'''
        <div style="background: var(--bg-card); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; border: 1px solid var(--border);">
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Текущая погода</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.5rem;">🌡️</span>
                <span style="font-size: 1.2rem; font-weight: 600; color: var(--accent);">{temp_str}°C</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.25rem;">
                <span style="font-size: 1rem;">💧</span>
                <span style="font-size: 0.9rem; color: var(--text-primary);">{precip_str} мм</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
else:
    st.sidebar.markdown('''
    <div style="background: var(--bg-secondary); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; border: 1px solid var(--border);">
        <div style="font-size: 0.8rem; color: var(--text-secondary);">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>📊</span>
                <span>Синтетические данные для тестирования</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Кнопка запуска с прогресс-индикатором
run_simulation_btn = st.sidebar.button(
    "▶️ Запустить симуляцию",
    type="primary",
    use_container_width=True,
    help="Запустить гидрологическое моделирование с выбранными параметрами"
)

st.sidebar.markdown("---")

# ML секция
st.sidebar.markdown('''
<div class="sidebar-section">
    <div class="sidebar-header">🤖 Машинное обучение</div>
</div>
''', unsafe_allow_html=True)

enable_ml = st.sidebar.checkbox(
    "Включить ML прогноз",
    value=True,
    help="Использовать ансамбль моделей машинного обучения для прогнозирования"
)

if enable_ml:
    st.sidebar.markdown('''
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; border: 1px solid var(--border);">
        <div style="font-size: 0.8rem; color: var(--text-secondary);">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">●</span>
                <span>Random Forest</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">●</span>
                <span>Gradient Boosting</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #4CAF50;">●</span>
                <span>Neural Network</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.sidebar.markdown("---")

# О системе
st.sidebar.markdown('''
<div class="sidebar-section">
    <div class="sidebar-header">📚 О системе</div>
    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin-top: 0.5rem;">
        <p style="margin-bottom: 0.75rem;">
            Интеллектуальная система прогнозирования паводков для Кыргызстана с использованием
            гидрологических моделей и машинного обучения.
        </p>
        <div style="margin-top: 1rem;">
            <strong style="color: var(--text-primary);">Методы анализа:</strong>
            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                <li>SCS-CN модель поверхностного стока</li>
                <li>Degree-day модель таяния снега</li>
                <li>Ансамбль ML моделей</li>
            </ul>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Версия и статус
st.sidebar.markdown('''
<div style="text-align: center; padding: 1rem 0; margin-top: 1rem;">
    <div style="display: inline-flex; align-items: center; gap: 0.5rem;
                background: var(--bg-card); padding: 0.5rem 1rem; border-radius: 50px;
                border: 1px solid var(--border); font-size: 0.75rem;">
        <span style="width: 8px; height: 8px; background: #4CAF50; border-radius: 50%;
                     animation: pulse 2s infinite;"></span>
        <span style="color: var(--text-secondary);">v1.0.0 | Активен</span>
    </div>
</div>
''', unsafe_allow_html=True)


# ===================== ОСНОВНОЙ КОНТЕНТ =====================
st.markdown('<h1 class="main-header">Прогноз паводков в Кыргызстане</h1>', unsafe_allow_html=True)

# Информация о регионе в виде карточек
region_config = REGIONS[selected_region]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(render_metric_card(
        "📍", "Регион", region_config.name,
        "Выбранный регион для мониторинга паводковой обстановки"
    ), unsafe_allow_html=True)
with col2:
    st.markdown(render_metric_card(
        "📐", "Площадь", f"{region_config.area_km2:,} км²",
        "Общая площадь водосборного бассейна региона"
    ), unsafe_allow_html=True)
with col3:
    st.markdown(render_metric_card(
        "⛰️", "Высота", f"{region_config.avg_elevation} м",
        "Средняя высота над уровнем моря"
    ), unsafe_allow_html=True)
with col4:
    rivers_text = ", ".join(region_config.main_rivers[:2])
    st.markdown(render_metric_card(
        "🏔️", "Реки", rivers_text,
        f"Основные реки: {', '.join(region_config.main_rivers)}"
    ), unsafe_allow_html=True)

st.markdown("---")

# Вкладки
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Симуляция",
    "🤖 ML Прогноз",
    "🗺️ Карта",
    "📈 Анализ",
    "⚠️ События",
    "📉 Расширенные графики"
])


# ===================== ВКЛАДКА СИМУЛЯЦИИ =====================
with tab1:
    st.markdown(f'''
    <div class="section-header">📊 Гидрологическая симуляция</div>
    {render_tooltip("", "Моделирование водного баланса и расчёт гидрографа стока на основе метеорологических данных")}
    ''', unsafe_allow_html=True)

    # Инициализация состояния сессии
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
        st.session_state.input_data = None

    # Запуск симуляции с улучшенной индикацией прогресса
    if run_simulation_btn or st.session_state.simulation_results is None:
        # Контейнер для прогресса
        progress_container = st.container()

        with progress_container:
            st.markdown(show_loading_animation("Подготовка симуляции..."), unsafe_allow_html=True)

            # Прогресс-бар
            progress_bar = st.progress(0, text="Инициализация...")

            # Этап 1: Загрузка/генерация данных
            data_source_text = "Загрузка реальных метеоданных..." if use_real_data else "Генерация синтетических данных..."
            progress_bar.progress(10, text=data_source_text)
            input_data = load_region_data(selected_region, simulation_days, scenario, use_real_data)
            st.session_state.input_data = input_data
            st.session_state.use_real_data = use_real_data

            # Этап 2: Инициализация модели
            progress_bar.progress(30, text="Инициализация гидрологической модели...")
            model = get_flood_model(selected_region)
            model.params.flood_discharge_threshold = flood_threshold

            # Этап 3: Запуск симуляции
            progress_bar.progress(50, text="Выполнение симуляции...")
            results = run_simulation(model, input_data)

            # Этап 4: Анализ событий
            progress_bar.progress(80, text="Анализ паводковых событий...")
            events = model.detect_flood_events(results)

            # Сохранение результатов
            st.session_state.simulation_results = results
            st.session_state.flood_events = events
            st.session_state.model = model

            # Завершение
            progress_bar.progress(100, text="Симуляция завершена!")

        # Уведомление об успехе
        st.success("Симуляция успешно завершена! Результаты готовы к просмотру.")

    # Отображение результатов
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        input_data = st.session_state.input_data

        # Метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            max_discharge = results['discharge_m3s'].max()
            st.metric(
                "🌊 Макс. расход",
                f"{max_discharge:.1f} м³/с",
                delta=f"{max_discharge - flood_threshold:.1f}" if max_discharge > flood_threshold else None,
                delta_color="inverse"
            )

        with col2:
            total_precip = input_data['precipitation_mm'].sum()
            st.metric("🌧️ Общие осадки", f"{total_precip:.1f} мм")

        with col3:
            snowmelt = results['snowmelt_mm'].sum()
            st.metric("❄️ Таяние снега", f"{snowmelt:.1f} мм")

        with col4:
            flood_hours = (results['discharge_m3s'] > flood_threshold).sum()
            st.metric("⚠️ Часов превышения", f"{flood_hours}")

        # График гидрографа
        st.markdown("### 📈 Гидрограф")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Расход воды (м³/с)', 'Осадки (мм)', 'Температура (°C)'),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Расход
        fig.add_trace(
            go.Scatter(
                x=results['timestamp'],
                y=results['discharge_m3s'],
                name='Расход',
                fill='tozeroy',
                fillcolor='rgba(30, 136, 229, 0.3)',
                line=dict(color='#1E88E5', width=2)
            ),
            row=1, col=1
        )

        # Порог паводка
        fig.add_hline(
            y=flood_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Порог: {flood_threshold} м³/с",
            row=1, col=1
        )

        # Осадки
        fig.add_trace(
            go.Bar(
                x=input_data['timestamp'],
                y=input_data['precipitation_mm'],
                name='Осадки',
                marker_color='#42A5F5'
            ),
            row=2, col=1
        )

        # Температура
        fig.add_trace(
            go.Scatter(
                x=input_data['timestamp'],
                y=input_data['temperature_c'],
                name='Температура',
                line=dict(color='#FF7043', width=2)
            ),
            row=3, col=1
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Уровни риска
        st.markdown("### ⚠️ Распределение уровней риска")

        risk_counts = results['risk_level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=[get_risk_label(r) for r in risk_counts.index],
            color=risk_counts.index,
            color_discrete_map={
                'low': '#4CAF50',
                'moderate': '#FFC107',
                'high': '#FF9800',
                'critical': '#F44336'
            },
            hole=0.4
        )
        fig_risk.update_layout(height=300)
        st.plotly_chart(fig_risk, use_container_width=True)


# ===================== ВКЛАДКА ML ПРОГНОЗА =====================
with tab2:
    st.subheader("🤖 Машинное обучение для прогноза паводков")

    if not enable_ml:
        st.info("Включите ML прогноз в боковой панели")
    else:
        with st.spinner("Загрузка ML модели..."):
            try:
                predictor = get_ml_predictor(selected_region)

                if predictor.is_trained:
                    st.success("✅ ML модель загружена и готова к работе")

                    # Информация о возможностях модели
                    st.markdown("### 🔧 Возможности модели")
                    cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)
                    with cap_col1:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Тип модели</div>
                            <div class="metric-value" style="font-size: 1.2rem;">{predictor.model_type.upper()}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with cap_col2:
                        cal_status = "Да" if predictor.use_calibration else "Нет"
                        cal_color = "#4CAF50" if predictor.use_calibration else "#9E9E9E"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Калибровка</div>
                            <div class="metric-value" style="font-size: 1.2rem; color: {cal_color};">{cal_status}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with cap_col3:
                        fs_status = "Да" if predictor.use_feature_selection else "Нет"
                        fs_color = "#4CAF50" if predictor.use_feature_selection else "#9E9E9E"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Отбор признаков</div>
                            <div class="metric-value" style="font-size: 1.2rem; color: {fs_color};">{fs_status}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with cap_col4:
                        stack_status = "Да" if predictor.stacking_model is not None else "Нет"
                        stack_color = "#4CAF50" if predictor.stacking_model is not None else "#9E9E9E"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Стекинг</div>
                            <div class="metric-value" style="font-size: 1.2rem; color: {stack_color};">{stack_status}</div>
                        </div>
                        ''', unsafe_allow_html=True)

                    st.markdown("---")

                    # Метрики обучения
                    if predictor.training_metrics:
                        st.markdown("### 📊 Метрики модели")
                        metrics_df = pd.DataFrame(predictor.training_metrics).T
                        st.dataframe(
                            metrics_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=None),
                            use_container_width=True
                        )

                        # Лучшие гиперпараметры (если были оптимизированы)
                        if predictor.best_params:
                            st.markdown("### 🎛️ Оптимизированные гиперпараметры")
                            for model_name, params in predictor.best_params.items():
                                with st.expander(f"📦 {model_name.upper()}"):
                                    st.json(params)

                    # Важность признаков
                    st.markdown("### 🎯 Важность признаков")
                    importance = predictor.get_feature_importance(15)
                    if not importance.empty:
                        fig_imp = px.bar(
                            importance,
                            x='mean_importance',
                            y='feature',
                            orientation='h',
                            title='Топ-15 важных признаков',
                            color='mean_importance',
                            color_continuous_scale='Blues'
                        )
                        fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_imp, use_container_width=True)

                    # Прогноз
                    st.markdown("### 🔮 Прогноз")

                    if st.session_state.input_data is not None:
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            hours_ahead = st.slider("Часов вперед", 6, 72, 24)

                        with col2:
                            forecast_scenario = st.selectbox(
                                "Сценарий прогноза",
                                ['normal', 'wet', 'dry'],
                                format_func=lambda x: {'normal': 'Обычный', 'wet': 'Влажный', 'dry': 'Сухой'}[x]
                            )

                        if st.button("🔮 Сделать прогноз", type="primary"):
                            with st.spinner("Прогнозирование..."):
                                forecast = predictor.predict_next_hours(
                                    st.session_state.input_data,
                                    hours_ahead=hours_ahead,
                                    scenario=forecast_scenario
                                )

                                # Визуализация прогноза
                                fig_forecast = go.Figure()

                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast['timestamp'],
                                    y=forecast['flood_probability'],
                                    mode='lines+markers',
                                    name='Вероятность паводка',
                                    line=dict(color='#E53935', width=3),
                                    fill='tozeroy',
                                    fillcolor='rgba(229, 57, 53, 0.2)'
                                ))

                                fig_forecast.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                                       annotation_text="Порог 50%")
                                fig_forecast.add_hline(y=0.75, line_dash="dash", line_color="red",
                                                       annotation_text="Высокий риск")

                                fig_forecast.update_layout(
                                    title=f'Прогноз вероятности паводка на {hours_ahead} часов',
                                    xaxis_title='Время',
                                    yaxis_title='Вероятность',
                                    yaxis=dict(range=[0, 1]),
                                    height=400
                                )

                                st.plotly_chart(fig_forecast, use_container_width=True)

                                # Таблица прогноза
                                st.dataframe(
                                    forecast[['timestamp', 'flood_probability', 'risk_level', 'prediction_confidence']].style.format({
                                        'flood_probability': '{:.2%}',
                                        'prediction_confidence': '{:.2%}'
                                    }).background_gradient(subset=['flood_probability'], cmap='RdYlGn_r'),
                                    use_container_width=True
                                )
                    else:
                        st.info("Сначала запустите симуляцию на вкладке 'Симуляция'")

                else:
                    st.warning("ML модель не обучена")

                    if st.button("🎓 Обучить модель"):
                        with st.spinner("Обучение модели (может занять время)..."):
                            data = generate_training_data(region=selected_region, days=365)
                            metrics = predictor.train(data)
                            st.success("Модель обучена!")
                            st.rerun()

            except Exception as e:
                st.error(f"Ошибка ML модели: {str(e)}")


# ===================== ВКЛАДКА КАРТЫ =====================
with tab3:
    st.subheader("🗺️ Карта региона")

    region_config = REGIONS[selected_region]

    # Центр карты
    center_lat = (region_config.lat_min + region_config.lat_max) / 2
    center_lon = (region_config.lon_min + region_config.lon_max) / 2

    # Создание карты
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )

    # Добавление слоев
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer('CartoDB positron', name='Light', attr='© CartoDB').add_to(m)

    # Границы региона
    bounds = [
        [region_config.lat_min, region_config.lon_min],
        [region_config.lat_min, region_config.lon_max],
        [region_config.lat_max, region_config.lon_max],
        [region_config.lat_max, region_config.lon_min],
        [region_config.lat_min, region_config.lon_min]
    ]

    folium.Polygon(
        locations=bounds,
        color='blue',
        weight=2,
        fill=True,
        fillColor='blue',
        fillOpacity=0.1,
        popup=f'Регион: {region_config.name}'
    ).add_to(m)

    # Маркеры рек
    for river in region_config.main_rivers:
        # Случайное положение внутри региона
        lat = np.random.uniform(region_config.lat_min + 0.2, region_config.lat_max - 0.2)
        lon = np.random.uniform(region_config.lon_min + 0.2, region_config.lon_max - 0.2)

        folium.Marker(
            location=[lat, lon],
            popup=f'Река: {river}',
            icon=folium.Icon(color='blue', icon='tint', prefix='fa')
        ).add_to(m)

    # Добавление точек риска если есть результаты
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results

        # Найти периоды высокого риска
        high_risk = results[results['risk_level'].isin(['high', 'critical'])]

        if len(high_risk) > 0:
            # Добавить маркеры для зон риска
            for idx, row in high_risk.iloc[::24].iterrows():  # Каждые 24 часа
                lat = np.random.uniform(region_config.lat_min + 0.1, region_config.lat_max - 0.1)
                lon = np.random.uniform(region_config.lon_min + 0.1, region_config.lon_max - 0.1)

                color = 'red' if row['risk_level'] == 'critical' else 'orange'

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.5,
                    popup=f"Риск: {get_risk_label(row['risk_level'])}<br>Расход: {row['discharge_m3s']:.1f} м³/с"
                ).add_to(m)

    folium.LayerControl().add_to(m)

    # Отображение карты
    st_folium(m, width=None, height=500, use_container_width=True)

    # Легенда
    st.markdown("""
    **Легенда:**
    - 🔵 Реки региона
    - 🟠 Зоны высокого риска
    - 🔴 Зоны критического риска
    """)


# ===================== ВКЛАДКА АНАЛИЗА =====================
with tab4:
    st.subheader("📈 Статистический анализ")

    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        input_data = st.session_state.input_data

        col1, col2 = st.columns(2)

        with col1:
            # Распределение расхода
            fig_dist = px.histogram(
                results,
                x='discharge_m3s',
                nbins=50,
                title='Распределение расхода воды',
                color_discrete_sequence=['#1E88E5']
            )
            fig_dist.add_vline(x=flood_threshold, line_dash="dash", line_color="red",
                              annotation_text="Порог")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Корреляция осадков и расхода
            fig_scatter = px.scatter(
                x=input_data['precipitation_mm'],
                y=results['discharge_m3s'],
                color=results['risk_level'],
                color_discrete_map={
                    'low': '#4CAF50',
                    'moderate': '#FFC107',
                    'high': '#FF9800',
                    'critical': '#F44336'
                },
                title='Осадки vs Расход',
                labels={'x': 'Осадки (мм)', 'y': 'Расход (м³/с)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Временная динамика
        st.markdown("### 📅 Суточная динамика")

        daily_stats = results.copy()
        daily_stats['date'] = pd.to_datetime(results['timestamp']).dt.date
        daily_agg = daily_stats.groupby('date').agg({
            'discharge_m3s': ['mean', 'max'],
            'surface_runoff_mm': 'sum',
            'snowmelt_mm': 'sum'
        }).reset_index()
        daily_agg.columns = ['Дата', 'Средний расход', 'Макс расход', 'Поверхностный сток', 'Таяние снега']

        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])

        fig_daily.add_trace(
            go.Bar(x=daily_agg['Дата'], y=daily_agg['Поверхностный сток'],
                   name='Поверхностный сток', marker_color='#42A5F5'),
            secondary_y=False
        )

        fig_daily.add_trace(
            go.Scatter(x=daily_agg['Дата'], y=daily_agg['Макс расход'],
                       name='Макс расход', line=dict(color='#E53935', width=2)),
            secondary_y=True
        )

        fig_daily.update_layout(title='Суточная динамика', height=400)
        fig_daily.update_yaxes(title_text="Сток (мм)", secondary_y=False)
        fig_daily.update_yaxes(title_text="Расход (м³/с)", secondary_y=True)

        st.plotly_chart(fig_daily, use_container_width=True)

        # Статистика
        st.markdown("### 📊 Сводная статистика")

        stats_data = {
            'Параметр': [
                'Общие осадки (мм)',
                'Средние осадки (мм/день)',
                'Макс осадки (мм/час)',
                'Общий сток (мм)',
                'Средний расход (м³/с)',
                'Макс расход (м³/с)',
                'Таяние снега (мм)',
                'Часов превышения порога'
            ],
            'Значение': [
                f"{input_data['precipitation_mm'].sum():.1f}",
                f"{input_data['precipitation_mm'].sum() / simulation_days:.1f}",
                f"{input_data['precipitation_mm'].max():.1f}",
                f"{results['surface_runoff_mm'].sum():.1f}",
                f"{results['discharge_m3s'].mean():.1f}",
                f"{results['discharge_m3s'].max():.1f}",
                f"{results['snowmelt_mm'].sum():.1f}",
                f"{(results['discharge_m3s'] > flood_threshold).sum()}"
            ]
        }

        st.table(pd.DataFrame(stats_data))

        # ===================== РАСШИРЕННЫЕ ВИЗУАЛИЗАЦИИ =====================
        st.markdown("---")
        st.markdown("## Расширенные визуализации")

        # 1. Календарная тепловая карта риска паводков
        st.markdown("### 📅 Календарная тепловая карта риска")
        st.markdown("*Визуализация максимального расхода воды по дням (стиль GitHub contribution graph)*")

        try:
            fig_calendar = create_calendar_heatmap(
                results,
                value_column='discharge_m3s',
                title='Календарь паводкового риска'
            )
            st.plotly_chart(fig_calendar, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось создать календарную тепловую карту: {e}")

        # 2. Сравнение сценариев
        st.markdown("### 🔄 Сравнение сценариев осадков")
        st.markdown("*Сопоставление гидрографов для разных климатических сценариев*")

        with st.spinner("Расчет сценариев для сравнения..."):
            try:
                scenarios_list = ['normal', 'wet', 'dry', 'extreme']
                scenario_names = {
                    'normal': 'Нормальный',
                    'wet': 'Влажный',
                    'dry': 'Засушливый',
                    'extreme': 'Экстремальный'
                }

                # Расчет для всех сценариев
                scenario_results = {}
                comparison_days = min(simulation_days, 30)  # Ограничиваем для быстрого расчета

                for sc in scenarios_list:
                    sc_data = load_region_data(selected_region, comparison_days, sc)
                    sc_model = get_flood_model(selected_region)
                    sc_model.params.flood_discharge_threshold = flood_threshold
                    sc_results = run_simulation(sc_model, sc_data)
                    scenario_results[scenario_names[sc]] = sc_results

                # Создание сравнительного графика
                fig_comparison = create_scenario_comparison(
                    scenario_results,
                    flood_threshold=flood_threshold
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Таблица сводки по сценариям
                st.markdown("#### Сводка по сценариям")
                comparison_stats = []
                for name, sc_results in scenario_results.items():
                    comparison_stats.append({
                        'Сценарий': name,
                        'Макс расход (м3/с)': f"{sc_results['discharge_m3s'].max():.1f}",
                        'Средний расход (м3/с)': f"{sc_results['discharge_m3s'].mean():.1f}",
                        'Часов > порога': int((sc_results['discharge_m3s'] > flood_threshold).sum()),
                        'Преобладающий риск': get_risk_label(sc_results['risk_level'].value_counts().index[0])
                    })

                comparison_df = pd.DataFrame(comparison_stats)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"Не удалось создать сравнение сценариев: {e}")

        # 3. Анимированный гидрограф
        st.markdown("### 🎬 Анимированный гидрограф")
        st.markdown("*Динамика изменения расхода воды во времени*")

        try:
            fig_animated = create_animated_hydrograph(
                results,
                input_data,
                flood_threshold=flood_threshold,
                animation_speed=100
            )
            st.plotly_chart(fig_animated, use_container_width=True)
            st.info("Используйте кнопки управления для воспроизведения анимации или перемещайте слайдер для просмотра конкретных дней.")
        except Exception as e:
            st.warning(f"Не удалось создать анимированный гидрограф: {e}")

        # 4. Корреляционная матрица
        st.markdown("### 📊 Корреляционный анализ")
        st.markdown("*Взаимосвязи между гидрологическими параметрами*")

        try:
            col_corr1, col_corr2 = st.columns([1, 1])
            with col_corr1:
                fig_corr = create_correlation_matrix(results, input_data)
                st.plotly_chart(fig_corr, use_container_width=True)
            with col_corr2:
                # Расширенная диаграмма распределения риска
                fig_risk_dist = create_risk_distribution_chart(results, chart_type='pie')
                st.plotly_chart(fig_risk_dist, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось создать корреляционный анализ: {e}")

    else:
        st.info("Запустите симуляцию для просмотра анализа")


# ===================== ВКЛАДКА СОБЫТИЙ =====================
with tab5:
    st.subheader("⚠️ Паводковые события")

    if 'flood_events' in st.session_state and st.session_state.flood_events:
        events = st.session_state.flood_events

        st.success(f"Обнаружено паводковых событий: {len(events)}")

        for i, event in enumerate(events, 1):
            with st.expander(f"🌊 Событие {i} - {get_risk_label(event.risk_level.value)}", expanded=i==1):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Начало:** {event.start_time}")
                    st.markdown(f"**Пик:** {event.peak_time}")
                    st.markdown(f"**Конец:** {event.end_time}")

                with col2:
                    st.markdown(f"**Длительность:** {event.duration_hours:.0f} часов")
                    st.markdown(f"**Пиковый расход:** {event.peak_discharge_m3s:.1f} м³/с")
                    st.markdown(f"**Объем:** {event.total_volume_m3/1e6:.2f} млн м³")

                with col3:
                    st.markdown(f"**Макс осадки:** {event.max_precipitation_mm:.1f} мм")
                    st.markdown(f"**Всего осадков:** {event.total_precipitation_mm:.1f} мм")

                    risk_class = f"risk-{event.risk_level.value}"
                    st.markdown(
                        f'<span class="{risk_class}" style="padding:5px 10px;border-radius:5px;display:inline-block;">'
                        f'{get_risk_label(event.risk_level.value)}</span>',
                        unsafe_allow_html=True
                    )

        # График событий
        if len(events) > 1:
            events_df = pd.DataFrame([
                {
                    'Событие': f"Событие {i+1}",
                    'Пиковый расход': e.peak_discharge_m3s,
                    'Длительность': e.duration_hours,
                    'Риск': e.risk_level.value
                }
                for i, e in enumerate(events)
            ])

            fig_events = px.bar(
                events_df,
                x='Событие',
                y='Пиковый расход',
                color='Риск',
                color_discrete_map={
                    'low': '#4CAF50',
                    'moderate': '#FFC107',
                    'high': '#FF9800',
                    'critical': '#F44336'
                },
                title='Сравнение паводковых событий'
            )
            st.plotly_chart(fig_events, use_container_width=True)

    elif st.session_state.simulation_results is not None:
        st.info("🎉 Паводковых событий не обнаружено в данном периоде")
    else:
        st.info("Запустите симуляцию для обнаружения событий")


# ===================== ВКЛАДКА РАСШИРЕННЫХ ГРАФИКОВ =====================
with tab6:
    st.subheader("📉 Расширенные визуализации")

    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        input_data = st.session_state.input_data

        # Конфигурация для скачивания графиков
        chart_config = get_chart_download_config()

        # Подвкладки для различных визуализаций
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "🎬 Анимированный гидрограф",
            "📅 Календарь риска",
            "⚖️ Сравнение сценариев",
            "🔗 Корреляционный анализ"
        ])

        with viz_tab1:
            st.markdown("### 🎬 Анимированный гидрограф")
            st.markdown("""
            Наблюдайте за изменением расхода воды во времени. Используйте кнопку воспроизведения
            для анимации или перетащите слайдер к нужному дню.
            """)

            # Контроль скорости анимации
            animation_speed = st.slider(
                "Скорость анимации (мс на кадр)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                help="Меньшие значения = быстрее анимация"
            )

            fig_animated = create_animated_hydrograph(
                results,
                input_data,
                flood_threshold=flood_threshold,
                animation_speed=animation_speed
            )
            st.plotly_chart(fig_animated, use_container_width=True, config=chart_config)

            # Улучшенный статический гидрограф
            st.markdown("---")
            st.markdown("### 📊 Улучшенный гидрограф")
            fig_enhanced = create_enhanced_hydrograph(
                results, input_data, flood_threshold, show_risk_zones=True
            )
            st.plotly_chart(fig_enhanced, use_container_width=True, config=chart_config)

        with viz_tab2:
            st.markdown("### 📅 Календарь риска паводков")
            st.markdown("""
            Тепловая карта в стиле GitHub, показывающая ежедневный уровень риска паводков.
            Более темные цвета указывают на более высокий риск.
            """)

            fig_calendar = create_calendar_heatmap(
                results,
                value_column='discharge_m3s',
                title='Обзор ежедневного риска паводков'
            )
            st.plotly_chart(fig_calendar, use_container_width=True, config=chart_config)

            # Диаграмма распределения риска
            st.markdown("### 📊 Распределение уровней риска")

            chart_type = st.radio(
                "Тип диаграммы",
                options=['pie', 'sunburst', 'treemap'],
                format_func=lambda x: {'pie': 'Круговая', 'sunburst': 'Солнечная', 'treemap': 'Древовидная'}[x],
                horizontal=True,
                help="Выберите стиль визуализации распределения риска"
            )

            fig_risk = create_risk_distribution_chart(results, chart_type=chart_type)
            st.plotly_chart(fig_risk, use_container_width=True, config=chart_config)

        with viz_tab3:
            st.markdown("### ⚖️ Сравнение сценариев")

            st.markdown("""
            Сравните различные сценарии осадков бок о бок.
            Посмотрите, как изменение условий влияет на риск паводков.
            """)

            # Генерация данных для нескольких сценариев
            if st.button("📊 Сгенерировать сравнение сценариев", type="primary"):
                with st.spinner("Генерация данных для всех сценариев..."):
                    scenarios = {}
                    scenario_names = {
                        'normal': 'Нормальный',
                        'wet': 'Влажный',
                        'dry': 'Засушливый',
                        'extreme': 'Экстремальный'
                    }

                    model = get_flood_model(selected_region)
                    model.params.flood_discharge_threshold = flood_threshold

                    for sc_key, sc_name in scenario_names.items():
                        sc_data = load_region_data(selected_region, simulation_days, sc_key)
                        sc_results = run_simulation(model, sc_data)
                        scenarios[sc_name] = sc_results

                    fig_comparison = create_scenario_comparison(
                        scenarios,
                        flood_threshold=flood_threshold
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True, config=chart_config)

                    # Сводная таблица по сценариям
                    st.markdown("### 📋 Сводка по сценариям")
                    summary_data = []
                    for sc_name, sc_results in scenarios.items():
                        summary_data.append({
                            'Сценарий': sc_name,
                            'Макс. расход (м³/с)': f"{sc_results['discharge_m3s'].max():.1f}",
                            'Средн. расход (м³/с)': f"{sc_results['discharge_m3s'].mean():.1f}",
                            'Часов превышения': int((sc_results['discharge_m3s'] > flood_threshold).sum()),
                            'Макс. риск': get_risk_label(sc_results.loc[sc_results['discharge_m3s'].idxmax(), 'risk_level'])
                        })
                    st.table(pd.DataFrame(summary_data))

        with viz_tab4:
            st.markdown("### 🔗 Корреляционный анализ")
            st.markdown("""
            Исследуйте взаимосвязи между гидрологическими переменными.
            Сильные корреляции отображаются более темными цветами.
            """)

            fig_corr = create_correlation_matrix(results, input_data)
            st.plotly_chart(fig_corr, use_container_width=True, config=chart_config)

            # Индикаторы в виде манометров
            st.markdown("### 🎯 Индикаторы в реальном времени")

            col1, col2, col3 = st.columns(3)

            with col1:
                current_discharge = results['discharge_m3s'].iloc[-1]
                max_discharge = results['discharge_m3s'].max()
                fig_gauge1 = create_gauge_chart(
                    current_discharge,
                    max_discharge * 1.2,
                    "Текущий расход (м³/с)",
                    thresholds={
                        'low': flood_threshold * 0.5,
                        'moderate': flood_threshold * 0.75,
                        'high': flood_threshold
                    }
                )
                st.plotly_chart(fig_gauge1, use_container_width=True, config=chart_config)

            with col2:
                total_precip = input_data['precipitation_mm'].sum()
                max_expected = total_precip * 1.5
                fig_gauge2 = create_gauge_chart(
                    total_precip,
                    max_expected,
                    "Общие осадки (мм)",
                    thresholds={
                        'low': max_expected * 0.3,
                        'moderate': max_expected * 0.5,
                        'high': max_expected * 0.75
                    }
                )
                st.plotly_chart(fig_gauge2, use_container_width=True, config=chart_config)

            with col3:
                flood_hours = (results['discharge_m3s'] > flood_threshold).sum()
                total_hours = len(results)
                fig_gauge3 = create_gauge_chart(
                    flood_hours,
                    total_hours * 0.3,
                    "Часы паводка",
                    thresholds={
                        'low': total_hours * 0.05,
                        'moderate': total_hours * 0.1,
                        'high': total_hours * 0.2
                    }
                )
                st.plotly_chart(fig_gauge3, use_container_width=True, config=chart_config)

        # Секция экспорта
        st.markdown("---")
        st.markdown("### 💾 Экспорт графиков")
        st.markdown("""
        Все графики поддерживают интерактивный экспорт. Используйте иконку камеры в правом верхнем
        углу каждого графика для скачивания в форматах:
        - **PNG**: Изображение высокого разрешения
        - **SVG**: Векторный формат для публикаций
        - **WebP**: Оптимизированный веб-формат
        """)

    else:
        st.info("Запустите симуляцию для просмотра расширенных визуализаций")


# ===================== ФУТЕР =====================
st.markdown("---")
st.markdown(f"""
<div class="footer fade-in">
    <div style="margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🌊</span>
    </div>
    <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);">
        FloodPredict KG - Система прогнозирования паводков
    </div>
    <div style="margin-bottom: 1rem;">
        Интеллектуальная платформа раннего предупреждения для Кыргызстана
    </div>
    <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0; flex-wrap: wrap;">
        <div class="tooltip">
            <span style="background: var(--bg-secondary); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border);">SCS-CN</span>
            <span class="tooltip-text">Метод Службы охраны почв для расчёта поверхностного стока</span>
        </div>
        <div class="tooltip">
            <span style="background: var(--bg-secondary); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border);">Degree-Day</span>
            <span class="tooltip-text">Температурный индекс для моделирования таяния снега</span>
        </div>
        <div class="tooltip">
            <span style="background: var(--bg-secondary); padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border);">ML Ensemble</span>
            <span class="tooltip-text">Ансамбль моделей машинного обучения для прогнозирования</span>
        </div>
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--border);">
        <span style="color: var(--text-secondary);">Дипломный проект | 2024</span>
    </div>
</div>
""", unsafe_allow_html=True)
