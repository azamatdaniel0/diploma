"""
Боковая панель: параметры симуляции и управление.
render_sidebar() возвращает типизированный объект SidebarParams,
устраняя зависимость от глобальных переменных.
"""

from dataclasses import dataclass

import streamlit as st

from src.config import REGIONS
from app.i18n import t, toggle_theme, toggle_language
from app.services import get_current_weather


@dataclass
class SidebarParams:
    """Параметры, выбранные пользователем в боковой панели."""
    selected_region: str
    scenario: str
    simulation_days: int
    flood_threshold: float
    use_real_data: bool
    enable_ml: bool
    run_simulation_btn: bool


def render_sidebar() -> SidebarParams:
    """
    Рендеринг боковой панели.
    Возвращает SidebarParams со всеми выбранными пользователем значениями.
    """
    # Логотип
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
    theme_label = t('dark_theme') if st.session_state.theme == 'light' else t('light_theme')

    col_theme1, col_theme2 = st.sidebar.columns([1, 3])
    with col_theme1:
        st.markdown(
            f"<div style='font-size: 1.5rem; text-align: center;'>{theme_icon}</div>",
            unsafe_allow_html=True
        )
    with col_theme2:
        if st.button(theme_label, key="theme_toggle", use_container_width=True):
            toggle_theme()
            st.rerun()

    # Переключатель языка
    lang_icon = "🇰🇬" if st.session_state.language == 'kg' else "🇷🇺"
    lang_label = t('russian_lang') if st.session_state.language == 'kg' else t('kyrgyz_lang')

    col_lang1, col_lang2 = st.sidebar.columns([1, 3])
    with col_lang1:
        st.markdown(
            f"<div style='font-size: 1.5rem; text-align: center;'>{lang_icon}</div>",
            unsafe_allow_html=True
        )
    with col_lang2:
        if st.button(lang_label, key="lang_toggle", use_container_width=True):
            toggle_language()
            st.rerun()

    st.sidebar.markdown("---")

    # Выбор региона
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

    st.sidebar.info("💡 **Для презентации**: Выберите 'Влажный' или 'Экстремальный' сценарий и период 60-180 дней")

    scenario = st.sidebar.selectbox(
        "Сценарий осадков",
        options=['normal', 'wet', 'dry', 'extreme'],
        format_func=lambda x: {
            'normal': '🌤️ Нормальный',
            'wet': '🌧️ Влажный (+50%)',
            'dry': '☀️ Засушливый (-50%)',
            'extreme': '⛈️ Экстремальный (+100%)'
        }[x],
        index=1,
        help="Сценарий определяет интенсивность осадков в симуляции"
    )

    simulation_days = st.sidebar.slider(
        "Период симуляции (дней)",
        min_value=7,
        max_value=364,
        value=91,
        step=7,
        help="Количество дней для моделирования гидрологической обстановки"
    )

    period_percent = (simulation_days - 7) / (364 - 7) * 100
    st.sidebar.markdown(f'''
<div style="background: var(--bg-secondary); border-radius: 4px; height: 6px; margin: 0.5rem 0;">
    <div style="background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
                border-radius: 4px; height: 100%; width: {period_percent}%; transition: width 0.3s;"></div>
</div>
''', unsafe_allow_html=True)

    st.sidebar.markdown("### 🌊 Порог паводка")
    flood_threshold = st.sidebar.number_input(
        "Порог (м³/с)",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=10.0,
        help="Критический уровень расхода воды"
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
        help="Использовать реальные метеоданные из Open-Meteo API."
    )

    if use_real_data:
        st.sidebar.markdown('''
    <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
                border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; border: 1px solid var(--border);">
        <div style="font-size: 0.8rem; color: var(--text-secondary);">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">✓</span><span>Прогноз до 16 дней</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">✓</span><span>Почасовые данные</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #2196F3;">ℹ</span><span>API: Open-Meteo (бесплатно)</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

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
                <span>📊</span><span>Синтетические данные для тестирования</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Кнопка запуска симуляции
    run_simulation_btn = st.sidebar.button(
        "▶️ Запустить симуляцию",
        type="primary",
        width="stretch",
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
                <span style="color: #4CAF50;">●</span><span>Random Forest</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 4px;">
                <span style="color: #4CAF50;">●</span><span>Gradient Boosting</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #4CAF50;">●</span><span>Neural Network</span>
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

    return SidebarParams(
        selected_region=selected_region,
        scenario=scenario,
        simulation_days=simulation_days,
        flood_threshold=flood_threshold,
        use_real_data=use_real_data,
        enable_ml=enable_ml,
        run_simulation_btn=run_simulation_btn,
    )
