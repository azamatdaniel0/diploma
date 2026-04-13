"""
Веб-интерфейс для системы прогнозирования паводков в Кыргызстане.

Запуск: streamlit run web_app.py

Архитектура:
    web_app.py          — оркестратор (~90 строк), только вызовы app/
    app/theme.py        — THEMES + inject_css()
    app/i18n.py         — TRANSLATIONS, t(), toggle_theme/language()
    app/ui_helpers.py   — HTML helper функции
    app/services.py     — @cache_data/@cache_resource функции + init_session_state()
    app/sidebar.py      — SidebarParams + render_sidebar()
    app/header.py       — render_header() + render_region_cards()
    app/tabs/           — по одному файлу на каждую вкладку
"""

import sys
from pathlib import Path

import streamlit as st

# Добавляем корень проекта в sys.path для импорта src/
sys.path.insert(0, str(Path(__file__).parent))

# ===================== ИМПОРТЫ app/ =====================
from app.theme import inject_css
from app.i18n import t
from app.services import init_session_state
from app.sidebar import render_sidebar
from app.header import render_header, render_region_cards
from app.tabs.tab_simulation import render_tab_simulation
from app.tabs.tab_ml import render_tab_ml
from app.tabs.tab_map import render_tab_map
from app.tabs.tab_analysis import render_tab_analysis
from app.tabs.tab_events import render_tab_events
from app.tabs.tab_advanced import render_tab_advanced

# ===================== 1. КОНФИГУРАЦИЯ СТРАНИЦЫ (первый вызов Streamlit) =====================
st.set_page_config(
    page_title="Прогноз паводков - Кыргызстан",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 2. БАЗОВЫЙ SESSION STATE (только theme + language) =====================
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if 'language' not in st.session_state:
    st.session_state.language = 'ru'

# ===================== 3. CSS (сразу после state, до UI) =====================
inject_css()

# ===================== 4. ИНИЦИАЛИЗАЦИЯ ДАННЫХ =====================
# Вызываем ПОСЛЕ импорта app.services — все функции уже определены,
# NameError архитектурно невозможен.
init_session_state()

# ===================== 5. САЙДБАР → типизированные параметры =====================
params = render_sidebar()

# ===================== 6. ОСНОВНОЙ КОНТЕНТ =====================
render_header()
render_region_cards(params.selected_region)
st.markdown("---")

# ===================== 7. ВКЛАДКИ =====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    f"📊 {t('simulation')}",
    f"🤖 {t('ml_prediction')}",
    f"🗺️ {t('map')}",
    f"📈 {t('analysis')}",
    f"⚠️ {t('events')}",
    f"📉 {t('advanced_charts')}"
])

with tab1:
    render_tab_simulation(params)

with tab2:
    render_tab_ml(params)

with tab3:
    render_tab_map(params)

with tab4:
    render_tab_analysis(params)

with tab5:
    render_tab_events()

with tab6:
    render_tab_advanced(params)

# ===================== 8. ФУТЕР =====================
st.markdown("---")
st.markdown("""
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
