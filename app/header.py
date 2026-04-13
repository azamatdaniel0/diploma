"""
Шапка страницы: заголовок и карточки с информацией о регионе.
"""

import streamlit as st

from src.config import REGIONS
from app.i18n import t
from app.ui_helpers import render_metric_card


def render_header() -> None:
    """Рендеринг главного заголовка страницы."""
    st.markdown(
        f'<h1 class="main-header">{t("title")}</h1>',
        unsafe_allow_html=True
    )


def render_region_cards(selected_region: str) -> None:
    """Рендеринг карточек с информацией о выбранном регионе."""
    region_config = REGIONS[selected_region]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(render_metric_card(
            "📍", t("region"), region_config.name,
            "Выбранный регион для мониторинга паводковой обстановки"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(render_metric_card(
            "📐", t("area"), f"{region_config.area_km2:,} км²",
            "Общая площадь водосборного бассейна региона"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(render_metric_card(
            "⛰️", t("elevation"), f"{region_config.avg_elevation} м",
            "Средняя высота над уровнем моря"
        ), unsafe_allow_html=True)

    with col4:
        rivers_text = ", ".join(region_config.main_rivers[:2])
        st.markdown(render_metric_card(
            "🏔️", t("rivers"), rivers_text,
            f"Основные реки: {', '.join(region_config.main_rivers)}"
        ), unsafe_allow_html=True)
