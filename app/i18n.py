"""
Интернационализация: переводы и переключатели языка/темы.
"""

import streamlit as st

TRANSLATIONS = {
    'ru': {
        'title': 'Прогноз паводков в Кыргызстане',
        'region': 'Регион',
        'area': 'Площадь',
        'elevation': 'Высота',
        'rivers': 'Реки',
        'simulation': 'Симуляция',
        'ml_prediction': 'ML Прогноз',
        'map': 'Карта',
        'analysis': 'Анализ',
        'events': 'События',
        'advanced_charts': 'Расширенные графики',
        'export': 'Экспорт',
        'dark_theme': 'Тёмная тема',
        'light_theme': 'Светлая тема',
        'kyrgyz_lang': 'Кыргызча',
        'russian_lang': 'Русский',
    },
    'kg': {
        'title': 'Кыргызстандагы тошкундарды болжолдоо',
        'region': 'Аймак',
        'area': 'Аянты',
        'elevation': 'Бийиктиги',
        'rivers': 'Дарыялар',
        'simulation': 'Симуляция',
        'ml_prediction': 'ML Болжолдоо',
        'map': 'Карта',
        'analysis': 'Анализ',
        'events': 'Окуялар',
        'advanced_charts': 'Кеңейтилген графиктер',
        'export': 'Экспорт',
        'dark_theme': 'Караңгы тема',
        'light_theme': 'Жарык тема',
        'kyrgyz_lang': 'Кыргызча',
        'russian_lang': 'Орусча',
    }
}


def t(key: str) -> str:
    """Получить перевод по ключу для текущего языка."""
    return TRANSLATIONS[st.session_state.language].get(key, key)


def toggle_theme() -> None:
    """Переключение между светлой и тёмной темой."""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'


def toggle_language() -> None:
    """Переключение между русским и кыргызским языком."""
    st.session_state.language = 'kg' if st.session_state.language == 'ru' else 'ru'
