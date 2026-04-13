"""
Вспомогательные функции для рендеринга HTML-компонентов UI.
Все функции чистые — возвращают HTML-строки без вызовов Streamlit.
"""


def render_tooltip(text: str, tooltip: str) -> str:
    """Рендеринг текста с всплывающей подсказкой."""
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
    """Рендеринг индикатора уровня риска."""
    labels = {
        'low': ('Низкий', ''),
        'moderate': ('Умеренный', ''),
        'high': ('Высокий', ''),
        'critical': ('Критический', '')
    }
    label, icon = labels.get(risk, (risk, ''))
    return f'<span class="risk-badge risk-{risk}">{icon} {label}</span>'


def show_loading_animation(message: str = "Загрузка...") -> str:
    """HTML-анимация загрузки."""
    return f'''
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
    </div>
    '''


def get_risk_color(risk: str) -> str:
    """Цвет для уровня риска (hex)."""
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
