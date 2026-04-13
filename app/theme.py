"""
Управление темами оформления и инжект CSS.
"""

import streamlit as st
from pathlib import Path

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

# JavaScript для тултипов
_TOOLTIP_JS = """
<script>
    document.addEventListener('DOMContentLoaded', function() {
        document.body.addEventListener('click', function(e) {
            const tooltip = e.target.closest('.tooltip');
            if (tooltip) {
                e.stopPropagation();
                document.querySelectorAll('.tooltip.active').forEach(t => {
                    if (t !== tooltip) t.classList.remove('active');
                });
                tooltip.classList.toggle('active');
            } else {
                document.querySelectorAll('.tooltip.active').forEach(t => {
                    t.classList.remove('active');
                });
            }
        });
    });
    if (window.addEventListener) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                document.body.addEventListener('click', function(e) {
                    const tooltip = e.target.closest('.tooltip');
                    if (tooltip) {
                        e.stopPropagation();
                        document.querySelectorAll('.tooltip.active').forEach(t => {
                            if (t !== tooltip) t.classList.remove('active');
                        });
                        tooltip.classList.toggle('active');
                    } else {
                        document.querySelectorAll('.tooltip.active').forEach(t => {
                            t.classList.remove('active');
                        });
                    }
                });
            }, 100);
        });
    }
</script>
"""


def inject_css() -> None:
    """
    Инжектирует CSS в страницу Streamlit.
    Строит :root{} блок из текущей темы, затем добавляет статический CSS из styles.css.
    """
    theme = THEMES[st.session_state.theme]

    root_block = f"""
    :root {{
        --bg-primary: {theme['bg_primary']};
        --bg-secondary: {theme['bg_secondary']};
        --bg-card: {theme['bg_card']};
        --text-primary: {theme['text_primary']};
        --text-secondary: {theme['text_secondary']};
        --accent: {theme['accent']};
        --accent-hover: {theme['accent_hover']};
        --border: {theme['border']};
        --shadow: {theme['shadow']};
        --gradient-start: {theme['gradient_start']};
        --gradient-end: {theme['gradient_end']};
        --text-on-accent: {theme['text_on_accent']};
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    """

    css_path = Path(__file__).parent / "styles.css"
    static_css = css_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{root_block}{static_css}</style>{_TOOLTIP_JS}",
        unsafe_allow_html=True
    )
