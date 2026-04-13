"""
Вкладка 5: Паводковые события.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_helpers import get_risk_label


def render_tab_events() -> None:
    """Рендеринг вкладки паводковых событий."""
    st.subheader("⚠️ Паводковые события")

    if 'flood_events' in st.session_state and st.session_state.flood_events:
        events = st.session_state.flood_events

        st.success(f"Обнаружено паводковых событий: {len(events)}")

        for i, event in enumerate(events, 1):
            with st.expander(
                f"🌊 Событие {i} - {get_risk_label(event.risk_level.value)}",
                expanded=(i == 1)
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Начало:** {event.start_time}")
                    st.markdown(f"**Пик:** {event.peak_time}")
                    st.markdown(f"**Конец:** {event.end_time}")

                with col2:
                    st.markdown(f"**Длительность:** {event.duration_hours:.0f} часов")
                    st.markdown(f"**Пиковый расход:** {event.peak_discharge_m3s:.1f} м³/с")
                    st.markdown(f"**Объем:** {event.total_volume_m3 / 1e6:.2f} млн м³")

                with col3:
                    st.markdown(f"**Макс осадки:** {event.max_precipitation_mm:.1f} мм")
                    st.markdown(f"**Всего осадков:** {event.total_precipitation_mm:.1f} мм")

                    risk_class = f"risk-{event.risk_level.value}"
                    st.markdown(
                        f'<span class="{risk_class}" style="padding:5px 10px;border-radius:5px;display:inline-block;">'
                        f'{get_risk_label(event.risk_level.value)}</span>',
                        unsafe_allow_html=True
                    )

        # График сравнения событий
        if len(events) > 1:
            events_df = pd.DataFrame([
                {
                    'Событие': f"Событие {i + 1}",
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
            st.plotly_chart(fig_events, width="stretch")

    elif st.session_state.get('simulation_results') is not None:
        st.info("🎉 Паводковых событий не обнаружено в данном периоде")
    else:
        st.info("Запустите симуляцию для обнаружения событий")
