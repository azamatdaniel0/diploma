"""
Вкладка 1: Гидрологическая симуляция.
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.sidebar import SidebarParams
from app.services import load_region_data, get_flood_model, run_simulation
from app.ui_helpers import show_loading_animation, get_risk_label


def render_tab_simulation(params: SidebarParams) -> None:
    """Рендеринг вкладки симуляции."""
    st.markdown('''
    <div class="section-header">📊 Гидрологическая симуляция</div>
    ''', unsafe_allow_html=True)

    # Запуск симуляции при нажатии кнопки
    if params.run_simulation_btn:
        progress_container = st.container()

        with progress_container:
            st.markdown(show_loading_animation("Подготовка симуляции..."), unsafe_allow_html=True)
            progress_bar = st.progress(0, text="Инициализация...")

            # Этап 1: Загрузка данных
            data_source_text = (
                "Загрузка реальных метеоданных..."
                if params.use_real_data
                else "Генерация синтетических данных..."
            )
            progress_bar.progress(10, text=data_source_text)
            input_data = load_region_data(
                params.selected_region,
                params.simulation_days,
                params.scenario,
                params.use_real_data
            )
            st.session_state.input_data = input_data
            st.session_state.use_real_data = params.use_real_data

            # Этап 2: Инициализация модели
            progress_bar.progress(30, text="Инициализация гидрологической модели...")
            model = get_flood_model(params.selected_region)
            model.params.flood_discharge_threshold = params.flood_threshold

            # Этап 3: Симуляция
            progress_bar.progress(50, text="Выполнение симуляции...")
            results = run_simulation(model, input_data)

            # Этап 4: Анализ событий
            progress_bar.progress(80, text="Анализ паводковых событий...")
            events = model.detect_flood_events(results)

            st.session_state.simulation_results = results
            st.session_state.flood_events = events
            st.session_state.model = model

            progress_bar.progress(100, text="Симуляция завершена!")

        st.success("Симуляция успешно завершена! Результаты готовы к просмотру.")

    # Отображение результатов
    if st.session_state.get('simulation_results') is not None:
        results = st.session_state.simulation_results
        input_data = st.session_state.input_data

        # Кнопки экспорта
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns([2, 1, 1, 2])
        with col_exp2:
            csv_data = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Скачать CSV",
                data=csv_data,
                file_name=f"simulation_results_{params.selected_region}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp3:
            results_dict = results.copy()
            for col in results_dict.columns:
                if results_dict[col].dtype == 'datetime64[ns]':
                    results_dict[col] = results_dict[col].astype(str)
                else:
                    results_dict[col] = results_dict[col].apply(
                        lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x
                    )
            json_string = json.dumps(
                results_dict.to_dict(orient='records'), indent=2, ensure_ascii=False
            )
            st.download_button(
                label="📄 Скачать JSON",
                data=json_string,
                file_name=f"simulation_results_{params.selected_region}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_discharge = results['discharge_m3s'].max()
            st.metric(
                "🌊 Макс. расход",
                f"{max_discharge:.1f} м³/с",
                delta=f"{max_discharge - params.flood_threshold:.1f}"
                if max_discharge > params.flood_threshold else None,
                delta_color="inverse"
            )
        with col2:
            total_precip = input_data['precipitation_mm'].sum()
            st.metric("🌧️ Общие осадки", f"{total_precip:.1f} мм")
        with col3:
            snowmelt = results['snowmelt_mm'].sum()
            st.metric("❄️ Таяние снега", f"{snowmelt:.1f} мм")
        with col4:
            flood_hours = (results['discharge_m3s'] > params.flood_threshold).sum()
            st.metric("⚠️ Часов превышения", f"{flood_hours}")

        # Гидрограф
        st.markdown("### 📈 Гидрограф")
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Расход воды (м³/с)', 'Осадки (мм)', 'Температура (°C)'),
            row_heights=[0.5, 0.25, 0.25]
        )

        fig.add_trace(
            go.Scatter(
                x=results['timestamp'], y=results['discharge_m3s'],
                name='Расход', fill='tozeroy',
                fillcolor='rgba(30, 136, 229, 0.3)',
                line=dict(color='#1E88E5', width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=params.flood_threshold, line_dash="dash", line_color="red",
            annotation_text=f"Порог: {params.flood_threshold} м³/с",
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=input_data['timestamp'], y=input_data['precipitation_mm'],
                name='Осадки', marker_color='#42A5F5'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=input_data['timestamp'], y=input_data['temperature_c'],
                name='Температура', line=dict(color='#FF7043', width=2)
            ),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        fig.update_layout(
            height=600, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, width="stretch")

        # Распределение уровней риска
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
        st.plotly_chart(fig_risk, width="stretch")
