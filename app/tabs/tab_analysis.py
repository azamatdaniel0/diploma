"""
Вкладка 4: Статистический анализ и расширенные визуализации.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.visualization.enhanced_charts import (
    create_animated_hydrograph,
    create_calendar_heatmap,
    create_scenario_comparison,
    create_correlation_matrix,
    create_risk_distribution_chart,
)
from app.sidebar import SidebarParams
from app.services import load_region_data, get_flood_model, run_simulation
from app.ui_helpers import get_risk_label


def render_tab_analysis(params: SidebarParams) -> None:
    """Рендеринг вкладки статистического анализа."""
    st.subheader("📈 Статистический анализ")

    if st.session_state.get('simulation_results') is None:
        st.info("Запустите симуляцию для просмотра анализа")
        return

    results = st.session_state.simulation_results
    input_data = st.session_state.input_data

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = px.histogram(
            results, x='discharge_m3s', nbins=50,
            title='Распределение расхода воды',
            color_discrete_sequence=['#1E88E5']
        )
        fig_dist.add_vline(
            x=params.flood_threshold, line_dash="dash",
            line_color="red", annotation_text="Порог"
        )
        st.plotly_chart(fig_dist, width="stretch")

    with col2:
        fig_scatter = px.scatter(
            x=input_data['precipitation_mm'],
            y=results['discharge_m3s'],
            color=results['risk_level'],
            color_discrete_map={
                'low': '#4CAF50', 'moderate': '#FFC107',
                'high': '#FF9800', 'critical': '#F44336'
            },
            title='Осадки vs Расход',
            labels={'x': 'Осадки (мм)', 'y': 'Расход (м³/с)'}
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # Суточная динамика
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
    st.plotly_chart(fig_daily, width="stretch")

    # Сводная статистика
    st.markdown("### 📊 Сводная статистика")
    stats_data = {
        'Параметр': [
            'Общие осадки (мм)', 'Средние осадки (мм/день)', 'Макс осадки (мм/час)',
            'Общий сток (мм)', 'Средний расход (м³/с)', 'Макс расход (м³/с)',
            'Таяние снега (мм)', 'Часов превышения порога'
        ],
        'Значение': [
            f"{input_data['precipitation_mm'].sum():.1f}",
            f"{input_data['precipitation_mm'].sum() / params.simulation_days:.1f}",
            f"{input_data['precipitation_mm'].max():.1f}",
            f"{results['surface_runoff_mm'].sum():.1f}",
            f"{results['discharge_m3s'].mean():.1f}",
            f"{results['discharge_m3s'].max():.1f}",
            f"{results['snowmelt_mm'].sum():.1f}",
            f"{(results['discharge_m3s'] > params.flood_threshold).sum()}"
        ]
    }
    st.table(pd.DataFrame(stats_data))

    # ===================== РАСШИРЕННЫЕ ВИЗУАЛИЗАЦИИ =====================
    st.markdown("---")
    st.markdown("## Расширенные визуализации")

    # 1. Календарная тепловая карта
    st.markdown("### 📅 Календарная тепловая карта риска")
    st.markdown("*Визуализация максимального расхода воды по дням (стиль GitHub contribution graph)*")
    try:
        fig_calendar = create_calendar_heatmap(
            results, value_column='discharge_m3s', title='Календарь паводкового риска'
        )
        st.plotly_chart(fig_calendar, width="stretch")
    except Exception as e:
        st.warning(f"Не удалось создать календарную тепловую карту: {e}")

    # 2. Сравнение сценариев
    st.markdown("### 🔄 Сравнение сценариев осадков")
    st.markdown("*Сопоставление гидрографов для разных климатических сценариев*")
    with st.spinner("Расчет сценариев для сравнения..."):
        try:
            scenarios_list = ['normal', 'wet', 'dry', 'extreme']
            scenario_names = {
                'normal': 'Нормальный', 'wet': 'Влажный',
                'dry': 'Засушливый', 'extreme': 'Экстремальный'
            }
            scenario_results = {}
            comparison_days = min(params.simulation_days, 30)

            for sc in scenarios_list:
                sc_data = load_region_data(params.selected_region, comparison_days, sc)
                sc_model = get_flood_model(params.selected_region)
                sc_model.params.flood_discharge_threshold = params.flood_threshold
                sc_results = run_simulation(sc_model, sc_data)
                scenario_results[scenario_names[sc]] = sc_results

            fig_comparison = create_scenario_comparison(
                scenario_results, flood_threshold=params.flood_threshold
            )
            st.plotly_chart(fig_comparison, width="stretch")

            st.markdown("#### Сводка по сценариям")
            comparison_stats = []
            for name, sc_results in scenario_results.items():
                risk_counts = sc_results['risk_level'].value_counts()
                predominant_risk = get_risk_label(risk_counts.index[0]) if len(risk_counts) > 0 else 'Нет данных'
                comparison_stats.append({
                    'Сценарий': name,
                    'Макс расход (м3/с)': f"{sc_results['discharge_m3s'].max():.1f}",
                    'Средний расход (м3/с)': f"{sc_results['discharge_m3s'].mean():.1f}",
                    'Часов > порога': int((sc_results['discharge_m3s'] > params.flood_threshold).sum()),
                    'Преобладающий риск': predominant_risk
                })
            st.dataframe(pd.DataFrame(comparison_stats), width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"Не удалось создать сравнение сценариев: {e}")

    # 3. Анимированный гидрограф
    st.markdown("### 🎬 Анимированный гидрограф")
    st.markdown("*Динамика изменения расхода воды во времени*")
    try:
        fig_animated = create_animated_hydrograph(
            results, input_data,
            flood_threshold=params.flood_threshold, animation_speed=100
        )
        st.plotly_chart(fig_animated, width="stretch")
        st.info("Используйте кнопки управления для воспроизведения анимации или перемещайте слайдер.")
    except Exception as e:
        st.warning(f"Не удалось создать анимированный гидрограф: {e}")

    # 4. Корреляционная матрица
    st.markdown("### 📊 Корреляционный анализ")
    st.markdown("*Взаимосвязи между гидрологическими параметрами*")
    try:
        col_corr1, col_corr2 = st.columns([1, 1])
        with col_corr1:
            fig_corr = create_correlation_matrix(results, input_data)
            st.plotly_chart(fig_corr, width="stretch")
        with col_corr2:
            fig_risk_dist = create_risk_distribution_chart(results, chart_type='pie')
            st.plotly_chart(fig_risk_dist, width="stretch")
    except Exception as e:
        st.warning(f"Не удалось создать корреляционный анализ: {e}")
