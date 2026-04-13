"""
Вкладка 6: Расширенные визуализации (анимации, gauge, сравнение сценариев).
"""

import streamlit as st

from src.visualization.enhanced_charts import (
    create_animated_hydrograph,
    create_calendar_heatmap,
    create_scenario_comparison,
    create_enhanced_hydrograph,
    create_risk_distribution_chart,
    create_correlation_matrix,
    create_gauge_chart,
    get_chart_download_config,
)
from app.sidebar import SidebarParams
from app.services import load_region_data, get_flood_model, run_simulation
from app.ui_helpers import get_risk_label


def render_tab_advanced(params: SidebarParams) -> None:
    """Рендеринг вкладки расширенных визуализаций."""
    st.subheader("📉 Расширенные визуализации")

    if st.session_state.get('simulation_results') is None:
        st.info("Запустите симуляцию для просмотра расширенных визуализаций")
        return

    results = st.session_state.simulation_results
    input_data = st.session_state.input_data
    chart_config = get_chart_download_config()

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

        animation_speed = st.slider(
            "Скорость анимации (мс на кадр)",
            min_value=50, max_value=500, value=100, step=50,
            help="Меньшие значения = быстрее анимация"
        )

        fig_animated = create_animated_hydrograph(
            results, input_data,
            flood_threshold=params.flood_threshold,
            animation_speed=animation_speed
        )
        st.plotly_chart(fig_animated, width="stretch", config=chart_config)

        st.markdown("---")
        st.markdown("### 📊 Улучшенный гидрограф")
        fig_enhanced = create_enhanced_hydrograph(
            results, input_data, params.flood_threshold, show_risk_zones=True
        )
        st.plotly_chart(fig_enhanced, width="stretch", config=chart_config)

    with viz_tab2:
        st.markdown("### 📅 Календарь риска паводков")
        st.markdown("""
        Тепловая карта в стиле GitHub, показывающая ежедневный уровень риска паводков.
        Более темные цвета указывают на более высокий риск.
        """)

        fig_calendar = create_calendar_heatmap(
            results, value_column='discharge_m3s',
            title='Обзор ежедневного риска паводков'
        )
        st.plotly_chart(fig_calendar, width="stretch", config=chart_config)

        st.markdown("### 📊 Распределение уровней риска")
        chart_type = st.radio(
            "Тип диаграммы",
            options=['pie', 'sunburst', 'treemap'],
            format_func=lambda x: {'pie': 'Круговая', 'sunburst': 'Солнечная', 'treemap': 'Древовидная'}[x],
            horizontal=True,
            help="Выберите стиль визуализации распределения риска"
        )
        fig_risk = create_risk_distribution_chart(results, chart_type=chart_type)
        st.plotly_chart(fig_risk, width="stretch", config=chart_config)

    with viz_tab3:
        st.markdown("### ⚖️ Сравнение сценариев")
        st.markdown("""
        Сравните различные сценарии осадков бок о бок.
        Посмотрите, как изменение условий влияет на риск паводков.
        """)

        if st.button("📊 Сгенерировать сравнение сценариев", type="primary"):
            with st.spinner("Генерация данных для всех сценариев..."):
                scenarios = {}
                scenario_names = {
                    'normal': 'Нормальный', 'wet': 'Влажный',
                    'dry': 'Засушливый', 'extreme': 'Экстремальный'
                }
                model = get_flood_model(params.selected_region)
                model.params.flood_discharge_threshold = params.flood_threshold

                for sc_key, sc_name in scenario_names.items():
                    sc_data = load_region_data(params.selected_region, params.simulation_days, sc_key)
                    sc_results = run_simulation(model, sc_data)
                    scenarios[sc_name] = sc_results

                fig_comparison = create_scenario_comparison(
                    scenarios, flood_threshold=params.flood_threshold
                )
                st.plotly_chart(fig_comparison, width="stretch", config=chart_config)

                st.markdown("### 📋 Сводка по сценариям")
                summary_data = []
                for sc_name, sc_results in scenarios.items():
                    summary_data.append({
                        'Сценарий': sc_name,
                        'Макс. расход (м³/с)': f"{sc_results['discharge_m3s'].max():.1f}",
                        'Средн. расход (м³/с)': f"{sc_results['discharge_m3s'].mean():.1f}",
                        'Часов превышения': int((sc_results['discharge_m3s'] > params.flood_threshold).sum()),
                        'Макс. риск': get_risk_label(
                            sc_results.loc[sc_results['discharge_m3s'].idxmax(), 'risk_level']
                        )
                    })
                import pandas as pd
                st.table(pd.DataFrame(summary_data))

    with viz_tab4:
        st.markdown("### 🔗 Корреляционный анализ")
        st.markdown("""
        Исследуйте взаимосвязи между гидрологическими переменными.
        Сильные корреляции отображаются более темными цветами.
        """)

        fig_corr = create_correlation_matrix(results, input_data)
        st.plotly_chart(fig_corr, width="stretch", config=chart_config)

        st.markdown("### 🎯 Индикаторы в реальном времени")

        if len(results) == 0 or len(input_data) == 0:
            st.warning("Недостаточно данных для отображения индикаторов")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                current_discharge = results['discharge_m3s'].iloc[-1]
                max_discharge = results['discharge_m3s'].max()
                fig_gauge1 = create_gauge_chart(
                    current_discharge, max_discharge * 1.2,
                    "Текущий расход (м³/с)",
                    thresholds={
                        'low': params.flood_threshold * 0.5,
                        'moderate': params.flood_threshold * 0.75,
                        'high': params.flood_threshold
                    }
                )
                st.plotly_chart(fig_gauge1, width="stretch", config=chart_config)

            with col2:
                total_precip = input_data['precipitation_mm'].sum()
                max_expected = total_precip * 1.5
                fig_gauge2 = create_gauge_chart(
                    total_precip, max_expected,
                    "Общие осадки (мм)",
                    thresholds={
                        'low': max_expected * 0.3,
                        'moderate': max_expected * 0.5,
                        'high': max_expected * 0.75
                    }
                )
                st.plotly_chart(fig_gauge2, width="stretch", config=chart_config)

            with col3:
                flood_hours = (results['discharge_m3s'] > params.flood_threshold).sum()
                total_hours = len(results)
                fig_gauge3 = create_gauge_chart(
                    flood_hours, total_hours * 0.3,
                    "Часы паводка",
                    thresholds={
                        'low': total_hours * 0.05,
                        'moderate': total_hours * 0.1,
                        'high': total_hours * 0.2
                    }
                )
                st.plotly_chart(fig_gauge3, width="stretch", config=chart_config)

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
