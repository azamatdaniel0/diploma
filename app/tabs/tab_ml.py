"""
Вкладка 2: ML прогноз паводков.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.ml_predictor import FloodMLPredictor, generate_training_data
from app.sidebar import SidebarParams
from app.services import get_ml_predictor


def render_tab_ml(params: SidebarParams) -> None:
    """Рендеринг вкладки ML прогноза."""
    st.subheader("🤖 Машинное обучение для прогноза паводков")

    # Показать доступные обученные модели
    models_dir = Path('models')
    all_trained_models = []
    if models_dir.exists():
        all_trained_models = list(models_dir.glob('flood_predictor_*_real.joblib'))

    if all_trained_models:
        with st.expander(f"📦 Обученные модели ({len(all_trained_models)} доступно)", expanded=False):
            cols = st.columns(3)
            for idx, model_path in enumerate(all_trained_models):
                model_name = model_path.stem
                region_name = model_name.split('_')[2] if len(model_name.split('_')) > 2 else 'unknown'
                file_size = model_path.stat().st_size / (1024 * 1024)
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="font-size: 2rem;">🎯</div>
                        <div class="metric-label">{region_name.upper()}</div>
                        <div class="metric-value" style="font-size: 1rem;">{file_size:.1f} MB</div>
                    </div>
                    """, unsafe_allow_html=True)

    if not params.enable_ml:
        st.info("Включите ML прогноз в боковой панели")
        return

    with st.spinner("Загрузка ML модели..."):
        try:
            predictor = get_ml_predictor(params.selected_region)

            if predictor.is_trained:
                _render_trained_model(predictor, params)
            else:
                _render_untrained_model(params)

        except Exception as e:
            st.error(f"Ошибка ML модели: {str(e)}")


def _render_trained_model(predictor: FloodMLPredictor, params: SidebarParams) -> None:
    """Отображение информации об обученной модели и прогноза."""
    st.success("✅ ML модель загружена и готова к работе")

    # Возможности модели
    st.markdown("### 🔧 Возможности модели")
    cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)
    with cap_col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Тип модели</div>
            <div class="metric-value" style="font-size: 1.2rem;">{predictor.model_type.upper()}</div>
        </div>
        ''', unsafe_allow_html=True)
    with cap_col2:
        cal_status = "Да" if predictor.use_calibration else "Нет"
        cal_color = "#4CAF50" if predictor.use_calibration else "var(--text-secondary)"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Калибровка</div>
            <div class="metric-value" style="font-size: 1.2rem; color: {cal_color};">{cal_status}</div>
        </div>
        ''', unsafe_allow_html=True)
    with cap_col3:
        fs_status = "Да" if predictor.use_feature_selection else "Нет"
        fs_color = "#4CAF50" if predictor.use_feature_selection else "var(--text-secondary)"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Отбор признаков</div>
            <div class="metric-value" style="font-size: 1.2rem; color: {fs_color};">{fs_status}</div>
        </div>
        ''', unsafe_allow_html=True)
    with cap_col4:
        stack_status = "Да" if predictor.stacking_model is not None else "Нет"
        stack_color = "#4CAF50" if predictor.stacking_model is not None else "var(--text-secondary)"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Стекинг</div>
            <div class="metric-value" style="font-size: 1.2rem; color: {stack_color};">{stack_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Метрики обучения
    if predictor.training_metrics:
        st.markdown("### 📊 Метрики модели")
        metrics_df = pd.DataFrame(predictor.training_metrics).T
        st.dataframe(
            metrics_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=None),
            width="stretch"
        )

        if predictor.best_params:
            st.markdown("### 🎛️ Оптимизированные гиперпараметры")
            for model_name, best_params in predictor.best_params.items():
                with st.expander(f"📦 {model_name.upper()}"):
                    st.json(best_params)

    # Важность признаков
    st.markdown("### 🎯 Важность признаков")
    importance = predictor.get_feature_importance(15)
    if not importance.empty:
        fig_imp = px.bar(
            importance, x='mean_importance', y='feature',
            orientation='h', title='Топ-15 важных признаков',
            color='mean_importance', color_continuous_scale='Blues'
        )
        fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, width="stretch")

    # Прогноз
    st.markdown("### 🔮 Прогноз")
    if st.session_state.get('input_data') is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            hours_ahead = st.slider("Часов вперед", 6, 72, 24)
        with col2:
            forecast_scenario = st.selectbox(
                "Сценарий прогноза",
                ['normal', 'wet', 'dry'],
                format_func=lambda x: {'normal': 'Обычный', 'wet': 'Влажный', 'dry': 'Сухой'}[x]
            )

        if st.button("🔮 Сделать прогноз", type="primary"):
            with st.spinner("Прогнозирование..."):
                forecast = predictor.predict_next_hours(
                    st.session_state.input_data,
                    hours_ahead=hours_ahead,
                    scenario=forecast_scenario
                )

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['timestamp'], y=forecast['flood_probability'],
                    mode='lines+markers', name='Вероятность паводка',
                    line=dict(color='#E53935', width=3),
                    fill='tozeroy', fillcolor='rgba(229, 57, 53, 0.2)'
                ))
                fig_forecast.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Порог 50%")
                fig_forecast.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Высокий риск")
                fig_forecast.update_layout(
                    title=f'Прогноз вероятности паводка на {hours_ahead} часов',
                    xaxis_title='Время', yaxis_title='Вероятность',
                    yaxis=dict(range=[0, 1]), height=400
                )
                st.plotly_chart(fig_forecast, width="stretch")

                st.dataframe(
                    forecast[['timestamp', 'flood_probability', 'risk_level', 'prediction_confidence']]
                    .style.format({'flood_probability': '{:.2%}', 'prediction_confidence': '{:.2%}'})
                    .background_gradient(subset=['flood_probability'], cmap='RdYlGn_r'),
                    width="stretch"
                )
    else:
        st.info("Сначала запустите симуляцию на вкладке 'Симуляция'")


def _render_untrained_model(params: SidebarParams) -> None:
    """Отображение интерфейса обучения модели."""
    st.warning("ML модель не обучена")

    models_dir = Path('models')
    available_models = list(models_dir.glob('flood_predictor_*_real.joblib')) if models_dir.exists() else []

    if available_models:
        st.info(f"📦 Найдено {len(available_models)} обученных моделей")
        with st.expander("📋 Доступные обученные модели"):
            for model_path in available_models:
                model_name = model_path.stem
                region_name = model_name.split('_')[2] if len(model_name.split('_')) > 2 else 'unknown'
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.text(f"🎯 {model_name}")
                with col2:
                    file_size = model_path.stat().st_size / (1024 * 1024)
                    st.text(f"📊 {file_size:.1f} MB")
                with col3:
                    if st.button("Загрузить", key=f"load_{model_name}"):
                        try:
                            loaded_predictor = FloodMLPredictor.load_model(str(model_path))
                            st.session_state[f'ml_predictor_{params.selected_region}'] = loaded_predictor
                            st.success(f"✅ Модель {region_name} загружена!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка загрузки: {e}")

    st.markdown("---")
    st.markdown("### 🎓 Обучить новую модель")

    data_source = st.radio(
        "Источник данных:",
        ["Синтетические данные (быстро)", "Реальные данные Open-Meteo (рекомендуется)"],
        key="data_source_radio"
    )

    if data_source.startswith("Синтетические"):
        _render_synthetic_training(params)
    else:
        _render_real_data_training(params)


def _render_synthetic_training(params: SidebarParams) -> None:
    """Обучение на синтетических данных."""
    st.info("💡 Синтетические данные хороши для быстрого тестирования")

    col1, col2 = st.columns(2)
    with col1:
        training_days = st.slider("Дней для обучения", 30, 730, 365)
    with col2:
        model_type = st.selectbox(
            "Тип модели",
            ['rf', 'xgb', 'lgb', 'ensemble'],
            index=3,
            format_func=lambda x: {
                'rf': 'Random Forest', 'xgb': 'XGBoost',
                'lgb': 'LightGBM', 'ensemble': 'Ансамбль (лучшее качество)'
            }[x]
        )

    if st.button("🎓 Обучить на синтетических данных", type="primary"):
        with st.spinner("Обучение модели (может занять время)..."):
            data = generate_training_data(
                region=params.selected_region, days=training_days, include_floods=True
            )
            predictor = FloodMLPredictor(model_type=model_type)
            metrics = predictor.train(data)

            model_path = f'models/flood_predictor_{params.selected_region}_synthetic.joblib'
            predictor.save_model(model_path)

            st.success("✅ Модель обучена на синтетических данных!")
            st.json(metrics)
            st.rerun()


def _render_real_data_training(params: SidebarParams) -> None:
    """Обучение на реальных данных Open-Meteo."""
    st.success("✅ Реальные данные из Open-Meteo API - лучший выбор для производства")

    flood_db_path = Path('data/FloodArchive.csv')
    has_flood_db = flood_db_path.exists()

    if has_flood_db:
        st.success("✅ База данных паводков DFO найдена")
    else:
        st.warning("⚠️ База данных паводков не найдена")
        st.info("Будет использован пороговый метод (менее точно)")
        with st.expander("ℹ️ Как получить базу данных паводков?"):
            st.markdown("""
            **Варианты получения исторических данных о паводках:**

            1. **Dartmouth Flood Observatory (DFO)**
               - Email: kettner@colorado.edu
               - Запросить FloodArchive.csv

            2. **EM-DAT**: https://www.emdat.be/
               - Регистрация бесплатна для исследователей

            3. **МЧС Кыргызстана**: https://mes.kg/
               - Наиболее точные локальные данные

            Подробнее см.: `data/README_FLOOD_DATA.md`
            """)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Начальная дата",
            value=pd.to_datetime("2020-01-01"),
            min_value=pd.to_datetime("2010-01-01"),
            max_value=pd.to_datetime("2023-12-31")
        )
    with col2:
        end_date = st.date_input(
            "Конечная дата",
            value=pd.to_datetime("2023-12-31"),
            min_value=pd.to_datetime("2010-01-01"),
            max_value=pd.to_datetime("2024-12-31")
        )

    col3, col4 = st.columns(2)
    with col3:
        model_type_real = st.selectbox(
            "Тип модели",
            ['rf', 'xgb', 'lgb', 'ensemble'],
            index=3,
            key="model_type_real",
            format_func=lambda x: {
                'rf': 'Random Forest', 'xgb': 'XGBoost',
                'lgb': 'LightGBM', 'ensemble': 'Ансамбль (рекомендуется)'
            }[x]
        )
    with col4:
        optimize = st.checkbox(
            "Оптимизация гиперпараметров",
            help="Улучшает качество, но занимает ~30+ минут"
        )

    days_count = (end_date - start_date).days
    hours_count = days_count * 24
    st.info(f"📊 Период обучения: {days_count} дней (~{hours_count:,} часовых записей)")

    if days_count < 30:
        st.warning("⚠️ Рекомендуется минимум 30 дней для обучения")
    elif days_count < 365:
        st.info("💡 Для лучших результатов рекомендуется 1+ год данных")
    else:
        st.success("✅ Отличный период для обучения!")

    if st.button("🚀 Обучить на реальных данных", type="primary"):
        with st.spinner(f"Обучение модели на реальных данных ({days_count} дней)..."):
            try:
                from src.models.ml_predictor import generate_training_data_real
                try:
                    from src.data.flood_database import load_kyrgyzstan_floods
                    _has_flood_db_import = True
                except ImportError:
                    _has_flood_db_import = False

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("📥 Загрузка базы данных паводков...")
                progress_bar.progress(10)

                flood_events = None
                if has_flood_db and _has_flood_db_import:
                    try:
                        flood_events = load_kyrgyzstan_floods('data/FloodArchive.csv')
                        st.info(f"✅ Загружено {len(flood_events)} событий паводков")
                    except Exception as e:
                        st.warning(f"Ошибка загрузки БД паводков: {e}")

                status_text.text("🌦️ Загрузка реальных метеоданных из Open-Meteo...")
                progress_bar.progress(30)

                training_data = generate_training_data_real(
                    region=params.selected_region,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    flood_events=flood_events
                )

                flood_count = training_data['flood'].sum() if 'flood' in training_data.columns else 0
                flood_pct = (flood_count / len(training_data)) * 100 if len(training_data) > 0 else 0

                st.info(f"✅ Загружено {len(training_data):,} записей")
                st.info(f"🌊 Паводковые часы: {flood_count} ({flood_pct:.2f}%)")

                if flood_count == 0:
                    st.error("❌ Нет паводковых событий в данных! Обучение невозможно.")
                    st.warning("Проверьте наличие базы данных паводков или выберите другой период.")
                    return

                status_text.text("🎓 Обучение ML модели...")
                progress_bar.progress(60)

                predictor = FloodMLPredictor(model_type=model_type_real)
                metrics = predictor.train(
                    training_data, target_column='flood', test_size=0.2,
                    use_feature_selection=True, optimize_hyperparameters=optimize
                )

                progress_bar.progress(90)

                status_text.text("💾 Сохранение модели...")
                model_path = f'models/flood_predictor_{params.selected_region}_real.joblib'
                Path('models').mkdir(exist_ok=True)
                predictor.save_model(model_path)

                progress_bar.progress(100)
                status_text.text("✅ Готово!")

                st.success("✅ Модель успешно обучена и сохранена!")
                st.success(f"📁 Файл: {model_path}")

                st.markdown("### 📊 Метрики производительности")
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                    with col2:
                        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")
                    with col3:
                        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2%}")
                else:
                    st.json(metrics)

                st.balloons()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Ошибка обучения: {str(e)}")
                st.exception(e)
