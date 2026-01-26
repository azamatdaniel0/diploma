#!/usr/bin/env python3
"""
Пример использования обученной модели прогнозирования паводков.

Этот скрипт демонстрирует:
1. Загрузку обученной модели
2. Получение прогноза погоды
3. Запуск гидрологического моделирования
4. Прогнозирование паводков
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta

from src.models.ml_predictor import FloodMLPredictor
from src.data.weather_api import KyrgyzstanWeatherLoader
from src.models.flood_model import FloodModel


def predict_flood_risk(
    region: str = 'chui',
    model_path: str = 'models/flood_predictor_chui_real.joblib',
    forecast_days: int = 7
):
    """
    Прогнозирование риска паводка на ближайшие дни.

    Args:
        region: Регион Кыргызстана
        model_path: Путь к обученной модели
        forecast_days: Количество дней для прогноза
    """
    print("=" * 80)
    print(f"ПРОГНОЗИРОВАНИЕ РИСКА ПАВОДКА")
    print("=" * 80)
    print(f"Регион: {region}")
    print(f"Период прогноза: {forecast_days} дней")
    print("=" * 80)

    # 1. Загрузка модели
    print(f"\n[1/4] Загрузка обученной модели...")
    predictor = FloodMLPredictor()
    predictor.load_model(model_path)
    print(f"✓ Модель загружена: {Path(model_path).name}")

    # 2. Получение данных о погоде
    print(f"\n[2/4] Загрузка данных о погоде...")

    # Получаем данные за последние 7 дней (для контекста) + прогноз
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    weather_loader = KyrgyzstanWeatherLoader(region=region, use_cache=True)

    try:
        # Загружаем исторические данные
        weather_df = weather_loader.get_historical(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            hourly=True
        )
        print(f"✓ Загружено {len(weather_df)} часовых записей")

        # В реальном приложении здесь можно добавить прогноз погоды
        # Пока используем последние данные

    except Exception as e:
        print(f"✗ Ошибка загрузки данных: {e}")
        return None

    # 3. Запуск гидрологического моделирования
    print(f"\n[3/4] Гидрологическое моделирование...")

    flood_model = FloodModel(region=region)
    flood_model.initialize_state(weather_df['timestamp'].iloc[0])

    simulation_results = flood_model.run_simulation(weather_df, dt_hours=1)

    print(f"✓ Моделирование завершено")
    print(f"  Максимальный расход: {simulation_results['discharge_m3s'].max():.1f} м³/с")
    print(f"  Средний расход: {simulation_results['discharge_m3s'].mean():.1f} м³/с")

    # 4. Прогнозирование паводков
    print(f"\n[4/4] Прогнозирование рисков...")

    predictions_df = predictor.predict(simulation_results)

    # Анализ результатов
    risk_hours = predictions_df['flood_predicted'].sum()
    risk_percentage = (risk_hours / len(predictions_df)) * 100

    print(f"\n" + "=" * 80)
    print(f"РЕЗУЛЬТАТЫ ПРОГНОЗА")
    print("=" * 80)
    print(f"Анализируемый период: {simulation_results['timestamp'].min()} - {simulation_results['timestamp'].max()}")
    print(f"Всего часов: {len(predictions_df)}")
    print(f"Часов с риском паводка: {risk_hours} ({risk_percentage:.1f}%)")

    if risk_hours > 0:
        print(f"\n⚠️  ОБНАРУЖЕН РИСК ПАВОДКА!")

        # Найдем периоды риска
        simulation_results['flood_risk'] = predictions_df['flood_predicted'].values
        risk_periods = simulation_results[simulation_results['flood_risk'] == 1]

        print(f"\nПериоды риска:")

        # Группируем последовательные часы
        current_start = None
        current_end = None

        for idx, row in risk_periods.iterrows():
            if current_start is None:
                current_start = row['timestamp']
                current_end = row['timestamp']
            elif (row['timestamp'] - current_end).total_seconds() <= 3600:  # Consecutive hour
                current_end = row['timestamp']
            else:
                # Print previous period
                duration = (current_end - current_start).total_seconds() / 3600 + 1
                print(f"  • {current_start} - {current_end} (продолжительность: {duration:.0f} часов)")
                current_start = row['timestamp']
                current_end = row['timestamp']

        # Print last period
        if current_start is not None:
            duration = (current_end - current_start).total_seconds() / 3600 + 1
            print(f"  • {current_start} - {current_end} (продолжительность: {duration:.0f} часов)")

        print(f"\nРекомендации:")
        print(f"  1. Усилить мониторинг уровней рек")
        print(f"  2. Подготовить системы оповещения")
        print(f"  3. Проверить готовность инфраструктуры")

    else:
        print(f"\n✓ Риск паводка не обнаружен")
        print(f"  Условия в норме")

    print("\n" + "=" * 80)

    # Показываем последние 24 часа детально
    print(f"\nДетальный прогноз (последние 24 часа):")
    print("=" * 80)

    last_24h = simulation_results.tail(24)

    print(f"{'Время':19s} {'Осадки':>8s} {'Темп':>6s} {'Расход':>8s} {'Риск':>6s}")
    print(f"{'':19s} {'(мм)':>8s} {'(°C)':>6s} {'(м³/с)':>8s} {'':>6s}")
    print("-" * 80)

    for i, (idx, row) in enumerate(last_24h.iterrows()):
        time_str = str(row['timestamp'])
        precip = row.get('precipitation_mm', 0)
        temp = row.get('temperature_c', 0)
        discharge = row['discharge_m3s']
        # Get prediction for this row
        pred_idx = len(predictions_df) - 24 + i
        risk = '⚠️ ДА' if predictions_df.iloc[pred_idx]['flood_predicted'] == 1 else '✓ НЕТ'

        print(f"{time_str:19s} {precip:8.2f} {temp:6.1f} {discharge:8.1f} {risk:>6s}")

    return simulation_results, predictions_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Прогнозирование риска паводка')
    parser.add_argument('--region', type=str, default='chui', help='Регион Кыргызстана')
    parser.add_argument('--model', type=str, default='models/flood_predictor_chui_real.joblib',
                       help='Путь к модели')
    parser.add_argument('--days', type=int, default=7, help='Дней для прогноза')

    args = parser.parse_args()

    results, predictions = predict_flood_risk(
        region=args.region,
        model_path=args.model,
        forecast_days=args.days
    )
