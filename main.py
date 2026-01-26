#!/usr/bin/env python3
"""
Главный скрипт для запуска моделирования паводков в Кыргызстане.

Использование:
    python main.py --region chui --scenario normal --days 365

Автор: [Ваше имя]
Дипломная работа: Компьютерное моделирование влияния осадков
                  на возникновение паводков в районах Кыргызстана
"""

import argparse
import sys
import os
from datetime import datetime

# Добавление пути к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import REGIONS, RESULTS_DIR
from src.data.precipitation import PrecipitationDataLoader
from src.data.terrain import TerrainAnalyzer, create_sample_terrain
from src.models.flood_model import FloodModel
from src.models.scs_cn import SCSCurveNumberModel
from src.models.snowmelt import SnowmeltModel
from src.visualization.plots import (
    create_dashboard, plot_simulation_results,
    plot_flood_events, plot_terrain
)
from src.utils.helpers import (
    create_output_directory, save_results,
    calculate_statistics, format_report, generate_sample_data
)


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Моделирование паводков в регионах Кыргызстана'
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        default='chui',
        choices=list(REGIONS.keys()),
        help='Регион для моделирования'
    )

    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default='normal',
        choices=['normal', 'wet', 'dry', 'extreme'],
        help='Сценарий осадков'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=365,
        help='Число дней моделирования'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Директория для результатов'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Не создавать графики'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )

    return parser.parse_args()


def run_simulation(
    region: str,
    scenario: str,
    days: int,
    output_dir: str,
    create_plots: bool = True,
    verbose: bool = False
):
    """
    Основная функция моделирования.

    Args:
        region: Код региона
        scenario: Сценарий осадков
        days: Число дней
        output_dir: Директория для результатов
        create_plots: Создавать графики
        verbose: Подробный вывод
    """
    region_config = REGIONS[region]

    print("=" * 60)
    print("МОДЕЛИРОВАНИЕ ПАВОДКОВ В КЫРГЫЗСТАНЕ")
    print("=" * 60)
    print(f"Регион: {region_config.name}")
    print(f"Сценарий: {scenario}")
    print(f"Период: {days} дней")
    print(f"Директория результатов: {output_dir}")
    print("=" * 60)
    print()

    # 1. Генерация входных данных
    print("1. Генерация входных данных...")
    precipitation_data = generate_sample_data(
        region=region,
        days=days,
        scenario=scenario
    )

    if verbose:
        print(f"   Сгенерировано {len(precipitation_data)} записей")
        print(f"   Средние осадки: {precipitation_data['precipitation_mm'].mean():.2f} мм/день")
        print(f"   Макс. осадки: {precipitation_data['precipitation_mm'].max():.1f} мм/день")

    # 2. Создание и инициализация модели
    print("2. Инициализация модели паводков...")
    model = FloodModel(region=region)
    model.initialize_state(
        timestamp=precipitation_data['timestamp'].iloc[0],
        soil_moisture=0.4,  # Начальная влажность почвы
        snow_water_equivalent=50 if precipitation_data['temperature_c'].iloc[0] < 5 else 0
    )

    if verbose:
        info = model.get_model_info()
        print(f"   Площадь водосбора: {info['watershed_area_km2']:.0f} км²")
        print(f"   Пороговый расход: {info['flood_threshold_m3s']:.0f} м³/с")

    # 3. Запуск моделирования
    print("3. Запуск моделирования...")
    results = model.run_simulation(precipitation_data, dt_hours=24)

    if verbose:
        print(f"   Расчетных шагов: {len(results)}")
        print(f"   Макс. расход: {results['discharge_m3s'].max():.1f} м³/с")
        print(f"   Дней с превышением порога: {(results['discharge_m3s'] > model.params.flood_discharge_threshold).sum()}")

    # 4. Выявление паводковых событий
    print("4. Анализ паводковых событий...")
    events = model.detect_flood_events(results, min_duration_hours=6)
    print(f"   Обнаружено событий: {len(events)}")

    if verbose and events:
        for i, event in enumerate(events[:5], 1):
            print(f"   Событие {i}: пик {event.peak_discharge_m3s:.1f} м³/с, "
                  f"риск: {event.risk_level.value}")
        if len(events) > 5:
            print(f"   ... и еще {len(events) - 5} событий")

    # 5. Расчет статистики
    print("5. Расчет статистики...")
    stats = calculate_statistics(results)
    flood_stats = model.calculate_flood_statistics()

    # 6. Сохранение результатов
    print("6. Сохранение результатов...")
    save_results(results, output_dir, 'simulation_results', formats=['csv'])

    # Сохранение статистики
    stats_combined = {**stats, 'flood_statistics': flood_stats}
    import json
    with open(os.path.join(output_dir, 'data', 'statistics.json'), 'w') as f:
        json.dump(stats_combined, f, indent=2, default=str)

    # Сохранение отчета
    report = format_report(stats, events, region)
    with open(os.path.join(output_dir, 'reports', 'report.txt'), 'w') as f:
        f.write(report)
    print(f"   Отчет сохранен в {output_dir}/reports/report.txt")

    # 7. Создание визуализаций
    if create_plots:
        print("7. Создание визуализаций...")
        figures_dir = os.path.join(output_dir, 'figures')

        try:
            # Результаты моделирования
            fig = plot_simulation_results(results)
            fig.savefig(os.path.join(figures_dir, 'simulation_results.png'),
                       dpi=150, bbox_inches='tight')
            print(f"   Сохранено: simulation_results.png")

            # Паводковые события
            if events:
                fig = plot_flood_events(events)
                fig.savefig(os.path.join(figures_dir, 'flood_events.png'),
                           dpi=150, bbox_inches='tight')
                print(f"   Сохранено: flood_events.png")

            # Информационная панель
            fig = create_dashboard(results, events, region=region)
            fig.savefig(os.path.join(figures_dir, 'dashboard.png'),
                       dpi=150, bbox_inches='tight')
            print(f"   Сохранено: dashboard.png")

            # Закрытие всех фигур для освобождения памяти
            import matplotlib.pyplot as plt
            plt.close('all')

        except Exception as e:
            print(f"   Ошибка при создании графиков: {e}")

    # 8. Вывод отчета
    print()
    print(report)

    return results, events, stats


def run_scenario_comparison(
    region: str,
    days: int,
    output_dir: str
):
    """
    Сравнение различных сценариев.

    Args:
        region: Код региона
        days: Число дней
        output_dir: Директория для результатов
    """
    scenarios = ['normal', 'wet', 'dry', 'extreme']
    all_results = {}

    print("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ СЦЕНАРИЕВ")
    print("=" * 60)

    for scenario in scenarios:
        print(f"\nСценарий: {scenario}")
        print("-" * 40)

        # Генерация данных
        precip_data = generate_sample_data(region, days, scenario)

        # Моделирование
        model = FloodModel(region=region)
        model.initialize_state(precip_data['timestamp'].iloc[0])
        results = model.run_simulation(precip_data)
        events = model.detect_flood_events(results)

        all_results[scenario] = {
            'total_precipitation': precip_data['precipitation_mm'].sum(),
            'max_discharge': results['discharge_m3s'].max(),
            'mean_discharge': results['discharge_m3s'].mean(),
            'flood_events': len(events),
            'max_event_discharge': max([e.peak_discharge_m3s for e in events]) if events else 0
        }

        print(f"  Осадки (сумма): {all_results[scenario]['total_precipitation']:.0f} мм")
        print(f"  Макс. расход: {all_results[scenario]['max_discharge']:.1f} м³/с")
        print(f"  Паводковых событий: {all_results[scenario]['flood_events']}")

    # Сохранение сравнения
    import json
    with open(os.path.join(output_dir, 'data', 'scenario_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Сравнение завершено. Результаты сохранены в scenario_comparison.json")


def main():
    """Главная функция."""
    args = parse_arguments()

    # Создание директории для результатов
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        for subdir in ['figures', 'data', 'maps', 'reports']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    else:
        output_dir = create_output_directory()

    print(f"Результаты будут сохранены в: {output_dir}")
    print()

    try:
        # Запуск основного моделирования
        results, events, stats = run_simulation(
            region=args.region,
            scenario=args.scenario,
            days=args.days,
            output_dir=output_dir,
            create_plots=not args.no_plots,
            verbose=args.verbose
        )

        # Сравнительный анализ сценариев (опционально)
        if args.scenario == 'normal' and args.days >= 30:
            response = input("\nВыполнить сравнительный анализ сценариев? (y/n): ")
            if response.lower() == 'y':
                run_scenario_comparison(args.region, args.days, output_dir)

        print("\nМоделирование завершено успешно!")
        print(f"Результаты: {output_dir}")

    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
