#!/usr/bin/env python3
"""
Обучение единой модели прогнозирования паводков для всего Кыргызстана.

Объединяет данные всех регионов для создания более робастной модели.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import argparse
import logging
from datetime import datetime

from src.models.ml_predictor import (
    FloodMLPredictor,
    generate_training_data_real
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_unified_model(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    model_type: str = 'ensemble',
    optimize_hyperparameters: bool = False,
    dfo_csv_path: str = 'data/FloodArchive.csv'
):
    """
    Обучение единой модели на данных всех регионов Кыргызстана.

    Args:
        start_date: Начальная дата обучающих данных (формат: YYYY-MM-DD)
        end_date: Конечная дата обучающих данных (формат: YYYY-MM-DD)
        model_type: Тип модели ('rf', 'xgb', 'lgb', 'ensemble')
        optimize_hyperparameters: Выполнять ли оптимизацию гиперпараметров
        dfo_csv_path: Путь к базе данных паводков DFO

    Returns:
        Обученная модель и объединенные данные
    """
    print("=" * 80)
    print(f"ОБУЧЕНИЕ ЕДИНОЙ МОДЕЛИ ДЛЯ ВСЕГО КЫРГЫЗСТАНА")
    print("=" * 80)
    print(f"Период: {start_date} - {end_date}")
    print(f"Тип модели: {model_type}")
    print(f"Оптимизация гиперпараметров: {'Да' if optimize_hyperparameters else 'Нет'}")
    print("=" * 80)

    # Все регионы Кыргызстана
    regions = ['chui', 'issyk_kul', 'naryn', 'osh', 'jalal_abad', 'talas', 'batken']

    # Шаг 1: Загрузка исторических событий паводков
    print(f"\n[1/5] Загрузка исторических событий паводков...")
    flood_events = None
    try:
        from src.data.flood_database import load_kyrgyzstan_floods

        dfo_path = Path(dfo_csv_path)
        if dfo_path.exists():
            logger.info(f"Загрузка базы данных паводков из {dfo_csv_path}")
            flood_events = load_kyrgyzstan_floods(dfo_csv_path)
            logger.info(f"✓ Найдено {len(flood_events)} событий паводков в базе данных")

            # Показываем статистику
            print(f"\nСтатистика паводков:")
            print(f"  - Всего событий: {len(flood_events)}")
            if 'severity' in flood_events.columns:
                print(f"  - По уровню серьезности:")
                for severity, count in flood_events['severity'].value_counts().items():
                    print(f"    • {severity}: {count}")

        else:
            logger.warning(f"⚠ База данных паводков не найдена: {dfo_csv_path}")
            logger.warning(f"  Будет использован пороговый метод (менее точный)")

    except Exception as e:
        logger.error(f"Ошибка загрузки базы данных паводков: {e}")
        logger.warning("Продолжаем с пороговым методом")

    # Шаг 2: Генерация обучающих данных для всех регионов
    print(f"\n[2/5] Генерация обучающих данных для {len(regions)} регионов...")
    print(f"  - Загрузка реальных метеоданных из Open-Meteo API")
    print(f"  - Запуск гидрологического моделирования")
    print(f"  - Присвоение меток паводков")
    print("  (это может занять 10-15 минут...)")

    all_training_data = []

    for i, region in enumerate(regions, 1):
        print(f"\n  [{i}/{len(regions)}] Обработка региона: {region.upper()}")

        try:
            region_data = generate_training_data_real(
                region=region,
                start_date=start_date,
                end_date=end_date,
                flood_events=flood_events,
                buffer_hours=24
            )

            # Добавляем идентификатор региона
            region_data['region'] = region

            all_training_data.append(region_data)

            flood_count = region_data['flood'].sum()
            logger.info(f"    ✓ {region}: {len(region_data)} записей, {flood_count} паводковых часов")

        except Exception as e:
            logger.error(f"    ✗ Ошибка обработки региона {region}: {e}")
            continue

    if len(all_training_data) == 0:
        logger.error("Не удалось загрузить данные ни для одного региона!")
        return None, None

    # Объединение всех данных
    print(f"\n  Объединение данных из {len(all_training_data)} регионов...")
    training_data = pd.concat(all_training_data, ignore_index=True)

    logger.info(f"✓ Объединенные данные готовы: {len(training_data)} записей")

    # Общая статистика
    flood_count = training_data['flood'].sum()
    flood_pct = (flood_count / len(training_data)) * 100

    print(f"\n{'=' * 80}")
    print(f"ОБЩАЯ СТАТИСТИКА ДАННЫХ:")
    print(f"{'=' * 80}")
    print(f"  - Всего записей: {len(training_data):,}")
    print(f"  - Паводковые часы: {flood_count:,} ({flood_pct:.2f}%)")
    print(f"  - Нормальные часы: {len(training_data) - flood_count:,} ({100 - flood_pct:.2f}%)")

    # Статистика по регионам
    print(f"\n  Распределение по регионам:")
    for region in regions:
        region_rows = training_data[training_data['region'] == region]
        if len(region_rows) > 0:
            region_floods = region_rows['flood'].sum()
            print(f"    • {region:12s}: {len(region_rows):6,} записей, {region_floods:4,} паводков")

    if flood_count == 0:
        logger.warning("⚠ ВНИМАНИЕ: Нет паводковых событий в данных!")
        logger.warning("  Модель не сможет обучиться распознавать паводки.")
        return None, None

    # Шаг 3: Создание признаков
    print(f"\n[3/5] Инженерия признаков...")
    try:
        from src.models.ml_predictor import FloodFeatureEngineering

        feature_eng = FloodFeatureEngineering()
        training_features = feature_eng.create_features(training_data)

        logger.info(f"✓ Создано {len(training_features.columns)} признаков")
        print(f"  - Количество признаков: {len(training_features.columns)}")

    except Exception as e:
        logger.error(f"Ошибка создания признаков: {e}")
        logger.warning("Используем исходные данные без дополнительных признаков")
        training_features = training_data

    # Шаг 4: Обучение модели
    print(f"\n[4/5] Обучение ML модели...")
    print(f"  Модель: {model_type}")
    if optimize_hyperparameters:
        print(f"  ⚠ Оптимизация гиперпараметров включена (может занять 30+ минут)")

    try:
        predictor = FloodMLPredictor(model_type=model_type)

        metrics = predictor.train(
            training_features,
            test_size=0.2,
            use_time_series_cv=False,
            tune_hyperparameters=optimize_hyperparameters
        )

        logger.info("✓ Обучение завершено")

    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise

    # Шаг 5: Результаты и сохранение
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)

    print("\nМетрики производительности модели:")
    if isinstance(metrics, dict):
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric_name, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")

    # Сохранение модели
    model_path = Path('models') / 'flood_predictor_kyrgyzstan_unified.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        predictor.save_model(str(model_path))
        print(f"\n✓ Модель сохранена: {model_path}")
        print(f"  Размер: {model_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Ошибка сохранения модели: {e}")

    # Сохранение обучающих данных
    data_path = Path('data/training') / f'kyrgyzstan_unified_{start_date}_{end_date}.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Удаляем колонку region перед сохранением (если есть проблемы с типами)
        save_data = training_features.drop(columns=['region'], errors='ignore')
        save_data.to_parquet(str(data_path), index=False)
        print(f"✓ Обучающие данные сохранены: {data_path}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить обучающие данные: {e}")

    # Важность признаков
    print("\n" + "=" * 80)
    print("ТОП-15 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ:")
    print("=" * 80)

    try:
        importance = predictor.get_feature_importance(top_n=15)
        if importance is not None and len(importance) > 0:
            for idx, row in importance.iterrows():
                feature_name = row['feature'] if 'feature' in row else 'Unknown'
                importance_value = row.get('mean_importance', row.get('importance', 0))
                print(f"{idx + 1:2d}. {feature_name:40s} {importance_value:.4f}")
        else:
            print("Важность признаков недоступна для этого типа модели")
    except Exception as e:
        logger.warning(f"Не удалось получить важность признаков: {e}")

    print("\n" + "=" * 80)
    print("ГОТОВО! Единая модель для всего Кыргызстана готова к использованию.")
    print("=" * 80)
    print(f"\nИспользование модели:")
    print(f"  from src.models.ml_predictor import FloodMLPredictor")
    print(f"  predictor = FloodMLPredictor()")
    print(f"  predictor.load_model('{model_path}')")
    print(f"  predictions = predictor.predict(your_data)")

    return predictor, training_features


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Обучение единой модели прогнозирования паводков для всего Кыргызстана'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 2020-01-01)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='Конечная дата в формате YYYY-MM-DD (по умолчанию: 2024-12-31)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='ensemble',
        choices=['rf', 'xgb', 'lgb', 'ensemble'],
        help='Тип модели (по умолчанию: ensemble)'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Выполнить оптимизацию гиперпараметров (медленнее, но лучше качество)'
    )

    parser.add_argument(
        '--dfo-path',
        type=str,
        default='data/FloodArchive.csv',
        help='Путь к базе данных DFO (по умолчанию: data/FloodArchive.csv)'
    )

    args = parser.parse_args()

    # Проверка формата дат
    try:
        datetime.strptime(args.start, '%Y-%m-%d')
        datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        print("Ошибка: Даты должны быть в формате YYYY-MM-DD")
        sys.exit(1)

    # Запуск обучения
    try:
        predictor, data = train_unified_model(
            start_date=args.start,
            end_date=args.end,
            model_type=args.model,
            optimize_hyperparameters=args.optimize,
            dfo_csv_path=args.dfo_path
        )

        if predictor is None:
            print("\n⚠ Обучение не удалось. Проверьте логи выше.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        logger.exception("Полная трассировка ошибки:")
        sys.exit(1)


if __name__ == '__main__':
    main()
