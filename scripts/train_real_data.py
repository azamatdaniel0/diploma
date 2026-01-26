#!/usr/bin/env python3
"""
Обучение модели прогнозирования паводков на РЕАЛЬНЫХ данных.

Этот скрипт:
1. Загружает реальные метеоданные из Open-Meteo API
2. Загружает исторические события паводков из базы данных DFO
3. Запускает гидрологическое моделирование
4. Обучает ML модель на реальных данных
5. Сохраняет обученную модель и метрики

Использование:
    python scripts/train_real_data.py
    python scripts/train_real_data.py --region osh --start 2015-01-01 --end 2024-12-31
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


def train_real_model(
    region: str = 'chui',
    start_date: str = '2018-01-01',
    end_date: str = '2023-12-31',
    model_type: str = 'ensemble',
    optimize_hyperparameters: bool = False,
    dfo_csv_path: str = 'data/FloodArchive.csv'
):
    """
    Обучение модели на реальных метеоданных и исторических паводках.

    Args:
        region: Регион Кыргызстана ('chui', 'issyk_kul', 'naryn', 'osh', 'jalal_abad', 'talas', 'batken')
        start_date: Начальная дата обучающих данных (формат: YYYY-MM-DD)
        end_date: Конечная дата обучающих данных (формат: YYYY-MM-DD)
        model_type: Тип модели ('rf', 'xgb', 'lgb', 'ensemble')
        optimize_hyperparameters: Выполнять ли оптимизацию гиперпараметров (медленнее, но лучше результаты)
        dfo_csv_path: Путь к базе данных паводков DFO

    Returns:
        Обученная модель и данные для обучения
    """
    print("=" * 60)
    print(f"ОБУЧЕНИЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ПАВОДКОВ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    print(f"Регион: {region}")
    print(f"Период: {start_date} - {end_date}")
    print(f"Тип модели: {model_type}")
    print(f"Оптимизация гиперпараметров: {'Да' if optimize_hyperparameters else 'Нет'}")
    print("=" * 60)

    # Шаг 1: Загрузка исторических событий паводков
    print(f"\n[1/5] Загрузка исторических событий паводков...")
    flood_events = None
    try:
        from src.data.flood_database import load_kyrgyzstan_floods, filter_floods_by_region

        dfo_path = Path(dfo_csv_path)
        if dfo_path.exists():
            logger.info(f"Загрузка базы данных паводков из {dfo_csv_path}")
            flood_events = load_kyrgyzstan_floods(dfo_csv_path)
            logger.info(f"✓ Найдено {len(flood_events)} событий паводков в базе данных")

            # Фильтрация по региону (опционально)
            # Примечание: DFO может не иметь детальной региональной информации для Кыргызстана
            # regional_floods = filter_floods_by_region(flood_events, region)
            # if len(regional_floods) > 0:
            #     flood_events = regional_floods
            #     logger.info(f"Отфильтровано {len(flood_events)} событий для региона {region}")

            # Показываем первые несколько событий
            print(f"\nПример событий паводков:")
            print(flood_events[['start_date', 'end_date', 'severity', 'Dead', 'Displaced']].head(10))

        else:
            logger.warning(f"⚠ База данных паводков не найдена: {dfo_csv_path}")
            logger.warning(f"  Скачайте FloodArchive.csv с:")
            logger.warning(f"  https://floodobservatory.colorado.edu/Archives/")
            logger.warning(f"  Сохраните в: {dfo_path.absolute()}")
            logger.warning(f"  Будет использован пороговый метод (менее точный)")

    except Exception as e:
        logger.error(f"Ошибка загрузки базы данных паводков: {e}")
        logger.warning("Продолжаем с пороговым методом")

    # Шаг 2: Генерация обучающих данных (реальные метеоданные + моделирование + метки)
    print(f"\n[2/5] Генерация обучающих данных...")
    print(f"  - Загрузка реальных метеоданных из Open-Meteo API")
    print(f"  - Запуск гидрологического моделирования")
    print(f"  - Присвоение меток паводков")
    print("  (это может занять несколько минут...)")

    try:
        training_data = generate_training_data_real(
            region=region,
            start_date=start_date,
            end_date=end_date,
            flood_events=flood_events,
            buffer_hours=24  # Расширяем окно на ±24 часа для учета периода формирования и спада паводка
        )

        logger.info(f"✓ Данные для обучения готовы: {len(training_data)} записей")

        # Статистика
        flood_count = training_data['flood'].sum()
        flood_pct = (flood_count / len(training_data)) * 100
        print(f"\nСтатистика данных:")
        print(f"  - Всего записей: {len(training_data)}")
        print(f"  - Паводковые часы: {flood_count} ({flood_pct:.2f}%)")
        print(f"  - Нормальные часы: {len(training_data) - flood_count} ({100 - flood_pct:.2f}%)")

        if flood_count == 0:
            logger.warning("⚠ ВНИМАНИЕ: Нет паводковых событий в данных!")
            logger.warning("  Модель не сможет обучиться распознавать паводки.")
            logger.warning("  Проверьте:")
            logger.warning("  1. Наличие базы данных паводков")
            logger.warning("  2. Соответствие периода данных периоду паводков")
            return None, None

    except Exception as e:
        logger.error(f"Ошибка генерации обучающих данных: {e}")
        raise

    # Шаг 3: Создание признаков (feature engineering)
    print(f"\n[3/5] Инженерия признаков...")
    try:
        from src.models.ml_predictor import FloodFeatureEngineering

        feature_eng = FloodFeatureEngineering()
        training_features = feature_eng.create_features(training_data)

        logger.info(f"✓ Создано {len(training_features.columns)} признаков")
        print(f"  - Количество признаков: {len(training_features.columns)}")

    except Exception as e:
        logger.error(f"Ошибка создания признаков: {e}")
        # Если feature engineering не работает, используем исходные данные
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
            use_time_series_cv=False,  # Используем стратифицированное разбиение для сбалансированности классов
            tune_hyperparameters=optimize_hyperparameters
        )

        logger.info("✓ Обучение завершено")

    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise

    # Шаг 5: Результаты и сохранение
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)

    print("\nМетрики производительности модели:")
    if isinstance(metrics, dict):
        # Если один тип модели
        if 'accuracy' in metrics:
            print(f"\n{model_type.upper()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
        else:
            # Если ансамбль с несколькими моделями
            for model_name, model_metrics in metrics.items():
                print(f"\n{model_name}:")
                for metric_name, value in model_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")

    # Сохранение модели
    model_path = Path('models') / f'flood_predictor_{region}_real.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        predictor.save_model(str(model_path))
        print(f"\n✓ Модель сохранена: {model_path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения модели: {e}")

    # Сохранение обучающих данных
    data_path = Path('data/training') / f'{region}_real_{start_date}_{end_date}.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        training_features.to_parquet(str(data_path), index=False)
        print(f"✓ Обучающие данные сохранены: {data_path}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить обучающие данные: {e}")

    # Важность признаков
    print("\n" + "=" * 60)
    print("ТОП-15 НАИБОЛЕЕ ВАЖНЫХ ПРИЗНАКОВ:")
    print("=" * 60)

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

    print("\n" + "=" * 60)
    print("ГОТОВО! Модель готова к использованию.")
    print("=" * 60)
    print(f"\nИспользование модели:")
    print(f"  from src.models.ml_predictor import FloodMLPredictor")
    print(f"  predictor = FloodMLPredictor.load_model('{model_path}')")
    print(f"  predictions = predictor.predict(your_data)")

    return predictor, training_features


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Обучение модели прогнозирования паводков на реальных данных'
    )

    parser.add_argument(
        '--region',
        type=str,
        default='chui',
        choices=['chui', 'issyk_kul', 'naryn', 'osh', 'jalal_abad', 'talas', 'batken'],
        help='Регион Кыргызстана (по умолчанию: chui)'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2018-01-01',
        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 2018-01-01)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2023-12-31',
        help='Конечная дата в формате YYYY-MM-DD (по умолчанию: 2023-12-31)'
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
        predictor, data = train_real_model(
            region=args.region,
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
