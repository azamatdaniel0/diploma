"""
Обучение ML-модели на реальном датасете МЧС + Open-Meteo.

Предварительно: запустите build_dataset.py для создания датасета.

Запуск:
    cd /home/ksw-pc/claude_docs/diploma
    source venv/bin/activate

    # Обучить на всех регионах
    python scripts/train_model.py

    # Только для региона Чуй
    python scripts/train_model.py --region chui

    # С выбором типа модели
    python scripts/train_model.py --model rf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ml_predictor import train_on_real_dataset


def main():
    parser = argparse.ArgumentParser(description='Обучение модели паводков')
    parser.add_argument('--dataset', default='data/kyrgyzstan_floods_dataset.csv')
    parser.add_argument('--region', default=None,
                        help='Регион: chui/issyk_kul/naryn/osh/jalal_abad/talas/batken')
    parser.add_argument('--model', default='ensemble',
                        choices=['rf', 'gb', 'mlp', 'ensemble'])
    parser.add_argument('--output', default=None,
                        help='Путь сохранения модели')
    args = parser.parse_args()

    print('=' * 60)
    print('Обучение модели прогнозирования паводков')
    print(f'Датасет: {args.dataset}')
    print(f'Регион: {args.region or "все"}')
    print(f'Модель: {args.model}')
    print('=' * 60)

    try:
        predictor, metrics = train_on_real_dataset(
            dataset_path=args.dataset,
            region=args.region,
            model_type=args.model,
            save_path=args.output,
        )

        print('\n' + '=' * 60)
        print('Важность признаков (топ-15):')
        fi = predictor.get_feature_importance(15)
        if fi is not None:
            print(fi.to_string(index=False))

    except FileNotFoundError as e:
        print(f'\nОшибка: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\nОшибка обучения: {e}')
        raise


if __name__ == '__main__':
    main()
