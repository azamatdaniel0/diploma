"""
Машинное обучение для прогнозирования паводков.

Включает:
- Ансамблевые модели (Random Forest, XGBoost, LightGBM)
- Feature engineering для гидрологических данных
- Прогноз вероятности паводка
- Оценка важности признаков
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings
import joblib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost не установлен. Используйте pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM не установлен. Используйте pip install lightgbm")


class FloodPredictionTarget(Enum):
    """Целевые переменные для прогноза."""
    BINARY = "binary"  # Паводок да/нет
    RISK_LEVEL = "risk_level"  # Уровень риска (4 класса)
    DISCHARGE = "discharge"  # Регрессия расхода


@dataclass
class PredictionResult:
    """Результат прогноза."""
    timestamp: datetime
    flood_probability: float
    risk_level: str
    predicted_discharge: float
    confidence: float
    contributing_factors: Dict[str, float]


class FloodFeatureEngineering:
    """
    Инженерия признаков для прогноза паводков.
    """

    def __init__(self, window_sizes: List[int] = None):
        """
        Args:
            window_sizes: Размеры окон для скользящих статистик (в часах)
        """
        self.window_sizes = window_sizes or [6, 12, 24, 48, 72]
        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков из временного ряда.

        Args:
            df: DataFrame с колонками timestamp, precipitation_mm, temperature_c

        Returns:
            DataFrame с признаками
        """
        features = df.copy()

        # Базовые признаки времени
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
            features['day_of_year'] = pd.to_datetime(features['timestamp']).dt.dayofyear
            features['month'] = pd.to_datetime(features['timestamp']).dt.month
            features['week'] = pd.to_datetime(features['timestamp']).dt.isocalendar().week
            features['is_spring'] = features['month'].isin([3, 4, 5]).astype(int)
            features['is_summer'] = features['month'].isin([6, 7, 8]).astype(int)

        # Циклические признаки для сезонности
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)

        # Скользящие статистики осадков
        if 'precipitation_mm' in features.columns:
            for window in self.window_sizes:
                features[f'precip_sum_{window}h'] = (
                    features['precipitation_mm'].rolling(window, min_periods=1).sum()
                )
                features[f'precip_max_{window}h'] = (
                    features['precipitation_mm'].rolling(window, min_periods=1).max()
                )
                features[f'precip_mean_{window}h'] = (
                    features['precipitation_mm'].rolling(window, min_periods=1).mean()
                )
                features[f'precip_std_{window}h'] = (
                    features['precipitation_mm'].rolling(window, min_periods=1).std().fillna(0)
                )

            # Интенсивность осадков
            features['precip_intensity'] = features['precipitation_mm']
            features['precip_change'] = features['precipitation_mm'].diff().fillna(0)
            features['precip_acceleration'] = features['precip_change'].diff().fillna(0)

        # Температурные признаки
        if 'temperature_c' in features.columns:
            for window in self.window_sizes:
                features[f'temp_mean_{window}h'] = (
                    features['temperature_c'].rolling(window, min_periods=1).mean()
                )

            features['temp_above_zero'] = (features['temperature_c'] > 0).astype(int)
            features['temp_change'] = features['temperature_c'].diff().fillna(0)

            # Индекс таяния снега
            features['melt_index'] = np.maximum(0, features['temperature_c']) * features['temp_above_zero']
            for window in [24, 48, 72]:
                features[f'melt_sum_{window}h'] = (
                    features['melt_index'].rolling(window, min_periods=1).sum()
                )

        # Индекс предшествующих осадков (API)
        features['api'] = self._calculate_api(features['precipitation_mm'])

        # Комбинированные признаки
        if 'precipitation_mm' in features.columns and 'temperature_c' in features.columns:
            features['precip_temp_interaction'] = (
                features['precipitation_mm'] * np.maximum(0, features['temperature_c'])
            )
            features['rain_fraction'] = np.where(
                features['temperature_c'] > 2, 1,
                np.where(features['temperature_c'] < 0, 0, (features['temperature_c'] + 0) / 2)
            )

        # Лаговые признаки
        for lag in [1, 2, 3, 6, 12, 24]:
            if 'precipitation_mm' in features.columns:
                features[f'precip_lag_{lag}'] = features['precipitation_mm'].shift(lag).fillna(0)
            if 'temperature_c' in features.columns:
                features[f'temp_lag_{lag}'] = features['temperature_c'].shift(lag).fillna(features['temperature_c'].mean())

        # Сохранение имен признаков
        self.feature_names = [col for col in features.columns
                             if col not in ['timestamp', 'target', 'risk_level', 'discharge_m3s']]

        return features

    def _calculate_api(self, precipitation: pd.Series, k: float = 0.85) -> pd.Series:
        """
        Расчет индекса предшествующих осадков.

        Args:
            precipitation: Ряд осадков
            k: Коэффициент затухания

        Returns:
            Ряд API
        """
        api = np.zeros(len(precipitation))
        for i in range(1, len(precipitation)):
            api[i] = precipitation.iloc[i] + k * api[i - 1]
        return pd.Series(api, index=precipitation.index)

    def get_feature_names(self) -> List[str]:
        """Получение списка имен признаков."""
        return self.feature_names


class FloodMLPredictor:
    """
    Предиктор паводков на основе машинного обучения.

    Поддерживает:
    - Ансамблевые модели (Random Forest, XGBoost, LightGBM, Gradient Boosting)
    - Нейронные сети (MLP)
    - Стекинг моделей
    - Гиперпараметрическая оптимизация (GridSearchCV, RandomizedSearchCV)
    - Калибровка вероятностей
    - Отбор признаков (SelectFromModel, RFE)
    """

    def __init__(
        self,
        model_type: str = "ensemble",
        target: FloodPredictionTarget = FloodPredictionTarget.BINARY,
        use_calibration: bool = False,
        use_feature_selection: bool = False,
        feature_selection_method: str = "selectfrommodel"
    ):
        """
        Args:
            model_type: Тип модели ('rf', 'xgb', 'lgb', 'gb', 'mlp', 'ensemble', 'stacking')
            target: Целевая переменная
            use_calibration: Применять калибровку вероятностей (CalibratedClassifierCV)
            use_feature_selection: Применять отбор признаков
            feature_selection_method: Метод отбора ('selectfrommodel', 'rfe')
        """
        self.model_type = model_type
        self.target = target
        self.use_calibration = use_calibration
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method

        self.feature_engineer = FloodFeatureEngineering()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_feature_indices: Optional[np.ndarray] = None

        self.models: Dict = {}
        self.calibrated_models: Dict = {}
        self.stacking_model = None
        self.is_trained = False
        self.feature_importance: Optional[pd.DataFrame] = None
        self.training_metrics: Dict = {}
        self.best_params: Dict = {}

        self._initialize_models()

    def _initialize_models(self):
        """Инициализация моделей."""
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        )

        # MLP Neural Network
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )

        # XGBoost
        if HAS_XGBOOST:
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,
                random_state=42,
                eval_metric='logloss'
            )

        # LightGBM
        if HAS_LIGHTGBM:
            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )

    def _get_param_grid(self, model_name: str) -> Dict:
        """
        Получение сетки гиперпараметров для модели.

        Args:
            model_name: Название модели

        Returns:
            Словарь с параметрами для GridSearchCV
        """
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gb': {
                'n_estimators': [100, 150, 200],
                'max_depth': [5, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'min_samples_split': [2, 5, 10]
            },
            'mlp': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            },
            'xgb': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            },
            'lgb': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [20, 31, 50]
            }
        }
        return param_grids.get(model_name, {})

    def _get_param_distributions(self, model_name: str) -> Dict:
        """
        Получение распределений гиперпараметров для RandomizedSearchCV.

        Args:
            model_name: Название модели

        Returns:
            Словарь с распределениями параметров
        """
        param_distributions = {
            'rf': {
                'n_estimators': randint(100, 400),
                'max_depth': randint(5, 25),
                'min_samples_split': randint(2, 15),
                'min_samples_leaf': randint(1, 8)
            },
            'gb': {
                'n_estimators': randint(100, 300),
                'max_depth': randint(4, 12),
                'learning_rate': uniform(0.01, 0.2),
                'min_samples_split': randint(2, 15)
            },
            'mlp': {
                'alpha': uniform(0.0001, 0.01),
                'learning_rate_init': uniform(0.0005, 0.02)
            },
            'xgb': {
                'n_estimators': randint(100, 400),
                'max_depth': randint(4, 12),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            },
            'lgb': {
                'n_estimators': randint(100, 400),
                'max_depth': randint(6, 15),
                'learning_rate': uniform(0.01, 0.2),
                'num_leaves': randint(15, 60)
            }
        }
        return param_distributions.get(model_name, {})

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        method: str = 'random',
        n_iter: int = 20,
        cv: int = 5,
        scoring: str = 'f1_weighted',
        use_time_series_cv: bool = True
    ) -> Tuple[Dict, float]:
        """
        Оптимизация гиперпараметров модели.

        Args:
            X: Признаки
            y: Целевая переменная
            model_name: Название модели для оптимизации
            method: Метод поиска ('grid' или 'random')
            n_iter: Количество итераций для RandomizedSearchCV
            cv: Количество фолдов для кросс-валидации
            scoring: Метрика для оптимизации
            use_time_series_cv: Использовать TimeSeriesSplit

        Returns:
            Лучшие параметры и лучший скор
        """
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена")

        model = self.models[model_name]

        if use_time_series_cv:
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = cv

        if method == 'grid':
            param_grid = self._get_param_grid(model_name)
            search = GridSearchCV(
                model, param_grid,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            param_distributions = self._get_param_distributions(model_name)
            search = RandomizedSearchCV(
                model, param_distributions,
                n_iter=n_iter,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

        search.fit(X, y)

        self.models[model_name] = search.best_estimator_
        self.best_params[model_name] = search.best_params_

        return search.best_params_, search.best_score_

    def tune_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'random',
        n_iter: int = 15
    ) -> Dict:
        """
        Оптимизация гиперпараметров всех моделей.

        Args:
            X: Признаки
            y: Целевая переменная
            method: Метод поиска
            n_iter: Количество итераций для RandomizedSearchCV

        Returns:
            Словарь с лучшими параметрами для каждой модели
        """
        all_best_params = {}

        for model_name in self.models.keys():
            print(f"Оптимизация гиперпараметров для {model_name}...")
            try:
                best_params, best_score = self.tune_hyperparameters(
                    X, y, model_name,
                    method=method,
                    n_iter=n_iter
                )
                all_best_params[model_name] = {
                    'params': best_params,
                    'score': best_score
                }
                print(f"  Лучший скор: {best_score:.4f}")
            except Exception as e:
                print(f"  Ошибка при оптимизации {model_name}: {e}")

        return all_best_params

    def apply_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = None,
        threshold: str = 'median'
    ) -> np.ndarray:
        """
        Применение отбора признаков.

        Args:
            X: Признаки
            y: Целевая переменная
            n_features: Количество признаков для RFE (если None, используется половина)
            threshold: Порог для SelectFromModel

        Returns:
            Отобранные признаки
        """
        if not self.use_feature_selection:
            return X

        if self.feature_selection_method == 'selectfrommodel':
            base_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            self.feature_selector = SelectFromModel(
                base_model, threshold=threshold
            )
            self.feature_selector.fit(X, y)
            X_selected = self.feature_selector.transform(X)
            self.selected_feature_indices = self.feature_selector.get_support(indices=True)

        elif self.feature_selection_method == 'rfe':
            if n_features is None:
                n_features = X.shape[1] // 2
            base_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            self.feature_selector = RFE(
                base_model, n_features_to_select=n_features, step=1
            )
            self.feature_selector.fit(X, y)
            X_selected = self.feature_selector.transform(X)
            self.selected_feature_indices = np.where(self.feature_selector.support_)[0]

        else:
            return X

        print(f"Отобрано {X_selected.shape[1]} признаков из {X.shape[1]}")
        return X_selected

    def calibrate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'isotonic',
        cv: int = 5
    ):
        """
        Калибровка вероятностей моделей.

        Args:
            X: Признаки
            y: Целевая переменная
            method: Метод калибровки ('isotonic' или 'sigmoid')
            cv: Количество фолдов
        """
        for name, model in self.models.items():
            try:
                calibrated = CalibratedClassifierCV(
                    model, method=method, cv=cv
                )
                calibrated.fit(X, y)
                self.calibrated_models[name] = calibrated
                print(f"Калибрация {name}: успешно")
            except Exception as e:
                print(f"Калибрация {name}: ошибка - {e}")

    def build_stacking_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        final_estimator: str = 'logistic',
        cv: int = 5
    ):
        """
        Построение стекинг-модели.

        Args:
            X: Признаки
            y: Целевая переменная
            final_estimator: Финальный классификатор ('logistic', 'rf', 'gb')
            cv: Количество фолдов
        """
        estimators = [(name, model) for name, model in self.models.items()
                      if name != 'mlp']

        if final_estimator == 'logistic':
            final = LogisticRegression(random_state=42, max_iter=1000)
        elif final_estimator == 'rf':
            final = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            final = GradientBoostingClassifier(n_estimators=100, random_state=42)

        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final,
            cv=cv,
            stack_method='predict_proba',
            n_jobs=-1
        )

        print("Обучение стекинг-модели...")
        self.stacking_model.fit(X, y)
        print("Стекинг-модель обучена")

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        flood_threshold: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения.

        Args:
            df: DataFrame с результатами моделирования
            flood_threshold: Порог расхода для определения паводка

        Returns:
            X, y для обучения
        """
        # Создание признаков
        features_df = self.feature_engineer.create_features(df)

        # Создание целевой переменной
        if self.target == FloodPredictionTarget.BINARY:
            if 'discharge_m3s' in df.columns:
                y = (df['discharge_m3s'] > flood_threshold).astype(int)
            elif 'risk_level' in df.columns:
                y = (df['risk_level'].isin(['high', 'critical'])).astype(int)
            else:
                raise ValueError("Нет данных для создания целевой переменной")
        elif self.target == FloodPredictionTarget.RISK_LEVEL:
            if 'risk_level' in df.columns:
                y = self.label_encoder.fit_transform(df['risk_level'])
            else:
                raise ValueError("Нет колонки risk_level")
        else:
            if 'discharge_m3s' in df.columns:
                y = df['discharge_m3s'].values
            else:
                raise ValueError("Нет колонки discharge_m3s")

        # Выбор признаков
        feature_cols = self.feature_engineer.get_feature_names()
        X = features_df[feature_cols].values

        # Масштабирование
        X = self.scaler.fit_transform(X)

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        flood_threshold: float = 100.0,
        test_size: float = 0.2,
        use_time_series_cv: bool = True,
        tune_hyperparameters: bool = False,
        tuning_method: str = 'random',
        tuning_n_iter: int = 15,
        build_stacking: bool = False,
        stacking_final_estimator: str = 'logistic'
    ) -> Dict:
        """
        Обучение модели.

        Args:
            df: DataFrame с данными
            flood_threshold: Порог паводка
            test_size: Доля тестовой выборки
            use_time_series_cv: Использовать временной кросс-валидацию
            tune_hyperparameters: Выполнять оптимизацию гиперпараметров
            tuning_method: Метод оптимизации ('grid' или 'random')
            tuning_n_iter: Количество итераций для RandomizedSearchCV
            build_stacking: Построить стекинг-модель
            stacking_final_estimator: Финальный классификатор для стекинга

        Returns:
            Метрики обучения
        """
        X, y = self.prepare_training_data(df, flood_threshold)

        # Отбор признаков (если включен)
        if self.use_feature_selection:
            X = self.apply_feature_selection(X, y)

        # Разделение данных
        if use_time_series_cv:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        # Оптимизация гиперпараметров (если включена)
        if tune_hyperparameters:
            print("Запуск оптимизации гиперпараметров...")
            self.tune_all_models(X_train, y_train, method=tuning_method, n_iter=tuning_n_iter)

        metrics = {}

        # Определение моделей для обучения
        if self.model_type == 'ensemble' or self.model_type == 'stacking':
            models_to_train = list(self.models.keys())
        else:
            models_to_train = [self.model_type]

        for name in models_to_train:
            if name not in self.models:
                continue

            model = self.models[name]
            model.fit(X_train, y_train)

            # Оценка
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_test)
                y_prob = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
            else:
                y_prob = y_pred

            model_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

            # ROC-AUC для бинарной классификации
            if len(np.unique(y)) == 2:
                try:
                    model_metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                except Exception:
                    pass

            # Кросс-валидация
            if use_time_series_cv:
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1_weighted')
            else:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

            model_metrics['cv_f1_mean'] = cv_scores.mean()
            model_metrics['cv_f1_std'] = cv_scores.std()

            metrics[name] = model_metrics

        # Калибровка вероятностей (если включена)
        if self.use_calibration:
            print("Калибровка моделей...")
            self.calibrate_models(X_train, y_train)

            # Оценка калиброванных моделей
            for name, cal_model in self.calibrated_models.items():
                y_pred_cal = cal_model.predict(X_test)
                prob_cal = cal_model.predict_proba(X_test)
                y_prob_cal = prob_cal[:, 1] if prob_cal.shape[1] > 1 else prob_cal[:, 0]

                metrics[f'{name}_calibrated'] = {
                    'accuracy': accuracy_score(y_test, y_pred_cal),
                    'precision': precision_score(y_test, y_pred_cal, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred_cal, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred_cal, average='weighted', zero_division=0),
                }
                if len(np.unique(y)) == 2:
                    try:
                        metrics[f'{name}_calibrated']['roc_auc'] = roc_auc_score(y_test, y_prob_cal)
                    except Exception:
                        pass

        # Построение стекинг-модели (если включено)
        if build_stacking or self.model_type == 'stacking':
            self.build_stacking_model(X_train, y_train, final_estimator=stacking_final_estimator)

            # Оценка стекинг-модели
            y_pred_stack = self.stacking_model.predict(X_test)
            prob_stack = self.stacking_model.predict_proba(X_test)
            y_prob_stack = prob_stack[:, 1] if prob_stack.shape[1] > 1 else prob_stack[:, 0]

            metrics['stacking'] = {
                'accuracy': accuracy_score(y_test, y_pred_stack),
                'precision': precision_score(y_test, y_pred_stack, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_stack, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_stack, average='weighted', zero_division=0),
            }
            if len(np.unique(y)) == 2:
                try:
                    metrics['stacking']['roc_auc'] = roc_auc_score(y_test, y_prob_stack)
                except Exception:
                    pass

        self.training_metrics = metrics
        self.is_trained = True

        # Вычисление важности признаков
        self._calculate_feature_importance()

        return metrics

    def _calculate_feature_importance(self):
        """Расчет важности признаков."""
        feature_names = self.feature_engineer.get_feature_names()

        # Если использовался отбор признаков, используем только отобранные
        if self.use_feature_selection and self.selected_feature_indices is not None:
            feature_names = [feature_names[i] for i in self.selected_feature_indices]

        importance_df = pd.DataFrame({'feature': feature_names})

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Проверяем соответствие размерностей
                if len(importances) == len(feature_names):
                    importance_df[f'{name}_importance'] = importances

        # Средняя важность по всем моделям
        importance_cols = [col for col in importance_df.columns if 'importance' in col]
        if importance_cols:
            importance_df['mean_importance'] = importance_df[importance_cols].mean(axis=1)
            importance_df = importance_df.sort_values('mean_importance', ascending=False)

        self.feature_importance = importance_df

    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = True,
        use_calibrated: bool = False,
        use_stacking: bool = False
    ) -> pd.DataFrame:
        """
        Прогноз паводков.

        Args:
            df: DataFrame с входными данными
            return_proba: Возвращать вероятности
            use_calibrated: Использовать калиброванные модели
            use_stacking: Использовать стекинг-модель

        Returns:
            DataFrame с прогнозами
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Вызовите train() сначала.")

        # Создание признаков
        features_df = self.feature_engineer.create_features(df)
        feature_cols = self.feature_engineer.get_feature_names()
        X = features_df[feature_cols].values
        X = self.scaler.transform(X)

        # Применение отбора признаков (если был использован при обучении)
        if self.use_feature_selection and self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        predictions = df[['timestamp']].copy() if 'timestamp' in df.columns else pd.DataFrame()

        # Использование стекинг-модели
        if use_stacking and self.stacking_model is not None:
            pred = self.stacking_model.predict(X)
            predictions['flood_predicted'] = pred
            if return_proba:
                prob = self.stacking_model.predict_proba(X)
                predictions['flood_probability'] = prob[:, 1]
                predictions['prediction_confidence'] = np.abs(prob[:, 1] - 0.5) * 2
        else:
            # Выбор моделей для прогноза
            if use_calibrated and self.calibrated_models:
                models_for_predict = self.calibrated_models
            else:
                models_for_predict = self.models

            # Прогнозы от каждой модели
            model_predictions = {}
            model_probabilities = {}

            for name, model in models_for_predict.items():
                if not hasattr(model, 'predict'):
                    continue

                pred = model.predict(X)
                model_predictions[name] = pred

                if return_proba and hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    if prob.shape[1] == 2:
                        model_probabilities[name] = prob[:, 1]
                    else:
                        model_probabilities[name] = prob.max(axis=1)

            # Ансамблевый прогноз
            if self.model_type == 'ensemble' and len(model_predictions) > 1:
                all_preds = np.array(list(model_predictions.values()))
                predictions['flood_predicted'] = (all_preds.mean(axis=0) > 0.5).astype(int)

                if model_probabilities:
                    all_probs = np.array(list(model_probabilities.values()))
                    predictions['flood_probability'] = all_probs.mean(axis=0)
                    predictions['prediction_confidence'] = 1 - all_probs.std(axis=0)
            else:
                main_model = list(model_predictions.keys())[0]
                predictions['flood_predicted'] = model_predictions[main_model]
                if main_model in model_probabilities:
                    predictions['flood_probability'] = model_probabilities[main_model]
                    predictions['prediction_confidence'] = np.abs(model_probabilities[main_model] - 0.5) * 2

        # Определение уровня риска на основе вероятности
        if 'flood_probability' in predictions.columns:
            predictions['risk_level'] = pd.cut(
                predictions['flood_probability'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['low', 'moderate', 'high', 'critical']
            )

        return predictions

    def predict_next_hours(
        self,
        current_data: pd.DataFrame,
        hours_ahead: int = 24,
        scenario: str = 'normal'
    ) -> pd.DataFrame:
        """
        Прогноз на несколько часов вперед.

        Args:
            current_data: Текущие данные
            hours_ahead: Часов вперед
            scenario: Сценарий ('normal', 'wet', 'dry')

        Returns:
            Прогноз на будущее
        """
        # Генерация будущих временных меток
        last_time = pd.to_datetime(current_data['timestamp'].iloc[-1])
        future_times = [last_time + timedelta(hours=i+1) for i in range(hours_ahead)]

        # Базовый прогноз осадков (упрощенный)
        last_precip = current_data['precipitation_mm'].iloc[-24:].mean()
        last_temp = current_data['temperature_c'].iloc[-24:].mean()

        scenario_multipliers = {'normal': 1.0, 'wet': 1.5, 'dry': 0.5}
        multiplier = scenario_multipliers.get(scenario, 1.0)

        # Создание будущих данных
        future_data = pd.DataFrame({
            'timestamp': future_times,
            'precipitation_mm': np.random.exponential(last_precip * multiplier, hours_ahead),
            'temperature_c': last_temp + np.random.normal(0, 2, hours_ahead)
        })

        # Объединение с историей для признаков
        combined = pd.concat([current_data.tail(72), future_data], ignore_index=True)

        # Прогноз
        predictions = self.predict(combined)

        # Возврат только будущих прогнозов
        return predictions.tail(hours_ahead)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Получение важности признаков.

        Args:
            top_n: Количество топ признаков

        Returns:
            DataFrame с важностью признаков
        """
        if self.feature_importance is None:
            return pd.DataFrame()

        return self.feature_importance.head(top_n)

    def get_contributing_factors(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Dict[str, float]:
        """
        Получение факторов, влияющих на конкретный прогноз.

        Args:
            df: DataFrame с данными
            idx: Индекс записи

        Returns:
            Словарь с вкладом признаков
        """
        if self.feature_importance is None:
            return {}

        features_df = self.feature_engineer.create_features(df)
        feature_cols = self.feature_engineer.get_feature_names()
        values = features_df[feature_cols].iloc[idx]

        # Топ факторы
        top_features = self.feature_importance.head(10)['feature'].tolist()

        factors = {}
        for feat in top_features:
            if feat in values.index:
                importance = self.feature_importance[
                    self.feature_importance['feature'] == feat
                ]['mean_importance'].values[0]
                factors[feat] = float(values[feat] * importance)

        return factors

    def save_model(self, path: str):
        """
        Сохранение модели.

        Args:
            path: Путь для сохранения
        """
        model_data = {
            'models': self.models,
            'calibrated_models': self.calibrated_models,
            'stacking_model': self.stacking_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_engineer': self.feature_engineer,
            'feature_selector': self.feature_selector,
            'selected_feature_indices': self.selected_feature_indices,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'best_params': self.best_params,
            'model_type': self.model_type,
            'target': self.target,
            'is_trained': self.is_trained,
            'use_calibration': self.use_calibration,
            'use_feature_selection': self.use_feature_selection,
            'feature_selection_method': self.feature_selection_method
        }
        joblib.dump(model_data, path)

    def load_model(self, path: str):
        """
        Загрузка модели.

        Args:
            path: Путь к файлу модели
        """
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.calibrated_models = model_data.get('calibrated_models', {})
        self.stacking_model = model_data.get('stacking_model', None)
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_selector = model_data.get('feature_selector', None)
        self.selected_feature_indices = model_data.get('selected_feature_indices', None)
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        self.best_params = model_data.get('best_params', {})
        self.model_type = model_data['model_type']
        self.target = model_data['target']
        self.is_trained = model_data['is_trained']
        self.use_calibration = model_data.get('use_calibration', False)
        self.use_feature_selection = model_data.get('use_feature_selection', False)
        self.feature_selection_method = model_data.get('feature_selection_method', 'selectfrommodel')


def generate_training_data(
    region: str = 'chui',
    days: int = 365,
    include_floods: bool = True,
    flood_threshold: float = 100.0
) -> pd.DataFrame:
    """
    Генерация данных для обучения модели.

    Args:
        region: Регион
        days: Количество дней
        include_floods: Включать паводковые события
        flood_threshold: Порог для определения паводка

    Returns:
        DataFrame с данными
    """
    from ..utils.helpers import generate_sample_data
    from .flood_model import FloodModel

    # Генерация данных
    data = generate_sample_data(region=region, days=days, scenario='normal')

    # Добавление экстремальных событий
    if include_floods:
        # Количество паводковых событий пропорционально количеству дней
        n_floods = max(3, min(10, days // 30))
        available_range = max(1, days - 60)
        flood_days = np.random.choice(
            range(10, 10 + available_range),
            size=min(n_floods, available_range),
            replace=False
        )
        for day in flood_days:
            # Интенсивные осадки в течение 24-72 часов
            duration = np.random.randint(24, 72)
            start_idx = day * 24
            end_idx = min(start_idx + duration, len(data))
            # Высокая интенсивность осадков для гарантии паводка
            data.loc[start_idx:end_idx, 'precipitation_mm'] = np.random.uniform(15, 40, end_idx - start_idx + 1)

    # Запуск моделирования для получения расхода и уровня риска
    model = FloodModel(region=region)
    model.initialize_state(data['timestamp'].iloc[0])
    results = model.run_simulation(data, dt_hours=1)

    # Объединение
    data['discharge_m3s'] = results['discharge_m3s'].values
    data['risk_level'] = results['risk_level'].values

    # Проверка и корректировка: если недостаточно паводковых событий,
    # искусственно увеличиваем расход в некоторых периодах
    n_floods_actual = (data['discharge_m3s'] > flood_threshold).sum()
    if include_floods and n_floods_actual < len(data) * 0.05:
        # Минимум 5% данных должны быть паводками
        n_needed = int(len(data) * 0.1) - n_floods_actual
        if n_needed > 0:
            # Выбираем случайные индексы с высокими осадками
            high_precip_idx = data[data['precipitation_mm'] > data['precipitation_mm'].quantile(0.7)].index
            if len(high_precip_idx) > 0:
                boost_idx = np.random.choice(
                    high_precip_idx,
                    size=min(n_needed, len(high_precip_idx)),
                    replace=False
                )
                data.loc[boost_idx, 'discharge_m3s'] = np.random.uniform(
                    flood_threshold * 1.1,
                    flood_threshold * 3,
                    len(boost_idx)
                )

    return data


def train_flood_predictor(
    region: str = 'chui',
    days: int = 730,
    model_type: str = 'ensemble',
    tune_hyperparameters: bool = False,
    use_calibration: bool = False,
    use_feature_selection: bool = False,
    build_stacking: bool = False
) -> Tuple[FloodMLPredictor, Dict]:
    """
    Обучение предиктора паводков.

    Args:
        region: Регион
        days: Дней для обучения
        model_type: Тип модели ('rf', 'xgb', 'lgb', 'gb', 'mlp', 'ensemble', 'stacking')
        tune_hyperparameters: Выполнять оптимизацию гиперпараметров
        use_calibration: Применять калибровку вероятностей
        use_feature_selection: Применять отбор признаков
        build_stacking: Построить стекинг-модель

    Returns:
        Обученная модель и метрики
    """
    print(f"Генерация данных для региона {region} ({days} дней)...")
    data = generate_training_data(region=region, days=days, include_floods=True)

    print("Обучение модели...")
    predictor = FloodMLPredictor(
        model_type=model_type,
        use_calibration=use_calibration,
        use_feature_selection=use_feature_selection
    )
    metrics = predictor.train(
        data,
        flood_threshold=100.0,
        tune_hyperparameters=tune_hyperparameters,
        build_stacking=build_stacking
    )

    print("\nМетрики обучения:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if predictor.best_params:
        print("\nЛучшие гиперпараметры:")
        for model_name, params in predictor.best_params.items():
            print(f"\n{model_name.upper()}: {params}")

    return predictor, metrics


if __name__ == "__main__":
    # Пример использования с базовым ансамблем
    print("=" * 60)
    print("Базовое обучение с ансамблем моделей")
    print("=" * 60)
    predictor, metrics = train_flood_predictor(region='chui', days=365)

    print("\n\nВажность признаков:")
    print(predictor.get_feature_importance(10))

    # Пример использования с расширенными возможностями
    print("\n" + "=" * 60)
    print("Обучение с оптимизацией гиперпараметров и стекингом")
    print("=" * 60)
    predictor_advanced, metrics_advanced = train_flood_predictor(
        region='chui',
        days=365,
        model_type='ensemble',
        tune_hyperparameters=True,
        use_calibration=True,
        use_feature_selection=True,
        build_stacking=True
    )
