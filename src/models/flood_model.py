"""
Комплексная модель паводков для регионов Кыргызстана.

Объединяет:
- Расчет поверхностного стока (SCS-CN)
- Таяние снега и ледников
- Маршрутизацию стока
- Оценку опасности затопления
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..config import (
    REGIONS, FloodThresholds, HydrologicalParams,
    SEASONAL_PARAMS, RegionConfig
)
from .scs_cn import SCSCurveNumberModel, AntecedentMoistureCondition
from .runoff import RunoffCalculator, WatershedParameters


class FloodRiskLevel(Enum):
    """Уровни риска паводка."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FloodEvent:
    """Описание паводкового события."""
    start_time: datetime
    peak_time: datetime
    end_time: datetime
    duration_hours: float
    peak_discharge_m3s: float
    total_volume_m3: float
    max_precipitation_mm: float
    total_precipitation_mm: float
    risk_level: FloodRiskLevel
    affected_area_km2: float
    region: str


@dataclass
class ModelState:
    """Состояние модели в момент времени."""
    timestamp: datetime
    soil_moisture: float  # 0-1
    snow_water_equivalent: float  # мм
    groundwater_level: float  # м
    antecedent_precip_index: float  # мм
    current_discharge: float  # м³/с


@dataclass
class FloodModelParameters:
    """Параметры модели паводков."""
    # Параметры водосбора
    watershed_area_km2: float
    main_channel_length_km: float
    average_slope: float
    mean_elevation_m: float

    # Параметры модели SCS
    curve_number: float = 75.0
    initial_abstraction_ratio: float = 0.2

    # Параметры таяния снега
    degree_day_factor: float = 4.5  # мм/°C/день
    snowmelt_temp_threshold: float = 0.0  # °C

    # Параметры инфильтрации
    max_infiltration_rate: float = 10.0  # мм/час
    soil_storage_capacity: float = 100.0  # мм

    # Параметры маршрутизации
    muskingum_K: float = 6.0  # часы
    muskingum_X: float = 0.2

    # Пороговые значения
    flood_discharge_threshold: float = 100.0  # м³/с


class FloodModel:
    """
    Комплексная модель паводков.

    Реализует концептуальную гидрологическую модель для
    прогнозирования паводков в горных районах Кыргызстана.
    """

    def __init__(
        self,
        region: str = "chui",
        params: Optional[FloodModelParameters] = None
    ):
        """
        Инициализация модели.

        Args:
            region: Код региона
            params: Параметры модели
        """
        if region not in REGIONS:
            raise ValueError(f"Неизвестный регион: {region}")

        self.region = region
        self.region_config = REGIONS[region]

        # Параметры по умолчанию
        if params is None:
            params = FloodModelParameters(
                watershed_area_km2=self.region_config.area_km2 / 10,  # Типичный подбассейн
                main_channel_length_km=50,
                average_slope=0.03,
                mean_elevation_m=self.region_config.avg_elevation
            )
        self.params = params

        # Инициализация компонентов
        self.scs_model = SCSCurveNumberModel(
            initial_abstraction_ratio=params.initial_abstraction_ratio,
            region=region
        )
        self.runoff_calc = RunoffCalculator(region=region)
        self.thresholds = FloodThresholds()

        # Состояние модели
        self.state: Optional[ModelState] = None
        self.flood_events: List[FloodEvent] = []

    def initialize_state(
        self,
        timestamp: datetime,
        soil_moisture: float = 0.5,
        snow_water_equivalent: float = 0,
        groundwater_level: float = 2.0
    ):
        """
        Инициализация начального состояния модели.

        Args:
            timestamp: Начальное время
            soil_moisture: Влажность почвы (0-1)
            snow_water_equivalent: Снегозапас (мм)
            groundwater_level: Уровень грунтовых вод (м)
        """
        self.state = ModelState(
            timestamp=timestamp,
            soil_moisture=soil_moisture,
            snow_water_equivalent=snow_water_equivalent,
            groundwater_level=groundwater_level,
            antecedent_precip_index=0,
            current_discharge=self._estimate_base_flow()
        )

    def _estimate_base_flow(self) -> float:
        """
        Оценка базового стока.

        Returns:
            Базовый сток (м³/с)
        """
        # Упрощенная оценка на основе площади водосбора
        specific_discharge = 5  # л/с/км²
        return self.params.watershed_area_km2 * specific_discharge / 1000

    def run_timestep(
        self,
        precipitation_mm: float,
        temperature_c: float,
        dt_hours: float = 1.0
    ) -> Dict:
        """
        Расчет одного временного шага модели.

        Args:
            precipitation_mm: Осадки за шаг (мм)
            temperature_c: Температура воздуха (°C)
            dt_hours: Длительность шага (часы)

        Returns:
            Результаты расчета
        """
        if self.state is None:
            raise ValueError("Модель не инициализирована. Вызовите initialize_state()")

        results = {}

        # 1. Разделение осадков на дождь и снег
        if temperature_c < 0:
            snowfall_mm = precipitation_mm
            rainfall_mm = 0
        elif temperature_c > 2:
            snowfall_mm = 0
            rainfall_mm = precipitation_mm
        else:
            # Переходная зона
            snow_fraction = (2 - temperature_c) / 2
            snowfall_mm = precipitation_mm * snow_fraction
            rainfall_mm = precipitation_mm * (1 - snow_fraction)

        # 2. Накопление снега
        self.state.snow_water_equivalent += snowfall_mm

        # 3. Таяние снега
        if temperature_c > self.params.snowmelt_temp_threshold:
            potential_melt = (
                self.params.degree_day_factor *
                (temperature_c - self.params.snowmelt_temp_threshold) *
                dt_hours / 24
            )
            actual_melt = min(potential_melt, self.state.snow_water_equivalent)
            self.state.snow_water_equivalent -= actual_melt
            results['snowmelt_mm'] = actual_melt
        else:
            actual_melt = 0
            results['snowmelt_mm'] = 0

        # 4. Общий вход воды
        water_input = rainfall_mm + actual_melt
        results['water_input_mm'] = water_input

        # 5. Определение условий увлажнения
        amc = self._determine_amc()

        # 6. Расчет стока методом SCS-CN
        cn = self._adjust_cn_for_conditions(amc, temperature_c)
        if water_input > 0:
            surface_runoff = self.scs_model.calculate_runoff(water_input, cn)
        else:
            surface_runoff = 0

        results['surface_runoff_mm'] = surface_runoff
        results['effective_cn'] = cn

        # 7. Инфильтрация
        infiltration = water_input - surface_runoff
        infiltration = min(infiltration, self.params.max_infiltration_rate * dt_hours)
        results['infiltration_mm'] = infiltration

        # 8. Обновление влажности почвы
        soil_capacity = self.params.soil_storage_capacity
        self.state.soil_moisture += infiltration / soil_capacity
        self.state.soil_moisture = min(1.0, self.state.soil_moisture)

        # Испарение/дренаж
        if temperature_c > 0:
            evap_rate = 0.1 * temperature_c * dt_hours / 24  # Упрощенная модель
            self.state.soil_moisture -= evap_rate / soil_capacity
            self.state.soil_moisture = max(0, self.state.soil_moisture)

        # 9. Расчет расхода
        watershed = WatershedParameters(
            area_km2=self.params.watershed_area_km2,
            main_channel_length_km=self.params.main_channel_length_km,
            average_slope=self.params.average_slope,
            cn=cn
        )

        if surface_runoff > 0:
            peak_q = self.runoff_calc.calculate_peak_discharge(
                watershed,
                water_input,
                surface_runoff
            )
        else:
            peak_q = 0

        # Добавление базового стока
        base_flow = self._estimate_base_flow()
        total_discharge = peak_q + base_flow

        self.state.current_discharge = total_discharge
        results['discharge_m3s'] = total_discharge
        results['base_flow_m3s'] = base_flow

        # 10. Обновление индекса предшествующих осадков
        decay = 0.85
        self.state.antecedent_precip_index = (
            precipitation_mm + decay * self.state.antecedent_precip_index
        )

        # 11. Оценка уровня риска
        risk_level = self._assess_flood_risk(
            precipitation_mm,
            total_discharge,
            self.state.soil_moisture
        )
        results['risk_level'] = risk_level.value

        # 12. Обновление времени
        self.state.timestamp += timedelta(hours=dt_hours)

        return results

    def _determine_amc(self) -> AntecedentMoistureCondition:
        """
        Определение условий предшествующего увлажнения.

        Returns:
            Класс AMC
        """
        api = self.state.antecedent_precip_index

        # Пороги для периода роста растительности
        if api < 35:
            return AntecedentMoistureCondition.DRY
        elif api < 53:
            return AntecedentMoistureCondition.NORMAL
        else:
            return AntecedentMoistureCondition.WET

    def _adjust_cn_for_conditions(
        self,
        amc: AntecedentMoistureCondition,
        temperature: float
    ) -> float:
        """
        Корректировка CN с учетом условий.

        Args:
            amc: Условия увлажнения
            temperature: Температура

        Returns:
            Скорректированный CN
        """
        base_cn = self.params.curve_number

        # Корректировка для AMC
        if amc == AntecedentMoistureCondition.DRY:
            cn = 4.2 * base_cn / (10 - 0.058 * base_cn)
        elif amc == AntecedentMoistureCondition.WET:
            cn = 23 * base_cn / (10 + 0.13 * base_cn)
        else:
            cn = base_cn

        # Корректировка для мерзлой почвы
        if temperature < -5:
            cn = min(cn * 1.3, 98)  # Мерзлая почва - высокий CN

        return cn

    def _assess_flood_risk(
        self,
        precipitation: float,
        discharge: float,
        soil_moisture: float
    ) -> FloodRiskLevel:
        """
        Оценка уровня риска паводка.

        Args:
            precipitation: Осадки (мм)
            discharge: Расход (м³/с)
            soil_moisture: Влажность почвы

        Returns:
            Уровень риска
        """
        # Множитель для насыщенной почвы
        moisture_factor = 1 + soil_moisture

        # Критерии по осадкам
        if precipitation >= self.thresholds.PRECIPITATION_EXTREME * moisture_factor:
            precip_risk = 4
        elif precipitation >= self.thresholds.PRECIPITATION_HIGH * moisture_factor:
            precip_risk = 3
        elif precipitation >= self.thresholds.PRECIPITATION_MODERATE * moisture_factor:
            precip_risk = 2
        elif precipitation >= self.thresholds.PRECIPITATION_LOW:
            precip_risk = 1
        else:
            precip_risk = 0

        # Критерии по расходу
        threshold = self.params.flood_discharge_threshold
        if discharge >= threshold * 3:
            discharge_risk = 4
        elif discharge >= threshold * 2:
            discharge_risk = 3
        elif discharge >= threshold * 1.5:
            discharge_risk = 2
        elif discharge >= threshold:
            discharge_risk = 1
        else:
            discharge_risk = 0

        # Комбинированная оценка
        max_risk = max(precip_risk, discharge_risk)

        if max_risk >= 4:
            return FloodRiskLevel.CRITICAL
        elif max_risk >= 3:
            return FloodRiskLevel.HIGH
        elif max_risk >= 2:
            return FloodRiskLevel.MODERATE
        else:
            return FloodRiskLevel.LOW

    def run_simulation(
        self,
        precipitation_series: pd.DataFrame,
        dt_hours: float = 1.0
    ) -> pd.DataFrame:
        """
        Запуск моделирования для временного ряда.

        Args:
            precipitation_series: DataFrame с колонками 'timestamp', 'precipitation_mm', 'temperature_c'
            dt_hours: Шаг времени

        Returns:
            DataFrame с результатами
        """
        results = []

        for _, row in precipitation_series.iterrows():
            step_result = self.run_timestep(
                precipitation_mm=row['precipitation_mm'],
                temperature_c=row.get('temperature_c', 10),
                dt_hours=dt_hours
            )
            step_result['timestamp'] = row['timestamp']
            step_result['precipitation_mm'] = row['precipitation_mm']
            step_result['temperature_c'] = row.get('temperature_c', 10)
            results.append(step_result)

        return pd.DataFrame(results)

    def detect_flood_events(
        self,
        simulation_results: pd.DataFrame,
        min_duration_hours: float = 6
    ) -> List[FloodEvent]:
        """
        Выявление паводковых событий в результатах моделирования.

        Args:
            simulation_results: Результаты моделирования
            min_duration_hours: Минимальная продолжительность события

        Returns:
            Список паводковых событий
        """
        # Проверка входных данных
        if simulation_results is None or len(simulation_results) == 0:
            return []

        if 'discharge_m3s' not in simulation_results.columns:
            raise ValueError("Результаты моделирования должны содержать колонку 'discharge_m3s'")

        threshold = self.params.flood_discharge_threshold
        events = []

        # Поиск периодов превышения порога
        above_threshold = simulation_results['discharge_m3s'] > threshold

        if not above_threshold.any():
            return events

        # Группировка последовательных превышений
        groups = (above_threshold != above_threshold.shift()).cumsum()
        flood_periods = simulation_results[above_threshold].groupby(groups[above_threshold])

        for _, period in flood_periods:
            duration = len(period)
            if duration < min_duration_hours:
                continue

            peak_idx = period['discharge_m3s'].idxmax()
            peak_row = simulation_results.loc[peak_idx]

            event = FloodEvent(
                start_time=period['timestamp'].iloc[0],
                peak_time=peak_row['timestamp'],
                end_time=period['timestamp'].iloc[-1],
                duration_hours=duration,
                peak_discharge_m3s=peak_row['discharge_m3s'],
                total_volume_m3=period['discharge_m3s'].sum() * 3600,  # м³
                max_precipitation_mm=period['precipitation_mm'].max(),
                total_precipitation_mm=period['precipitation_mm'].sum(),
                risk_level=FloodRiskLevel(peak_row['risk_level']),
                affected_area_km2=self.params.watershed_area_km2 * 0.1,  # Оценка
                region=self.region
            )
            events.append(event)

        self.flood_events = events
        return events

    def calculate_flood_statistics(self) -> Dict:
        """
        Расчет статистики по паводкам.

        Returns:
            Словарь со статистикой
        """
        if not self.flood_events:
            return {
                'events': 0,
                'peak_discharge': {
                    'mean': 0.0,
                    'max': 0.0,
                    'std': 0.0
                },
                'duration_hours': {
                    'mean': 0.0,
                    'max': 0.0,
                    'total': 0.0
                },
                'volume_m3': {
                    'mean': 0.0,
                    'max': 0.0,
                    'total': 0.0
                },
                'risk_distribution': {
                    level.value: 0 for level in FloodRiskLevel
                }
            }

        peak_discharges = [e.peak_discharge_m3s for e in self.flood_events]
        durations = [e.duration_hours for e in self.flood_events]
        volumes = [e.total_volume_m3 for e in self.flood_events]

        return {
            'events': len(self.flood_events),
            'peak_discharge': {
                'mean': np.mean(peak_discharges),
                'max': np.max(peak_discharges),
                'std': np.std(peak_discharges)
            },
            'duration_hours': {
                'mean': np.mean(durations),
                'max': np.max(durations),
                'total': np.sum(durations)
            },
            'volume_m3': {
                'mean': np.mean(volumes),
                'max': np.max(volumes),
                'total': np.sum(volumes)
            },
            'risk_distribution': {
                level.value: sum(1 for e in self.flood_events if e.risk_level == level)
                for level in FloodRiskLevel
            }
        }

    def scenario_analysis(
        self,
        base_precipitation: pd.DataFrame,
        scenarios: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Сценарный анализ паводков.

        Args:
            base_precipitation: Базовый ряд осадков
            scenarios: Словарь {название: множитель осадков}

        Returns:
            Результаты для каждого сценария
        """
        results = {}

        for scenario_name, multiplier in scenarios.items():
            # Модификация осадков
            modified = base_precipitation.copy()
            modified['precipitation_mm'] *= multiplier

            # Переинициализация состояния
            self.initialize_state(modified['timestamp'].iloc[0])

            # Моделирование
            sim_results = self.run_simulation(modified)
            self.detect_flood_events(sim_results)

            results[scenario_name] = {
                'simulation': sim_results,
                'events': self.flood_events.copy(),
                'statistics': self.calculate_flood_statistics()
            }

        return results

    def get_model_info(self) -> Dict:
        """
        Получение информации о модели.

        Returns:
            Словарь с параметрами модели
        """
        return {
            'region': self.region,
            'region_name': self.region_config.name,
            'watershed_area_km2': self.params.watershed_area_km2,
            'mean_elevation_m': self.params.mean_elevation_m,
            'curve_number': self.params.curve_number,
            'flood_threshold_m3s': self.params.flood_discharge_threshold,
            'state': {
                'timestamp': str(self.state.timestamp) if self.state else None,
                'soil_moisture': self.state.soil_moisture if self.state else None,
                'snow_water_equivalent': self.state.snow_water_equivalent if self.state else None,
                'current_discharge': self.state.current_discharge if self.state else None
            }
        }


def create_sample_simulation():
    """
    Создание примера моделирования.
    """
    from ..data.precipitation import PrecipitationDataLoader

    # Загрузка данных
    loader = PrecipitationDataLoader(region="chui")
    precip_data = loader.load_synthetic(
        start_date=datetime(2024, 4, 1),
        end_date=datetime(2024, 4, 30),
        include_extreme_events=True
    )

    # Агрегация по дням
    daily_data = precip_data.groupby(precip_data['timestamp'].dt.date).agg({
        'precipitation_mm': 'mean',
        'temperature_c': 'mean'
    }).reset_index()
    daily_data.columns = ['timestamp', 'precipitation_mm', 'temperature_c']
    daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])

    # Создание и запуск модели
    model = FloodModel(region="chui")
    model.initialize_state(daily_data['timestamp'].iloc[0])

    results = model.run_simulation(daily_data, dt_hours=24)
    events = model.detect_flood_events(results)

    print(f"Обнаружено паводковых событий: {len(events)}")
    for i, event in enumerate(events, 1):
        print(f"\nСобытие {i}:")
        print(f"  Начало: {event.start_time}")
        print(f"  Пик: {event.peak_time}")
        print(f"  Пиковый расход: {event.peak_discharge_m3s:.1f} м³/с")
        print(f"  Уровень риска: {event.risk_level.value}")

    return model, results


if __name__ == "__main__":
    create_sample_simulation()
