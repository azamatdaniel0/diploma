"""
Модель SCS Curve Number для расчета поверхностного стока.

Метод SCS-CN (Soil Conservation Service Curve Number) - один из
наиболее распространенных методов расчета поверхностного стока
с водосборного бассейна.

Основные уравнения:
Q = (P - Ia)² / (P - Ia + S)  при P > Ia
Q = 0                          при P <= Ia

где:
Q - сток (мм)
P - осадки (мм)
Ia - начальные абстракции (мм), Ia = λ * S
S - максимальное удержание (мм), S = 25400/CN - 254
CN - номер кривой стока (0-100)
λ - коэффициент начальных абстракций (обычно 0.2)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..config import SCSParameters, REGIONS


class SoilGroup(Enum):
    """Гидрологические группы почв."""
    A = "A"  # Низкий потенциал стока (песчаные почвы)
    B = "B"  # Умеренно низкий потенциал
    C = "C"  # Умеренно высокий потенциал
    D = "D"  # Высокий потенциал стока (глинистые почвы)


class LandUse(Enum):
    """Типы землепользования."""
    URBAN = "urban"
    RESIDENTIAL = "residential"
    AGRICULTURAL = "agricultural"
    PASTURE = "pasture"
    MEADOW = "meadow"
    FOREST = "forest"
    BARE_SOIL = "bare_soil"
    ROCK = "rock"
    GLACIER = "glacier"
    WATER = "water"


class AntecedentMoistureCondition(Enum):
    """Условия предшествующего увлажнения."""
    DRY = 1      # AMC I - сухие условия
    NORMAL = 2   # AMC II - нормальные условия
    WET = 3      # AMC III - влажные условия


@dataclass
class CNLookupTable:
    """
    Таблица номеров кривых стока для различных условий.

    Значения CN для нормальных условий увлажнения (AMC II).
    """

    # CN для различных типов землепользования и групп почв
    # Формат: {land_use: {soil_group: CN}}
    CN_VALUES = {
        LandUse.URBAN: {SoilGroup.A: 89, SoilGroup.B: 92, SoilGroup.C: 94, SoilGroup.D: 95},
        LandUse.RESIDENTIAL: {SoilGroup.A: 77, SoilGroup.B: 85, SoilGroup.C: 90, SoilGroup.D: 92},
        LandUse.AGRICULTURAL: {SoilGroup.A: 67, SoilGroup.B: 78, SoilGroup.C: 85, SoilGroup.D: 89},
        LandUse.PASTURE: {SoilGroup.A: 49, SoilGroup.B: 69, SoilGroup.C: 79, SoilGroup.D: 84},
        LandUse.MEADOW: {SoilGroup.A: 30, SoilGroup.B: 58, SoilGroup.C: 71, SoilGroup.D: 78},
        LandUse.FOREST: {SoilGroup.A: 36, SoilGroup.B: 60, SoilGroup.C: 73, SoilGroup.D: 79},
        LandUse.BARE_SOIL: {SoilGroup.A: 77, SoilGroup.B: 86, SoilGroup.C: 91, SoilGroup.D: 94},
        LandUse.ROCK: {SoilGroup.A: 96, SoilGroup.B: 96, SoilGroup.C: 96, SoilGroup.D: 96},
        LandUse.GLACIER: {SoilGroup.A: 98, SoilGroup.B: 98, SoilGroup.C: 98, SoilGroup.D: 98},
        LandUse.WATER: {SoilGroup.A: 100, SoilGroup.B: 100, SoilGroup.C: 100, SoilGroup.D: 100},
    }

    @classmethod
    def get_cn(
        cls,
        land_use: LandUse,
        soil_group: SoilGroup,
        amc: AntecedentMoistureCondition = AntecedentMoistureCondition.NORMAL
    ) -> float:
        """
        Получение номера кривой стока.

        Args:
            land_use: Тип землепользования
            soil_group: Группа почвы
            amc: Условия предшествующего увлажнения

        Returns:
            Номер кривой стока
        """
        cn_ii = cls.CN_VALUES[land_use][soil_group]

        # Корректировка для AMC I и AMC III
        if amc == AntecedentMoistureCondition.DRY:
            return cls._cn_to_amc1(cn_ii)
        elif amc == AntecedentMoistureCondition.WET:
            return cls._cn_to_amc3(cn_ii)

        return cn_ii

    @staticmethod
    def _cn_to_amc1(cn_ii: float) -> float:
        """Преобразование CN из AMC II в AMC I."""
        return 4.2 * cn_ii / (10 - 0.058 * cn_ii)

    @staticmethod
    def _cn_to_amc3(cn_ii: float) -> float:
        """Преобразование CN из AMC II в AMC III."""
        return 23 * cn_ii / (10 + 0.13 * cn_ii)


class SCSCurveNumberModel:
    """
    Модель SCS Curve Number для расчета поверхностного стока.
    """

    def __init__(
        self,
        initial_abstraction_ratio: float = 0.2,
        region: str = "chui"
    ):
        """
        Инициализация модели.

        Args:
            initial_abstraction_ratio: Коэффициент начальных абстракций (λ)
            region: Код региона
        """
        self.lambda_coeff = initial_abstraction_ratio
        self.region = region
        self.region_config = REGIONS.get(region)

        # Параметры модели
        self.cn_grid: Optional[np.ndarray] = None
        self.land_use_grid: Optional[np.ndarray] = None
        self.soil_group_grid: Optional[np.ndarray] = None

    def calculate_runoff(
        self,
        precipitation: Union[float, np.ndarray],
        cn: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Расчет поверхностного стока по методу SCS-CN.

        Args:
            precipitation: Осадки (мм)
            cn: Номер кривой стока

        Returns:
            Сток (мм)
        """
        # Максимальное удержание почвой
        S = self._calculate_retention(cn)

        # Начальные абстракции
        Ia = self.lambda_coeff * S

        # Расчет стока
        runoff = np.where(
            precipitation > Ia,
            (precipitation - Ia) ** 2 / (precipitation - Ia + S),
            0
        )

        return runoff

    def _calculate_retention(self, cn: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Расчет максимального удержания почвой.

        Args:
            cn: Номер кривой стока

        Returns:
            Максимальное удержание S (мм)
        """
        # S = 25400/CN - 254 (для CN в системе США)
        # или S = 254 * (100/CN - 1)
        return 25400 / cn - 254

    def create_cn_grid(
        self,
        land_use_grid: np.ndarray,
        soil_group_grid: np.ndarray,
        amc: AntecedentMoistureCondition = AntecedentMoistureCondition.NORMAL
    ) -> np.ndarray:
        """
        Создание сетки номеров кривых стока.

        Args:
            land_use_grid: Сетка типов землепользования (коды 0-9)
            soil_group_grid: Сетка групп почв (коды 0-3)
            amc: Условия предшествующего увлажнения

        Returns:
            Сетка значений CN
        """
        land_use_map = list(LandUse)
        soil_group_map = list(SoilGroup)

        cn_grid = np.zeros_like(land_use_grid, dtype=float)

        for i in range(land_use_grid.shape[0]):
            for j in range(land_use_grid.shape[1]):
                lu_code = int(land_use_grid[i, j])
                sg_code = int(soil_group_grid[i, j])

                if lu_code < len(land_use_map) and sg_code < len(soil_group_map):
                    land_use = land_use_map[lu_code]
                    soil_group = soil_group_map[sg_code]
                    cn_grid[i, j] = CNLookupTable.get_cn(land_use, soil_group, amc)
                else:
                    cn_grid[i, j] = 75  # Значение по умолчанию

        self.cn_grid = cn_grid
        self.land_use_grid = land_use_grid
        self.soil_group_grid = soil_group_grid

        return cn_grid

    def generate_synthetic_land_use(
        self,
        shape: Tuple[int, int],
        elevation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Генерация синтетической карты землепользования.

        Для Кыргызстана характерно:
        - Низины: сельскохозяйственные земли, города
        - Средние высоты: пастбища, леса
        - Высокие горы: скалы, ледники

        Args:
            shape: Размеры сетки
            elevation: Карта высот (опционально)

        Returns:
            Сетка кодов землепользования
        """
        land_use = np.zeros(shape, dtype=int)

        if elevation is None:
            # Генерация простого градиента высот
            elevation = np.linspace(1000, 5000, shape[0])[:, np.newaxis]
            elevation = np.tile(elevation, (1, shape[1]))
            elevation += np.random.randn(*shape) * 200

        # Классификация по высоте
        land_use = np.where(elevation < 1500, 2, land_use)  # Сельхозземли
        land_use = np.where((elevation >= 1500) & (elevation < 2500), 3, land_use)  # Пастбища
        land_use = np.where((elevation >= 2500) & (elevation < 3500), 5, land_use)  # Леса
        land_use = np.where((elevation >= 3500) & (elevation < 4500), 7, land_use)  # Скалы
        land_use = np.where(elevation >= 4500, 8, land_use)  # Ледники

        # Добавление городов (случайно в низинах)
        urban_mask = (elevation < 1200) & (np.random.rand(*shape) < 0.1)
        land_use = np.where(urban_mask, 0, land_use)

        return land_use

    def generate_synthetic_soil_groups(
        self,
        shape: Tuple[int, int],
        elevation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Генерация синтетической карты групп почв.

        Args:
            shape: Размеры сетки
            elevation: Карта высот (опционально)

        Returns:
            Сетка кодов групп почв (0=A, 1=B, 2=C, 3=D)
        """
        soil_groups = np.zeros(shape, dtype=int)

        if elevation is None:
            elevation = np.linspace(1000, 5000, shape[0])[:, np.newaxis]
            elevation = np.tile(elevation, (1, shape[1]))

        # Связь с высотой и случайные вариации
        base = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        soil_groups = (base * 3 + np.random.rand(*shape) * 1.5).astype(int)
        soil_groups = np.clip(soil_groups, 0, 3)

        return soil_groups

    def calculate_composite_cn(
        self,
        land_use_fractions: Dict[LandUse, float],
        soil_group: SoilGroup,
        amc: AntecedentMoistureCondition = AntecedentMoistureCondition.NORMAL
    ) -> float:
        """
        Расчет композитного CN для смешанного водосбора.

        Args:
            land_use_fractions: Доли типов землепользования
            soil_group: Преобладающая группа почв
            amc: Условия предшествующего увлажнения

        Returns:
            Взвешенный средний CN
        """
        total_cn = 0
        total_fraction = 0

        for land_use, fraction in land_use_fractions.items():
            cn = CNLookupTable.get_cn(land_use, soil_group, amc)
            total_cn += cn * fraction
            total_fraction += fraction

        if total_fraction > 0:
            return total_cn / total_fraction
        return 75  # Значение по умолчанию

    def calculate_distributed_runoff(
        self,
        precipitation_grid: np.ndarray,
        cn_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Расчет распределенного стока по территории.

        Args:
            precipitation_grid: Сетка осадков (мм)
            cn_grid: Сетка номеров кривых (опционально)

        Returns:
            Сетка стока (мм)
        """
        if cn_grid is None:
            if self.cn_grid is None:
                raise ValueError("CN grid не задана")
            cn_grid = self.cn_grid

        return self.calculate_runoff(precipitation_grid, cn_grid)

    def sensitivity_analysis(
        self,
        precipitation: float,
        base_cn: float = 75,
        cn_range: Tuple[float, float] = (50, 95)
    ) -> Dict[str, np.ndarray]:
        """
        Анализ чувствительности модели к изменению CN.

        Args:
            precipitation: Осадки (мм)
            base_cn: Базовое значение CN
            cn_range: Диапазон значений CN

        Returns:
            Словарь с результатами анализа
        """
        cn_values = np.linspace(cn_range[0], cn_range[1], 50)
        runoff_values = self.calculate_runoff(precipitation, cn_values)

        # Коэффициент стока
        runoff_coeff = runoff_values / precipitation

        return {
            'cn_values': cn_values,
            'runoff_mm': runoff_values,
            'runoff_coefficient': runoff_coeff,
            'sensitivity': np.gradient(runoff_values, cn_values)
        }

    def get_model_statistics(self) -> Dict:
        """
        Получение статистики модели.

        Returns:
            Словарь со статистикой
        """
        stats = {
            'region': self.region,
            'lambda_coefficient': self.lambda_coeff,
        }

        if self.cn_grid is not None:
            stats.update({
                'cn_mean': float(np.mean(self.cn_grid)),
                'cn_min': float(np.min(self.cn_grid)),
                'cn_max': float(np.max(self.cn_grid)),
                'cn_std': float(np.std(self.cn_grid)),
                'grid_shape': self.cn_grid.shape
            })

        return stats


def example_scs_calculation():
    """
    Пример расчета стока методом SCS-CN.
    """
    model = SCSCurveNumberModel(region="chui")

    # Тестовые данные
    precipitations = [10, 25, 50, 75, 100]  # мм
    cn = 75  # Типичное значение для смешанного водосбора

    print("Расчет стока методом SCS-CN")
    print("=" * 50)
    print(f"CN = {cn}")
    print(f"λ = {model.lambda_coeff}")
    print("-" * 50)
    print(f"{'Осадки (мм)':<15} {'Сток (мм)':<15} {'Коэфф. стока':<15}")
    print("-" * 50)

    for P in precipitations:
        Q = model.calculate_runoff(P, cn)
        coeff = Q / P if P > 0 else 0
        print(f"{P:<15} {Q:<15.2f} {coeff:<15.3f}")

    return model


if __name__ == "__main__":
    example_scs_calculation()
