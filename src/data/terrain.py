"""
Модуль анализа рельефа местности.

Включает:
- Анализ цифровой модели рельефа (DEM)
- Расчет уклонов и направлений стока
- Определение водосборных бассейнов
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from scipy import ndimage

from ..config import REGIONS, RegionConfig


@dataclass
class TerrainCell:
    """Ячейка рельефа."""
    elevation: float  # Высота в метрах
    slope: float  # Уклон в градусах
    aspect: float  # Экспозиция склона (направление)
    flow_direction: int  # Направление стока (1-8)
    flow_accumulation: float  # Аккумуляция стока
    is_channel: bool  # Является ли ячейка частью русла


class TerrainAnalyzer:
    """
    Анализатор рельефа местности для гидрологического моделирования.
    """

    # Направления стока (код D8)
    FLOW_DIRECTIONS = {
        1: (0, 1),    # Восток
        2: (1, 1),    # Юго-восток
        4: (1, 0),    # Юг
        8: (1, -1),   # Юго-запад
        16: (0, -1),  # Запад
        32: (-1, -1), # Северо-запад
        64: (-1, 0),  # Север
        128: (-1, 1)  # Северо-восток
    }

    def __init__(self, region: str = "chui"):
        """
        Инициализация анализатора.

        Args:
            region: Код региона
        """
        if region not in REGIONS:
            raise ValueError(f"Неизвестный регион: {region}")

        self.region = region
        self.region_config = REGIONS[region]
        self.dem: Optional[np.ndarray] = None
        self.cell_size: float = 100.0  # метры
        self.slope: Optional[np.ndarray] = None
        self.aspect: Optional[np.ndarray] = None
        self.flow_direction: Optional[np.ndarray] = None
        self.flow_accumulation: Optional[np.ndarray] = None

    def generate_synthetic_dem(
        self,
        resolution: int = 100,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Генерация синтетической цифровой модели рельефа.

        Создает реалистичный рельеф с учетом:
        - Горных хребтов
        - Долин рек
        - Случайных вариаций

        Args:
            resolution: Разрешение сетки (число ячеек по каждой оси)
            noise_level: Уровень случайного шума

        Returns:
            2D массив высот
        """
        bounds = self.region_config.bounds
        avg_elevation = self.region_config.avg_elevation

        # Создание базового рельефа
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Основной горный хребет (диагональный)
        mountain_ridge = np.sin(np.pi * (X + Y)) * 1500

        # Вторичные хребты
        secondary_ridges = (
            np.sin(4 * np.pi * X) * np.cos(2 * np.pi * Y) * 500 +
            np.sin(2 * np.pi * X) * np.sin(6 * np.pi * Y) * 300
        )

        # Речные долины (понижения)
        valleys = -np.abs(np.sin(3 * np.pi * Y) * np.cos(2 * np.pi * X)) * 400

        # Случайный шум (фрактальный)
        noise = self._generate_fractal_noise(resolution, octaves=4) * noise_level * avg_elevation

        # Комбинирование компонентов
        dem = avg_elevation + mountain_ridge + secondary_ridges + valleys + noise

        # Обеспечение положительных высот
        dem = np.maximum(dem, 500)

        self.dem = dem
        self.cell_size = (bounds[2] - bounds[0]) * 111000 / resolution  # примерно в метрах

        return dem

    def _generate_fractal_noise(
        self,
        size: int,
        octaves: int = 4,
        persistence: float = 0.5
    ) -> np.ndarray:
        """
        Генерация фрактального шума для реалистичного рельефа.

        Args:
            size: Размер сетки
            octaves: Число октав
            persistence: Коэффициент затухания

        Returns:
            2D массив шума
        """
        noise = np.zeros((size, size))

        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave

            # Случайный шум на данной частоте
            octave_noise = np.random.randn(size // freq + 1, size // freq + 1)

            # Интерполяция до полного размера
            from scipy.ndimage import zoom
            octave_noise = zoom(octave_noise, freq, order=1)[:size, :size]

            noise += octave_noise * amp

        return noise / octaves

    def calculate_slope(self) -> np.ndarray:
        """
        Расчет уклонов поверхности.

        Returns:
            2D массив уклонов в градусах
        """
        if self.dem is None:
            raise ValueError("DEM не загружена")

        # Градиенты по осям
        dy, dx = np.gradient(self.dem, self.cell_size)

        # Уклон в градусах
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        self.slope = np.degrees(slope_rad)

        return self.slope

    def calculate_aspect(self) -> np.ndarray:
        """
        Расчет экспозиции склонов.

        Returns:
            2D массив направлений в градусах (0-360, 0=север)
        """
        if self.dem is None:
            raise ValueError("DEM не загружена")

        dy, dx = np.gradient(self.dem, self.cell_size)

        # Экспозиция в градусах
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        self.aspect = aspect
        return self.aspect

    def calculate_flow_direction(self) -> np.ndarray:
        """
        Расчет направлений стока методом D8.

        Returns:
            2D массив кодов направлений
        """
        if self.dem is None:
            raise ValueError("DEM не загружена")

        rows, cols = self.dem.shape
        flow_dir = np.zeros_like(self.dem, dtype=np.int32)

        # Смещения для 8 соседей
        offsets = [
            (0, 1, 1),    # E
            (1, 1, 2),    # SE
            (1, 0, 4),    # S
            (1, -1, 8),   # SW
            (0, -1, 16),  # W
            (-1, -1, 32), # NW
            (-1, 0, 64),  # N
            (-1, 1, 128)  # NE
        ]

        # Расстояния до соседей
        distances = [1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                max_drop = 0
                direction = 0

                for (di, dj, code), dist in zip(offsets, distances):
                    ni, nj = i + di, j + dj
                    drop = (self.dem[i, j] - self.dem[ni, nj]) / (dist * self.cell_size)

                    if drop > max_drop:
                        max_drop = drop
                        direction = code

                flow_dir[i, j] = direction if direction > 0 else 0

        self.flow_direction = flow_dir
        return flow_dir

    def calculate_flow_accumulation(self) -> np.ndarray:
        """
        Расчет аккумуляции стока.

        Returns:
            2D массив значений аккумуляции
        """
        if self.flow_direction is None:
            self.calculate_flow_direction()

        rows, cols = self.dem.shape
        accumulation = np.ones_like(self.dem)

        # Сортировка ячеек по высоте (от высоких к низким)
        flat_indices = np.argsort(self.dem.ravel())[::-1]

        for idx in flat_indices:
            i, j = divmod(idx, cols)

            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                continue

            direction = self.flow_direction[i, j]
            if direction == 0:
                continue

            # Находим соседа по направлению стока
            for di, dj, code in [
                (0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
                (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128)
            ]:
                if direction == code:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        accumulation[ni, nj] += accumulation[i, j]
                    break

        self.flow_accumulation = accumulation
        return accumulation

    def identify_stream_network(
        self,
        threshold: float = 100
    ) -> np.ndarray:
        """
        Определение речной сети.

        Args:
            threshold: Пороговое значение аккумуляции

        Returns:
            Бинарная маска речной сети
        """
        if self.flow_accumulation is None:
            self.calculate_flow_accumulation()

        return self.flow_accumulation > threshold

    def delineate_watershed(
        self,
        outlet_row: int,
        outlet_col: int
    ) -> np.ndarray:
        """
        Определение водосборного бассейна.

        Args:
            outlet_row: Строка точки выхода
            outlet_col: Столбец точки выхода

        Returns:
            Бинарная маска водосбора
        """
        if self.flow_direction is None:
            self.calculate_flow_direction()

        rows, cols = self.dem.shape
        watershed = np.zeros_like(self.dem, dtype=bool)
        watershed[outlet_row, outlet_col] = True

        # Обратная трассировка
        changed = True
        while changed:
            changed = False
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if watershed[i, j]:
                        continue

                    direction = self.flow_direction[i, j]
                    if direction == 0:
                        continue

                    # Проверяем, ведет ли сток в водосбор
                    for di, dj, code in [
                        (0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
                        (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128)
                    ]:
                        if direction == code:
                            ni, nj = i + di, j + dj
                            if watershed[ni, nj]:
                                watershed[i, j] = True
                                changed = True
                            break

        return watershed

    def get_terrain_statistics(self) -> Dict:
        """
        Получение статистики по рельефу.

        Returns:
            Словарь со статистикой
        """
        if self.dem is None:
            raise ValueError("DEM не загружена")

        stats = {
            'min_elevation': float(np.min(self.dem)),
            'max_elevation': float(np.max(self.dem)),
            'mean_elevation': float(np.mean(self.dem)),
            'std_elevation': float(np.std(self.dem)),
            'relief': float(np.max(self.dem) - np.min(self.dem))
        }

        if self.slope is not None:
            stats['mean_slope'] = float(np.mean(self.slope))
            stats['max_slope'] = float(np.max(self.slope))

        if self.flow_accumulation is not None:
            stream_network = self.identify_stream_network()
            stats['stream_density'] = float(np.sum(stream_network)) / stream_network.size

        return stats

    def calculate_twi(self) -> np.ndarray:
        """
        Расчет топографического индекса влажности (TWI).

        TWI = ln(a / tan(β))
        где a - площадь водосбора, β - уклон

        Returns:
            2D массив значений TWI
        """
        if self.slope is None:
            self.calculate_slope()
        if self.flow_accumulation is None:
            self.calculate_flow_accumulation()

        # Площадь водосбора на единицу ширины
        specific_catchment = self.flow_accumulation * self.cell_size

        # Уклон в радианах (минимум для избежания деления на 0)
        slope_rad = np.radians(np.maximum(self.slope, 0.01))

        # TWI
        twi = np.log(specific_catchment / np.tan(slope_rad))

        return twi


def create_sample_terrain(region: str = "chui") -> TerrainAnalyzer:
    """
    Создание примера анализатора рельефа с синтетическими данными.

    Args:
        region: Код региона

    Returns:
        Настроенный TerrainAnalyzer
    """
    analyzer = TerrainAnalyzer(region=region)
    analyzer.generate_synthetic_dem(resolution=100)
    analyzer.calculate_slope()
    analyzer.calculate_aspect()
    analyzer.calculate_flow_direction()
    analyzer.calculate_flow_accumulation()

    return analyzer
