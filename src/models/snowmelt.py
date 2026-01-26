"""
Модуль моделирования таяния снега и ледников.

Для горных районов Кыргызстана таяние снега и ледников
является одним из основных источников паводков.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..config import REGIONS, HydrologicalParams


@dataclass
class SnowpackState:
    """Состояние снежного покрова."""
    snow_water_equivalent: float  # мм
    snow_depth: float  # см
    density: float  # кг/м³
    albedo: float  # 0-1
    temperature: float  # °C (температура снега)
    liquid_water_content: float  # мм
    is_melting: bool


@dataclass
class GlacierState:
    """Состояние ледника."""
    ice_thickness: float  # м
    area_km2: float
    equilibrium_line_altitude: float  # м
    mass_balance: float  # м.в.э./год
    melt_rate: float  # мм/день


class SnowmeltModel:
    """
    Модель таяния снега.

    Реализует методы:
    - Температурный индекс (degree-day)
    - Энергетический баланс (упрощенный)
    """

    # Типичные значения для Кыргызстана
    DEFAULT_DEGREE_DAY_FACTOR = 4.5  # мм/°C/день для снега
    GLACIER_DEGREE_DAY_FACTOR = 7.5  # мм/°C/день для льда
    NEW_SNOW_DENSITY = 100  # кг/м³
    OLD_SNOW_DENSITY = 350  # кг/м³
    MELTING_POINT = 0.0  # °C

    def __init__(self, region: str = "chui"):
        """
        Инициализация модели.

        Args:
            region: Код региона
        """
        self.region = region
        self.region_config = REGIONS.get(region)
        self.params = HydrologicalParams()
        self.snowpack: Optional[SnowpackState] = None
        self.glacier: Optional[GlacierState] = None

    def initialize_snowpack(
        self,
        snow_water_equivalent: float = 0,
        density: float = None
    ):
        """
        Инициализация снежного покрова.

        Args:
            snow_water_equivalent: Запас воды в снеге (мм)
            density: Плотность снега (кг/м³)
        """
        if density is None:
            density = self.NEW_SNOW_DENSITY if snow_water_equivalent < 50 else self.OLD_SNOW_DENSITY

        snow_depth = snow_water_equivalent / density * 1000  # см

        self.snowpack = SnowpackState(
            snow_water_equivalent=snow_water_equivalent,
            snow_depth=snow_depth,
            density=density,
            albedo=0.8 if snow_water_equivalent > 0 else 0.2,
            temperature=min(0, -5),  # Начальная температура снега
            liquid_water_content=0,
            is_melting=False
        )

    def initialize_glacier(
        self,
        area_km2: float,
        ice_thickness: float = 50,
        ela: float = 4000
    ):
        """
        Инициализация ледника.

        Args:
            area_km2: Площадь ледника
            ice_thickness: Толщина льда (м)
            ela: Высота границы питания (м)
        """
        self.glacier = GlacierState(
            ice_thickness=ice_thickness,
            area_km2=area_km2,
            equilibrium_line_altitude=ela,
            mass_balance=0,
            melt_rate=0
        )

    def degree_day_melt(
        self,
        temperature: float,
        degree_day_factor: float = None,
        dt_hours: float = 24
    ) -> float:
        """
        Расчет таяния методом температурного индекса.

        M = DDF * (T - T_base) * dt

        Args:
            temperature: Температура воздуха (°C)
            degree_day_factor: Коэффициент таяния (мм/°C/день)
            dt_hours: Временной шаг (часы)

        Returns:
            Количество талой воды (мм)
        """
        if degree_day_factor is None:
            degree_day_factor = self.DEFAULT_DEGREE_DAY_FACTOR

        if temperature <= self.MELTING_POINT:
            return 0

        # Расчет таяния
        melt = degree_day_factor * (temperature - self.MELTING_POINT) * (dt_hours / 24)

        return max(0, melt)

    def energy_balance_melt(
        self,
        temperature: float,
        solar_radiation: float,
        wind_speed: float,
        humidity: float,
        dt_hours: float = 24
    ) -> float:
        """
        Расчет таяния методом энергетического баланса (упрощенный).

        Q_melt = Q_net_rad + Q_sensible + Q_latent + Q_ground + Q_rain

        Args:
            temperature: Температура воздуха (°C)
            solar_radiation: Солнечная радиация (Вт/м²)
            wind_speed: Скорость ветра (м/с)
            humidity: Относительная влажность (%)
            dt_hours: Временной шаг (часы)

        Returns:
            Количество талой воды (мм)
        """
        if self.snowpack is None or self.snowpack.snow_water_equivalent <= 0:
            return 0

        if temperature < self.MELTING_POINT:
            return 0

        # Коэффициенты
        LATENT_HEAT_FUSION = 334000  # Дж/кг
        STEFAN_BOLTZMANN = 5.67e-8  # Вт/(м²·К⁴)

        # Чистая радиация
        albedo = self.snowpack.albedo
        net_shortwave = solar_radiation * (1 - albedo)
        net_longwave = -STEFAN_BOLTZMANN * ((273.15 + temperature) ** 4 - (273.15) ** 4) * 0.5
        Q_net_rad = net_shortwave + net_longwave

        # Турбулентные потоки (упрощенно)
        Q_sensible = 1.5 * wind_speed * temperature * 10  # Приближение
        Q_latent = 0.5 * wind_speed * (humidity / 100 - 0.5) * 100  # Приближение

        # Тепловой поток от почвы
        Q_ground = 2  # Вт/м² (небольшое постоянное значение)

        # Суммарный поток энергии
        Q_total = Q_net_rad + Q_sensible + Q_latent + Q_ground

        # Пересчет в таяние
        if Q_total > 0:
            melt = Q_total * dt_hours * 3600 / LATENT_HEAT_FUSION  # мм
            return melt
        return 0

    def update_snowpack(
        self,
        precipitation: float,
        temperature: float,
        melt: float,
        dt_hours: float = 24
    ):
        """
        Обновление состояния снежного покрова.

        Args:
            precipitation: Осадки (мм)
            temperature: Температура (°C)
            melt: Таяние (мм)
            dt_hours: Временной шаг (часы)
        """
        if self.snowpack is None:
            self.initialize_snowpack()

        # Добавление свежего снега
        if temperature < 0 and precipitation > 0:
            # Новый снег
            new_swe = precipitation
            new_depth = new_swe / self.NEW_SNOW_DENSITY * 1000

            # Смешивание плотностей
            if self.snowpack.snow_water_equivalent > 0:
                total_swe = self.snowpack.snow_water_equivalent + new_swe
                self.snowpack.density = (
                    (self.snowpack.density * self.snowpack.snow_water_equivalent +
                     self.NEW_SNOW_DENSITY * new_swe) / total_swe
                )
            else:
                self.snowpack.density = self.NEW_SNOW_DENSITY

            self.snowpack.snow_water_equivalent += new_swe
            self.snowpack.albedo = 0.85  # Свежий снег

        # Таяние
        actual_melt = min(melt, self.snowpack.snow_water_equivalent)
        self.snowpack.snow_water_equivalent -= actual_melt
        self.snowpack.is_melting = actual_melt > 0

        # Уплотнение снега со временем
        if temperature > -5 and self.snowpack.snow_water_equivalent > 0:
            compaction_rate = 0.01 * (temperature + 5) * (dt_hours / 24)
            self.snowpack.density = min(
                500,
                self.snowpack.density * (1 + compaction_rate)
            )

        # Обновление глубины
        if self.snowpack.snow_water_equivalent > 0:
            self.snowpack.snow_depth = (
                self.snowpack.snow_water_equivalent / self.snowpack.density * 1000
            )
        else:
            self.snowpack.snow_depth = 0

        # Старение альбедо
        if not self.snowpack.is_melting:
            self.snowpack.albedo = max(0.4, self.snowpack.albedo - 0.01 * (dt_hours / 24))

    def calculate_glacier_melt(
        self,
        temperature: float,
        elevation: float,
        dt_hours: float = 24
    ) -> float:
        """
        Расчет таяния ледника.

        Args:
            temperature: Температура на уровне ледника (°C)
            elevation: Высота (м)
            dt_hours: Временной шаг (часы)

        Returns:
            Таяние (мм)
        """
        if self.glacier is None:
            return 0

        # Температурный градиент
        lapse_rate = 0.0065  # °C/м
        ela = self.glacier.equilibrium_line_altitude

        # Температура на высоте
        temp_at_elevation = temperature - lapse_rate * (elevation - 2000)

        if elevation > ela:
            # Зона аккумуляции - меньше таяния
            effective_ddf = self.GLACIER_DEGREE_DAY_FACTOR * 0.3
        else:
            # Зона абляции
            effective_ddf = self.GLACIER_DEGREE_DAY_FACTOR

        melt = self.degree_day_melt(temp_at_elevation, effective_ddf, dt_hours)

        self.glacier.melt_rate = melt

        return melt * self.glacier.area_km2 * 1000  # м³

    def simulate_snow_season(
        self,
        temperature_series: np.ndarray,
        precipitation_series: np.ndarray,
        solar_radiation: np.ndarray = None,
        method: str = "degree_day"
    ) -> Dict[str, np.ndarray]:
        """
        Моделирование снежного сезона.

        Args:
            temperature_series: Ряд температур (°C)
            precipitation_series: Ряд осадков (мм)
            solar_radiation: Ряд солнечной радиации (Вт/м²)
            method: Метод расчета таяния

        Returns:
            Словарь с результатами
        """
        n_steps = len(temperature_series)

        # Инициализация
        self.initialize_snowpack(snow_water_equivalent=0)

        # Массивы результатов
        swe = np.zeros(n_steps)
        depth = np.zeros(n_steps)
        melt = np.zeros(n_steps)
        runoff = np.zeros(n_steps)

        for i in range(n_steps):
            temp = temperature_series[i]
            precip = precipitation_series[i]

            # Расчет таяния
            if method == "degree_day":
                daily_melt = self.degree_day_melt(temp)
            elif method == "energy_balance" and solar_radiation is not None:
                daily_melt = self.energy_balance_melt(
                    temp, solar_radiation[i], 3, 60
                )
            else:
                daily_melt = self.degree_day_melt(temp)

            # Ограничение таяния доступным снегом
            daily_melt = min(daily_melt, self.snowpack.snow_water_equivalent)

            # Обновление снежного покрова
            self.update_snowpack(precip, temp, daily_melt)

            # Сохранение результатов
            swe[i] = self.snowpack.snow_water_equivalent
            depth[i] = self.snowpack.snow_depth
            melt[i] = daily_melt
            runoff[i] = daily_melt  # Упрощение: весь талый снег = сток

        return {
            'snow_water_equivalent': swe,
            'snow_depth': depth,
            'snowmelt': melt,
            'runoff': runoff
        }

    def estimate_peak_snowmelt_flood(
        self,
        max_swe: float,
        melt_period_days: int,
        watershed_area_km2: float
    ) -> Dict:
        """
        Оценка пикового паводка при таянии снега.

        Args:
            max_swe: Максимальный снегозапас (мм)
            melt_period_days: Период активного таяния (дни)
            watershed_area_km2: Площадь водосбора (км²)

        Returns:
            Характеристики паводка
        """
        # Средняя интенсивность таяния
        avg_melt_rate = max_swe / melt_period_days  # мм/день

        # Пиковое таяние (обычно 1.5-2 раза выше среднего)
        peak_melt_rate = avg_melt_rate * 1.7

        # Объем талых вод
        total_volume = max_swe * watershed_area_km2 * 1000  # м³

        # Пиковый расход (упрощенная оценка)
        # Q = (melt_rate * area) / (24 * 3600) * 1000
        peak_discharge = (peak_melt_rate * watershed_area_km2 * 1000) / (24 * 3600)

        # Базовый расход при умеренном таянии
        base_discharge = (avg_melt_rate * watershed_area_km2 * 1000) / (24 * 3600)

        return {
            'max_swe_mm': max_swe,
            'melt_period_days': melt_period_days,
            'avg_melt_rate_mm_day': avg_melt_rate,
            'peak_melt_rate_mm_day': peak_melt_rate,
            'total_volume_m3': total_volume,
            'peak_discharge_m3s': peak_discharge,
            'base_discharge_m3s': base_discharge,
            'flood_duration_days': melt_period_days * 1.2
        }

    def get_seasonal_snowmelt_pattern(
        self,
        region: str = None
    ) -> Dict[str, float]:
        """
        Получение типичного сезонного паттерна таяния для региона.

        Args:
            region: Код региона

        Returns:
            Месячные доли годового таяния
        """
        if region is None:
            region = self.region

        config = REGIONS.get(region)
        if config is None:
            return {}

        avg_elevation = config.avg_elevation

        # Паттерн зависит от высоты
        if avg_elevation > 3000:
            # Высокогорье - позднее таяние
            pattern = {
                'January': 0.0,
                'February': 0.0,
                'March': 0.02,
                'April': 0.08,
                'May': 0.20,
                'June': 0.30,
                'July': 0.25,
                'August': 0.12,
                'September': 0.03,
                'October': 0.0,
                'November': 0.0,
                'December': 0.0
            }
        elif avg_elevation > 2000:
            # Среднегорье
            pattern = {
                'January': 0.0,
                'February': 0.02,
                'March': 0.10,
                'April': 0.25,
                'May': 0.30,
                'June': 0.20,
                'July': 0.10,
                'August': 0.03,
                'September': 0.0,
                'October': 0.0,
                'November': 0.0,
                'December': 0.0
            }
        else:
            # Низкогорье и долины
            pattern = {
                'January': 0.02,
                'February': 0.05,
                'March': 0.20,
                'April': 0.35,
                'May': 0.25,
                'June': 0.10,
                'July': 0.03,
                'August': 0.0,
                'September': 0.0,
                'October': 0.0,
                'November': 0.0,
                'December': 0.0
            }

        return pattern


def example_snowmelt_calculation():
    """
    Пример расчета таяния снега.
    """
    model = SnowmeltModel(region="issyk_kul")

    # Инициализация с накопленным снегом
    model.initialize_snowpack(snow_water_equivalent=200)

    print("Моделирование таяния снега")
    print("=" * 50)
    print(f"Начальный снегозапас: {model.snowpack.snow_water_equivalent} мм")
    print("-" * 50)

    # Имитация 10 дней таяния
    temperatures = [2, 5, 8, 10, 12, 8, 5, 10, 15, 12]

    for day, temp in enumerate(temperatures, 1):
        melt = model.degree_day_melt(temp)
        model.update_snowpack(0, temp, melt)

        print(f"День {day}: T={temp}°C, Таяние={melt:.1f} мм, "
              f"Остаток={model.snowpack.snow_water_equivalent:.1f} мм")

    return model


if __name__ == "__main__":
    example_snowmelt_calculation()
