"""
Конфигурация системы моделирования паводков для Кыргызстана.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

# Базовый путь проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass
class RegionConfig:
    """Конфигурация региона для моделирования."""
    name: str
    name_en: str
    bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    area_km2: float
    avg_elevation: float  # метры
    main_rivers: List[str]


# Регионы Кыргызстана для моделирования
REGIONS: Dict[str, RegionConfig] = {
    "chui": RegionConfig(
        name="Чуйская область",
        name_en="Chui Region",
        bounds=(73.5, 42.0, 76.5, 43.5),
        area_km2=20200,
        avg_elevation=1500,
        main_rivers=["Чу", "Аламедин", "Ала-Арча"]
    ),
    "issyk_kul": RegionConfig(
        name="Иссык-Кульская область",
        name_en="Issyk-Kul Region",
        bounds=(76.0, 41.5, 80.5, 43.0),
        area_km2=43100,
        avg_elevation=2500,
        main_rivers=["Тюп", "Джергалан", "Каракол"]
    ),
    "naryn": RegionConfig(
        name="Нарынская область",
        name_en="Naryn Region",
        bounds=(74.5, 40.5, 77.5, 42.5),
        area_km2=45200,
        avg_elevation=2800,
        main_rivers=["Нарын", "Ат-Баши", "Кекемерен"]
    ),
    "osh": RegionConfig(
        name="Ошская область",
        name_en="Osh Region",
        bounds=(71.5, 39.5, 74.5, 41.5),
        area_km2=29200,
        avg_elevation=2200,
        main_rivers=["Ак-Буура", "Кара-Дарья"]
    ),
    "jalal_abad": RegionConfig(
        name="Джалал-Абадская область",
        name_en="Jalal-Abad Region",
        bounds=(70.5, 40.5, 74.0, 42.5),
        area_km2=33700,
        avg_elevation=2000,
        main_rivers=["Нарын", "Чаткал", "Кара-Суу"]
    ),
    "talas": RegionConfig(
        name="Таласская область",
        name_en="Talas Region",
        bounds=(70.5, 42.0, 73.5, 43.0),
        area_km2=11400,
        avg_elevation=2200,
        main_rivers=["Талас", "Кара-Буура"]
    ),
    "batken": RegionConfig(
        name="Баткенская область",
        name_en="Batken Region",
        bounds=(69.5, 39.5, 72.0, 41.0),
        area_km2=17000,
        avg_elevation=1800,
        main_rivers=["Исфара", "Сох", "Исфайрам"]
    )
}


# Параметры модели SCS-CN
@dataclass
class SCSParameters:
    """Параметры метода SCS Curve Number."""
    # Номера кривых стока для различных типов поверхности
    CN_URBAN: int = 89  # Городская застройка
    CN_AGRICULTURAL: int = 78  # Сельскохозяйственные земли
    CN_FOREST: int = 60  # Лесные массивы
    CN_MEADOW: int = 69  # Луга и пастбища
    CN_BARE_SOIL: int = 82  # Голая почва
    CN_ROCK: int = 96  # Скальные породы
    CN_GLACIER: int = 98  # Ледники

    # Коэффициент начальных абстракций
    INITIAL_ABSTRACTION_RATIO: float = 0.2


# Параметры гидрологической модели
@dataclass
class HydrologicalParams:
    """Параметры гидрологической модели."""
    # Время концентрации (часы)
    DEFAULT_CONCENTRATION_TIME: float = 6.0

    # Коэффициент шероховатости Маннинга
    MANNING_N_RIVER: float = 0.035
    MANNING_N_FLOODPLAIN: float = 0.05

    # Коэффициент испарения
    EVAPORATION_COEFF: float = 0.7

    # Коэффициент инфильтрации (мм/час)
    INFILTRATION_RATE: float = 10.0

    # Температура таяния снега (°C)
    SNOWMELT_TEMP: float = 0.0

    # Коэффициент таяния (мм/°C/день)
    MELT_FACTOR: float = 4.5


# Пороговые значения для предупреждений о паводках
@dataclass
class FloodThresholds:
    """Пороговые значения уровней опасности паводков."""
    # Осадки (мм/сутки)
    PRECIPITATION_LOW: float = 20.0
    PRECIPITATION_MODERATE: float = 40.0
    PRECIPITATION_HIGH: float = 60.0
    PRECIPITATION_EXTREME: float = 100.0

    # Расход воды (м³/с) - относительные множители к норме
    DISCHARGE_NORMAL: float = 1.0
    DISCHARGE_ELEVATED: float = 1.5
    DISCHARGE_HIGH: float = 2.0
    DISCHARGE_CRITICAL: float = 3.0


# Сезонные параметры для Кыргызстана
SEASONAL_PARAMS = {
    "spring": {  # Март-Май
        "months": [3, 4, 5],
        "snowmelt_active": True,
        "flood_risk_multiplier": 1.5,
        "description": "Весенний период - активное таяние снега"
    },
    "summer": {  # Июнь-Август
        "months": [6, 7, 8],
        "snowmelt_active": True,  # Ледники
        "flood_risk_multiplier": 1.3,
        "description": "Летний период - таяние ледников, ливневые дожди"
    },
    "autumn": {  # Сентябрь-Ноябрь
        "months": [9, 10, 11],
        "snowmelt_active": False,
        "flood_risk_multiplier": 0.8,
        "description": "Осенний период - снижение стока"
    },
    "winter": {  # Декабрь-Февраль
        "months": [12, 1, 2],
        "snowmelt_active": False,
        "flood_risk_multiplier": 0.3,
        "description": "Зимний период - минимальный сток"
    }
}


# Настройки визуализации
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 100,
    "colormap_precipitation": "Blues",
    "colormap_flood_risk": "RdYlGn_r",
    "colormap_elevation": "terrain",
    "font_family": "DejaVu Sans"
}
