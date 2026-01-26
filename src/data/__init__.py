"""
Модуль работы с данными для моделирования паводков.
"""

from .precipitation import PrecipitationDataLoader, generate_synthetic_precipitation
from .terrain import TerrainAnalyzer
from .preprocessing import DataPreprocessor

__all__ = [
    'PrecipitationDataLoader',
    'generate_synthetic_precipitation',
    'TerrainAnalyzer',
    'DataPreprocessor'
]
