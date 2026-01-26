"""
Модуль визуализации результатов моделирования паводков.
"""

from .plots import (
    plot_precipitation_map,
    plot_hydrograph,
    plot_flood_risk_map,
    plot_terrain,
    create_dashboard
)
from .maps import FloodRiskMapper

__all__ = [
    'plot_precipitation_map',
    'plot_hydrograph',
    'plot_flood_risk_map',
    'plot_terrain',
    'create_dashboard',
    'FloodRiskMapper'
]
