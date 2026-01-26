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
from .enhanced_charts import (
    create_animated_hydrograph,
    create_calendar_heatmap,
    create_scenario_comparison,
    create_enhanced_hydrograph,
    create_risk_distribution_chart,
    create_correlation_matrix,
    get_chart_download_config,
    add_download_buttons_html,
    create_mini_sparkline,
    create_gauge_chart,
    RISK_COLORS,
    RISK_LABELS,
    COLOR_SCHEMES
)

__all__ = [
    'plot_precipitation_map',
    'plot_hydrograph',
    'plot_flood_risk_map',
    'plot_terrain',
    'create_dashboard',
    'FloodRiskMapper',
    # Enhanced charts
    'create_animated_hydrograph',
    'create_calendar_heatmap',
    'create_scenario_comparison',
    'create_enhanced_hydrograph',
    'create_risk_distribution_chart',
    'create_correlation_matrix',
    'get_chart_download_config',
    'add_download_buttons_html',
    'create_mini_sparkline',
    'create_gauge_chart',
    'RISK_COLORS',
    'RISK_LABELS',
    'COLOR_SCHEMES'
]
