"""
Web components for Streamlit integration.
Contains functions to render enhanced visualization tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .enhanced_charts import (
    create_animated_hydrograph,
    create_calendar_heatmap,
    create_scenario_comparison,
    create_enhanced_hydrograph,
    create_risk_distribution_chart,
    create_correlation_matrix,
    get_chart_download_config,
    create_gauge_chart
)


def render_enhanced_visualizations_tab(
    results: pd.DataFrame,
    input_data: pd.DataFrame,
    flood_threshold: float,
    scenario_data: Optional[Dict[str, pd.DataFrame]] = None
):
    """
    Render the enhanced visualizations tab content.

    Args:
        results: Simulation results DataFrame
        input_data: Input weather data DataFrame
        flood_threshold: Flood threshold value
        scenario_data: Optional dictionary of scenario results for comparison
    """
    st.subheader("Advanced Visualization")

    # Chart download config
    chart_config = get_chart_download_config()

    # Sub-tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Animated Hydrograph",
        "Risk Calendar",
        "Scenario Comparison",
        "Correlation Analysis"
    ])

    with viz_tab1:
        st.markdown("### Animated Hydrograph")
        st.markdown("""
        Watch the discharge evolve over time. Use the play button to animate
        or drag the slider to a specific day.
        """)

        # Animation speed control
        animation_speed = st.slider(
            "Animation speed (ms per frame)",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Lower values = faster animation"
        )

        fig_animated = create_animated_hydrograph(
            results,
            input_data,
            flood_threshold=flood_threshold,
            animation_speed=animation_speed
        )
        st.plotly_chart(fig_animated, use_container_width=True, config=chart_config)

        # Download button for static version
        st.markdown("---")
        if st.button("Download Static Hydrograph", key="dl_hydrograph"):
            fig_static = create_enhanced_hydrograph(
                results, input_data, flood_threshold
            )
            st.plotly_chart(fig_static, use_container_width=True, config=chart_config)

    with viz_tab2:
        st.markdown("### Flood Risk Calendar")
        st.markdown("""
        GitHub-style calendar heatmap showing daily flood risk levels.
        Darker colors indicate higher risk.
        """)

        fig_calendar = create_calendar_heatmap(
            results,
            value_column='discharge_m3s',
            title='Daily Flood Risk Overview'
        )
        st.plotly_chart(fig_calendar, use_container_width=True, config=chart_config)

        # Risk distribution chart
        st.markdown("### Risk Level Distribution")

        chart_type = st.radio(
            "Chart type",
            options=['pie', 'sunburst', 'treemap'],
            horizontal=True,
            help="Select visualization style for risk distribution"
        )

        fig_risk = create_risk_distribution_chart(results, chart_type=chart_type)
        st.plotly_chart(fig_risk, use_container_width=True, config=chart_config)

    with viz_tab3:
        st.markdown("### Scenario Comparison")

        if scenario_data and len(scenario_data) > 1:
            st.markdown("""
            Compare different precipitation scenarios side by side.
            See how varying conditions affect flood risk.
            """)

            fig_comparison = create_scenario_comparison(
                scenario_data,
                flood_threshold=flood_threshold
            )
            st.plotly_chart(fig_comparison, use_container_width=True, config=chart_config)
        else:
            st.info("""
            To compare scenarios, run simulations with different precipitation
            scenarios and they will appear here for comparison.

            Available scenarios:
            - Normal: Average precipitation conditions
            - Wet: +50% precipitation
            - Dry: -50% precipitation
            - Extreme: +100% precipitation
            """)

            # Option to generate comparison data
            if st.button("Generate Scenario Comparison", type="primary"):
                st.info("Run simulations for each scenario from the sidebar to generate comparison data.")

    with viz_tab4:
        st.markdown("### Correlation Analysis")
        st.markdown("""
        Explore relationships between hydrological variables.
        Strong correlations appear in darker colors.
        """)

        fig_corr = create_correlation_matrix(results, input_data)
        st.plotly_chart(fig_corr, use_container_width=True, config=chart_config)

        # Gauge charts for key metrics
        st.markdown("### Real-time Gauges")

        col1, col2, col3 = st.columns(3)

        with col1:
            current_discharge = results['discharge_m3s'].iloc[-1]
            max_discharge = results['discharge_m3s'].max()
            fig_gauge1 = create_gauge_chart(
                current_discharge,
                max_discharge * 1.2,
                "Current Discharge (m³/s)",
                thresholds={
                    'low': flood_threshold * 0.5,
                    'moderate': flood_threshold * 0.75,
                    'high': flood_threshold
                }
            )
            st.plotly_chart(fig_gauge1, use_container_width=True, config=chart_config)

        with col2:
            total_precip = input_data['precipitation_mm'].sum()
            max_expected = total_precip * 1.5
            fig_gauge2 = create_gauge_chart(
                total_precip,
                max_expected,
                "Total Precipitation (mm)",
                thresholds={
                    'low': max_expected * 0.3,
                    'moderate': max_expected * 0.5,
                    'high': max_expected * 0.75
                }
            )
            st.plotly_chart(fig_gauge2, use_container_width=True, config=chart_config)

        with col3:
            flood_hours = (results['discharge_m3s'] > flood_threshold).sum()
            total_hours = len(results)
            fig_gauge3 = create_gauge_chart(
                flood_hours,
                total_hours * 0.3,  # Max expected flood hours
                "Flood Hours",
                thresholds={
                    'low': total_hours * 0.05,
                    'moderate': total_hours * 0.1,
                    'high': total_hours * 0.2
                }
            )
            st.plotly_chart(fig_gauge3, use_container_width=True, config=chart_config)


def render_download_section():
    """Render the chart download section with format options."""
    st.markdown("### Export Charts")
    st.markdown("""
    All charts support interactive export. Use the camera icon in the top-right
    corner of each chart to download as:
    - **PNG**: High-resolution image (default)
    - **SVG**: Vector format for publications
    - **WebP**: Optimized web format

    You can also:
    - Zoom and pan charts interactively
    - Select data points to see details
    - Use the compare data tool for side-by-side analysis
    """)
