"""
Enhanced visualization module with animated charts, calendar heatmaps,
scenario comparison, and export functionality.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import base64
import io


# Color palette for flood risk levels
RISK_COLORS = {
    'low': '#2ecc71',
    'moderate': '#f39c12',
    'high': '#e67e22',
    'critical': '#e74c3c'
}

RISK_LABELS = {
    'low': 'Низкий',
    'moderate': 'Умеренный',
    'high': 'Высокий',
    'critical': 'Критический'
}

# Professional color schemes
COLOR_SCHEMES = {
    'water': ['#e3f2fd', '#90caf9', '#42a5f5', '#1e88e5', '#1565c0', '#0d47a1'],
    'risk': ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
    'temperature': ['#2196f3', '#64b5f6', '#fff59d', '#ffb74d', '#ef5350'],
    'precipitation': ['#ffffff', '#bbdefb', '#64b5f6', '#1976d2', '#0d47a1']
}


def create_animated_hydrograph(
    results: pd.DataFrame,
    input_data: pd.DataFrame,
    flood_threshold: float = 100.0,
    animation_speed: int = 100
) -> go.Figure:
    """
    Create an animated hydrograph showing discharge progression over time.

    Args:
        results: Simulation results DataFrame with timestamp and discharge_m3s
        input_data: Input data DataFrame with precipitation
        flood_threshold: Flood threshold value
        animation_speed: Animation frame duration in ms

    Returns:
        Plotly Figure with animation
    """
    # Prepare data - aggregate to daily for smoother animation
    results = results.copy()
    results['date'] = pd.to_datetime(results['timestamp']).dt.date

    daily_data = results.groupby('date').agg({
        'discharge_m3s': 'max',
        'risk_level': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'low'
    }).reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data['day_num'] = range(len(daily_data))

    # Create frames for animation
    frames = []
    for i in range(1, len(daily_data) + 1):
        frame_data = daily_data.iloc[:i].copy()
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=frame_data['date'],
                    y=frame_data['discharge_m3s'],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(30, 136, 229, 0.3)',
                    line=dict(color='#1e88e5', width=3),
                    name='Расход воды'
                )
            ],
            name=str(i)
        ))

    # Initial figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[daily_data['date'].iloc[0]],
                y=[daily_data['discharge_m3s'].iloc[0]],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(30, 136, 229, 0.3)',
                line=dict(color='#1e88e5', width=3),
                name='Расход воды'
            )
        ],
        frames=frames
    )

    # Add flood threshold line
    fig.add_hline(
        y=flood_threshold,
        line_dash='dash',
        line_color='#e74c3c',
        annotation_text=f'Порог паводка: {flood_threshold} м³/с',
        annotation_position='top right'
    )

    # Add animation controls
    fig.update_layout(
        title=dict(
            text='<b>Анимированный гидрограф</b>',
            font=dict(size=20, color='#1a237e')
        ),
        xaxis=dict(
            title='Дата',
            range=[daily_data['date'].min(), daily_data['date'].max()],
            tickformat='%d %b',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Расход воды (м³/с)',
            range=[0, daily_data['discharge_m3s'].max() * 1.1],
            gridcolor='rgba(0,0,0,0.1)'
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='▶ Воспроизвести',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': animation_speed, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    ),
                    dict(
                        label='⏸ Пауза',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏮ Сброс',
                        method='animate',
                        args=[[str(1)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 14},
                'prefix': 'День: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05,
            'y': 0,
            'steps': [
                {
                    'args': [[str(i)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(daily_data['date'].iloc[i-1].strftime('%d.%m')),
                    'method': 'animate'
                }
                for i in range(1, len(daily_data) + 1)
            ]
        }],
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_calendar_heatmap(
    results: pd.DataFrame,
    value_column: str = 'discharge_m3s',
    title: str = 'Календарь риска паводков'
) -> go.Figure:
    """
    Create a GitHub-style calendar heatmap showing flood risk by day.

    Args:
        results: Simulation results DataFrame
        value_column: Column to use for heatmap values
        title: Chart title

    Returns:
        Plotly Figure with calendar heatmap
    """
    # Aggregate to daily data
    results = results.copy()
    results['date'] = pd.to_datetime(results['timestamp']).dt.date

    daily_data = results.groupby('date').agg({
        value_column: 'max',
        'risk_level': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'low'
    }).reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])

    # Calculate week and day of week
    daily_data['week'] = daily_data['date'].dt.isocalendar().week
    daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
    daily_data['month'] = daily_data['date'].dt.month
    daily_data['day'] = daily_data['date'].dt.day

    # Map risk levels to numeric values for coloring
    risk_map = {'low': 0, 'moderate': 1, 'high': 2, 'critical': 3}
    daily_data['risk_numeric'] = daily_data['risk_level'].map(risk_map)

    # Create the heatmap matrix
    # Group by week number (adjust for year boundary)
    min_date = daily_data['date'].min()
    daily_data['week_num'] = ((daily_data['date'] - min_date).dt.days // 7).astype(int)

    # Pivot to create matrix
    heatmap_data = daily_data.pivot_table(
        index='day_of_week',
        columns='week_num',
        values='risk_numeric',
        aggfunc='max'
    ).fillna(-1)

    # Create hover text
    hover_data = daily_data.pivot_table(
        index='day_of_week',
        columns='week_num',
        values=value_column,
        aggfunc='max'
    ).fillna(0)

    date_data = daily_data.pivot_table(
        index='day_of_week',
        columns='week_num',
        values='date',
        aggfunc='first'
    )

    # Custom colorscale for risk levels
    colorscale = [
        [0, '#f0f0f0'],      # No data
        [0.25, '#2ecc71'],   # Low
        [0.5, '#f39c12'],    # Moderate
        [0.75, '#e67e22'],   # High
        [1.0, '#e74c3c']     # Critical
    ]

    # Create hover text matrix
    hover_text = []
    for dow in range(7):
        row = []
        for week in heatmap_data.columns:
            if dow in heatmap_data.index and week in heatmap_data.columns:
                val = hover_data.loc[dow, week] if dow in hover_data.index else 0
                date = date_data.loc[dow, week] if dow in date_data.index and pd.notna(date_data.loc[dow, week]) else None
                risk = heatmap_data.loc[dow, week]
                risk_label = ['Низкий', 'Умеренный', 'Высокий', 'Критический'][int(risk)] if risk >= 0 else 'Нет данных'
                date_str = date.strftime('%d.%m.%Y') if date else ''
                row.append(f'Дата: {date_str}<br>Расход: {val:.1f} м³/с<br>Риск: {risk_label}')
            else:
                row.append('')
        hover_text.append(row)

    day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(len(heatmap_data.columns))),
        y=day_names,
        colorscale=colorscale,
        zmin=-1,
        zmax=3,
        hovertext=hover_text,
        hoverinfo='text',
        showscale=True,
        colorbar=dict(
            title='Уровень риска',
            tickvals=[0, 1, 2, 3],
            ticktext=['Низкий', 'Умеренный', 'Высокий', 'Критический'],
            len=0.6
        )
    ))

    # Add month labels
    month_positions = []
    month_labels = []
    months_ru = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

    current_month = None
    for idx, row in daily_data.iterrows():
        if row['month'] != current_month:
            current_month = row['month']
            month_positions.append(row['week_num'])
            month_labels.append(months_ru[current_month - 1])

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(size=18, color='#1a237e'),
            x=0.5
        ),
        xaxis=dict(
            title='Неделя',
            tickvals=month_positions,
            ticktext=month_labels,
            side='top',
            gridcolor='white'
        ),
        yaxis=dict(
            title='',
            autorange='reversed',
            gridcolor='white'
        ),
        height=280,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_scenario_comparison(
    scenarios: Dict[str, pd.DataFrame],
    flood_threshold: float = 100.0
) -> go.Figure:
    """
    Create a side-by-side comparison of multiple scenarios.

    Args:
        scenarios: Dictionary mapping scenario names to result DataFrames
        flood_threshold: Flood threshold value

    Returns:
        Plotly Figure with scenario comparison
    """
    scenario_colors = {
        'normal': '#3498db',
        'wet': '#2ecc71',
        'dry': '#e67e22',
        'extreme': '#e74c3c',
        'Нормальный': '#3498db',
        'Влажный': '#2ecc71',
        'Засушливый': '#e67e22',
        'Экстремальный': '#e74c3c'
    }

    n_scenarios = len(scenarios)

    # Create subplots - 2 rows: time series on top, statistics below
    fig = make_subplots(
        rows=2, cols=n_scenarios,
        subplot_titles=list(scenarios.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{'type': 'scatter'}] * n_scenarios,
               [{'type': 'bar'}] * n_scenarios]
    )

    stats_data = []

    for i, (name, df) in enumerate(scenarios.items(), 1):
        color = scenario_colors.get(name, '#7f8c8d')

        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['discharge_m3s'],
                mode='lines',
                name=name,
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}',
                line=dict(color=color, width=2),
                showlegend=(i == 1)
            ),
            row=1, col=i
        )

        # Add threshold line
        fig.add_hline(
            y=flood_threshold,
            line_dash='dash',
            line_color='#e74c3c',
            row=1, col=i
        )

        # Calculate statistics
        max_discharge = df['discharge_m3s'].max()
        mean_discharge = df['discharge_m3s'].mean()
        hours_above = (df['discharge_m3s'] > flood_threshold).sum()

        stats_data.append({
            'name': name,
            'max': max_discharge,
            'mean': mean_discharge,
            'hours_above': hours_above,
            'color': color
        })

        # Statistics bar chart
        fig.add_trace(
            go.Bar(
                x=['Макс', 'Сред', 'Часы>'],
                y=[max_discharge, mean_discharge, hours_above],
                marker_color=[color, color, '#e74c3c' if hours_above > 0 else color],
                showlegend=False,
                text=[f'{max_discharge:.0f}', f'{mean_discharge:.0f}', f'{hours_above}'],
                textposition='outside'
            ),
            row=2, col=i
        )

    fig.update_layout(
        title=dict(
            text='<b>Сравнение сценариев</b>',
            font=dict(size=20, color='#1a237e'),
            x=0.5
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update axes
    for i in range(1, n_scenarios + 1):
        fig.update_xaxes(title_text='Время', row=1, col=i, tickformat='%d.%m')
        fig.update_yaxes(title_text='м³/с' if i == 1 else '', row=1, col=i)
        fig.update_yaxes(title_text='Значение' if i == 1 else '', row=2, col=i)

    return fig


def create_enhanced_hydrograph(
    results: pd.DataFrame,
    input_data: pd.DataFrame,
    flood_threshold: float = 100.0,
    show_risk_zones: bool = True
) -> go.Figure:
    """
    Create an enhanced hydrograph with better styling and annotations.

    Args:
        results: Simulation results
        input_data: Input weather data
        flood_threshold: Flood threshold
        show_risk_zones: Whether to show colored risk zones

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            '<b>Расход воды</b>',
            '<b>Осадки</b>',
            '<b>Температура</b>'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Discharge with gradient fill
    fig.add_trace(
        go.Scatter(
            x=results['timestamp'],
            y=results['discharge_m3s'],
            mode='lines',
            name='Расход воды',
            line=dict(color='#1565c0', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(21, 101, 192, 0.2)'
        ),
        row=1, col=1
    )

    # Add risk zone shading
    if show_risk_zones and 'risk_level' in results.columns:
        # Find contiguous regions of each risk level
        for risk, color in RISK_COLORS.items():
            mask = results['risk_level'] == risk
            if mask.any():
                # Add scatter trace for risk highlighting
                risk_data = results[mask]
                fig.add_trace(
                    go.Scatter(
                        x=risk_data['timestamp'],
                        y=risk_data['discharge_m3s'],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=4,
                            opacity=0.6
                        ),
                        name=f'Риск: {RISK_LABELS[risk]}',
                        showlegend=True
                    ),
                    row=1, col=1
                )

    # Add flood threshold with annotation
    fig.add_hline(
        y=flood_threshold,
        line_dash='dash',
        line_color='#c62828',
        line_width=2,
        row=1, col=1
    )

    fig.add_annotation(
        x=results['timestamp'].iloc[-1],
        y=flood_threshold,
        text=f'Порог: {flood_threshold} м³/с',
        showarrow=False,
        yshift=10,
        font=dict(color='#c62828', size=11),
        row=1, col=1
    )

    # Mark peak discharge
    peak_idx = results['discharge_m3s'].idxmax()
    peak_time = results.loc[peak_idx, 'timestamp']
    peak_value = results.loc[peak_idx, 'discharge_m3s']

    fig.add_annotation(
        x=peak_time,
        y=peak_value,
        text=f'Пик: {peak_value:.1f} м³/с',
        showarrow=True,
        arrowhead=2,
        arrowcolor='#1565c0',
        font=dict(color='#1565c0', size=11),
        bgcolor='white',
        bordercolor='#1565c0',
        row=1, col=1
    )

    # Precipitation bars with gradient colors
    precip_colors = [
        '#bbdefb' if p < 5 else
        '#64b5f6' if p < 15 else
        '#1976d2' if p < 30 else
        '#0d47a1'
        for p in input_data['precipitation_mm']
    ]

    fig.add_trace(
        go.Bar(
            x=input_data['timestamp'],
            y=input_data['precipitation_mm'],
            name='Осадки',
            marker_color=precip_colors,
            opacity=0.8
        ),
        row=2, col=1
    )

    # Temperature with gradient color
    temp_colors = [
        '#2196f3' if t < 0 else
        '#64b5f6' if t < 5 else
        '#fff59d' if t < 15 else
        '#ffb74d' if t < 25 else
        '#ef5350'
        for t in input_data['temperature_c']
    ]

    fig.add_trace(
        go.Scatter(
            x=input_data['timestamp'],
            y=input_data['temperature_c'],
            mode='lines+markers',
            name='Температура',
            line=dict(color='#ff7043', width=2),
            marker=dict(color=temp_colors, size=4)
        ),
        row=3, col=1
    )

    # Zero line for temperature
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=3, col=1)

    # Layout updates
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        margin=dict(l=60, r=30, t=80, b=30)
    )

    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        tickformat='%d %b'
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)'
    )

    fig.update_yaxes(title_text='м³/с', row=1, col=1)
    fig.update_yaxes(title_text='мм', row=2, col=1)
    fig.update_yaxes(title_text='°C', row=3, col=1)

    return fig


def create_risk_distribution_chart(
    results: pd.DataFrame,
    chart_type: str = 'sunburst'
) -> go.Figure:
    """
    Create an interactive risk distribution chart.

    Args:
        results: Simulation results
        chart_type: 'sunburst', 'treemap', or 'pie'

    Returns:
        Plotly Figure
    """
    # Aggregate risk levels
    risk_counts = results['risk_level'].value_counts()

    data = []
    for risk in ['low', 'moderate', 'high', 'critical']:
        if risk in risk_counts.index:
            data.append({
                'risk': risk,
                'label': RISK_LABELS[risk],
                'count': risk_counts[risk],
                'color': RISK_COLORS[risk],
                'percentage': risk_counts[risk] / len(results) * 100
            })

    df = pd.DataFrame(data)

    if chart_type == 'sunburst':
        fig = px.sunburst(
            df,
            names='label',
            values='count',
            color='label',
            color_discrete_map={RISK_LABELS[k]: v for k, v in RISK_COLORS.items()}
        )
    elif chart_type == 'treemap':
        fig = px.treemap(
            df,
            names='label',
            values='count',
            color='label',
            color_discrete_map={RISK_LABELS[k]: v for k, v in RISK_COLORS.items()}
        )
    else:  # pie
        fig = go.Figure(data=[go.Pie(
            labels=df['label'],
            values=df['count'],
            hole=0.4,
            marker_colors=df['color'],
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05 if r == 'critical' else 0 for r in df['risk']]
        )])

    fig.update_layout(
        title=dict(
            text='<b>Распределение уровней риска</b>',
            font=dict(size=18, color='#1a237e'),
            x=0.5
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def create_correlation_matrix(
    results: pd.DataFrame,
    input_data: pd.DataFrame
) -> go.Figure:
    """
    Create correlation matrix heatmap between variables.

    Args:
        results: Simulation results
        input_data: Input data

    Returns:
        Plotly Figure
    """
    # Combine relevant columns
    combined = pd.DataFrame({
        'Расход': results['discharge_m3s'],
        'Поверх. сток': results['surface_runoff_mm'],
        'Таяние снега': results['snowmelt_mm'],
        'Осадки': input_data['precipitation_mm'],
        'Температура': input_data['temperature_c']
    })

    corr_matrix = combined.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=12),
        hoverongaps=False,
        colorbar=dict(title='Корреляция')
    ))

    fig.update_layout(
        title=dict(
            text='<b>Корреляционная матрица</b>',
            font=dict(size=18, color='#1a237e'),
            x=0.5
        ),
        height=450,
        width=550,
        xaxis=dict(tickangle=45),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def get_chart_download_config() -> dict:
    """
    Get Plotly config for chart download buttons.

    Returns:
        Config dictionary for Plotly
    """
    return {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'flood_chart',
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'downloadSVG',
            'downloadPDF'
        ],
        'modeBarButtonsToRemove': [
            'lasso2d',
            'select2d'
        ]
    }


def add_download_buttons_html() -> str:
    """
    Generate HTML for custom download buttons (for Streamlit).

    Returns:
        HTML string with download functionality
    """
    return """
    <style>
    .download-btn {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px;
        background-color: #1e88e5;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        transition: background-color 0.3s;
    }
    .download-btn:hover {
        background-color: #1565c0;
    }
    .download-container {
        margin: 10px 0;
        text-align: right;
    }
    </style>
    """


def create_mini_sparkline(
    values: List[float],
    color: str = '#1e88e5',
    height: int = 50
) -> go.Figure:
    """
    Create a minimal sparkline chart for dashboard display.

    Args:
        values: List of values to plot
        color: Line color
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=values,
        mode='lines',
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
        line=dict(color=color, width=1.5),
        hoverinfo='y'
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_gauge_chart(
    value: float,
    max_value: float,
    title: str,
    thresholds: Dict[str, float] = None
) -> go.Figure:
    """
    Create a gauge chart for displaying current values.

    Args:
        value: Current value
        max_value: Maximum value for gauge
        title: Chart title
        thresholds: Dictionary of threshold levels

    Returns:
        Plotly Figure
    """
    if thresholds is None:
        thresholds = {
            'low': max_value * 0.25,
            'moderate': max_value * 0.5,
            'high': max_value * 0.75
        }

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1},
            'bar': {'color': '#1e88e5'},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, thresholds['low']], 'color': '#e8f5e9'},
                {'range': [thresholds['low'], thresholds['moderate']], 'color': '#fff3e0'},
                {'range': [thresholds['moderate'], thresholds['high']], 'color': '#fbe9e7'},
                {'range': [thresholds['high'], max_value], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': '#e74c3c', 'width': 4},
                'thickness': 0.75,
                'value': thresholds['high']
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white'
    )

    return fig
