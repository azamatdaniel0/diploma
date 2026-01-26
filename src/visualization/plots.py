"""
Модуль построения графиков и диаграмм.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime

from ..config import VISUALIZATION_CONFIG, REGIONS, FloodThresholds


def setup_plot_style():
    """Настройка стиля графиков."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figure_size']
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_precipitation_map(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    precip_grid: np.ndarray,
    title: str = "Распределение осадков",
    region: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Построение карты осадков.

    Args:
        lon_grid: Сетка долгот
        lat_grid: Сетка широт
        precip_grid: Сетка осадков
        title: Заголовок
        region: Код региона для отображения границ
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Создание сетки координат
    if lon_grid.ndim == 1:
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
    else:
        LON, LAT = lon_grid, lat_grid

    # Построение карты
    cmap = plt.cm.Blues
    cmap.set_under('white')

    im = ax.pcolormesh(
        LON, LAT, precip_grid,
        cmap=cmap,
        vmin=0.1,
        shading='auto'
    )

    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax, extend='max')
    cbar.set_label('Осадки (мм)', fontsize=12)

    # Границы региона
    if region and region in REGIONS:
        bounds = REGIONS[region].bounds
        ax.plot(
            [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]],
            [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]],
            'r-', linewidth=2, label=REGIONS[region].name
        )
        ax.legend()

    ax.set_xlabel('Долгота (°E)')
    ax.set_ylabel('Широта (°N)')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_hydrograph(
    time: Union[np.ndarray, pd.DatetimeIndex],
    discharge: np.ndarray,
    precipitation: np.ndarray = None,
    threshold: float = None,
    title: str = "Гидрограф стока",
    save_path: str = None
) -> plt.Figure:
    """
    Построение гидрографа с осадками.

    Args:
        time: Временной ряд
        discharge: Расход воды (м³/с)
        precipitation: Осадки (мм), опционально
        threshold: Пороговый расход для отметки
        title: Заголовок
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    if precipitation is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 3])
    else:
        fig, ax2 = plt.subplots(figsize=(12, 6))
        ax1 = None

    # График осадков (сверху, инвертированный)
    if ax1 is not None:
        ax1.bar(time, precipitation, color='steelblue', alpha=0.7, width=0.8)
        ax1.set_ylabel('Осадки (мм)')
        ax1.set_ylim(ax1.get_ylim()[::-1])  # Инверсия оси
        ax1.set_xticklabels([])
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

    # График расхода
    ax2.plot(time, discharge, 'b-', linewidth=2, label='Расход воды')
    ax2.fill_between(time, 0, discharge, alpha=0.3)

    if threshold is not None:
        ax2.axhline(y=threshold, color='r', linestyle='--',
                   linewidth=2, label=f'Порог ({threshold} м³/с)')

        # Заливка области превышения
        above_threshold = discharge > threshold
        ax2.fill_between(
            time, threshold, discharge,
            where=above_threshold,
            color='red', alpha=0.3,
            label='Превышение порога'
        )

    ax2.set_xlabel('Время')
    ax2.set_ylabel('Расход (м³/с)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Форматирование дат
    if isinstance(time[0], (datetime, pd.Timestamp)):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time) // 10)))
        plt.xticks(rotation=45)

    if ax1 is None:
        ax2.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_flood_risk_map(
    risk_grid: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    title: str = "Карта риска паводков",
    region: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Построение карты риска паводков.

    Args:
        risk_grid: Сетка уровней риска (0-4)
        lon_grid: Сетка долгот
        lat_grid: Сетка широт
        title: Заголовок
        region: Код региона
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Цветовая карта для уровней риска
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    labels = ['Низкий', 'Умеренный', 'Повышенный', 'Высокий', 'Критический']
    cmap = LinearSegmentedColormap.from_list('flood_risk', colors, N=5)

    # Создание сетки
    if lon_grid.ndim == 1:
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
    else:
        LON, LAT = lon_grid, lat_grid

    # Построение карты
    im = ax.pcolormesh(
        LON, LAT, risk_grid,
        cmap=cmap,
        vmin=0, vmax=4,
        shading='auto'
    )

    # Цветовая шкала с подписями
    cbar = plt.colorbar(im, ax=ax, ticks=[0.4, 1.2, 2.0, 2.8, 3.6])
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Уровень риска', fontsize=12)

    # Границы региона
    if region and region in REGIONS:
        bounds = REGIONS[region].bounds
        ax.plot(
            [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]],
            [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]],
            'k-', linewidth=2
        )

    ax.set_xlabel('Долгота (°E)')
    ax.set_ylabel('Широта (°N)')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_terrain(
    dem: np.ndarray,
    slope: np.ndarray = None,
    flow_accumulation: np.ndarray = None,
    title: str = "Рельеф местности",
    save_path: str = None
) -> plt.Figure:
    """
    Построение карты рельефа.

    Args:
        dem: Цифровая модель рельефа
        slope: Карта уклонов (опционально)
        flow_accumulation: Аккумуляция стока (опционально)
        title: Заголовок
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    n_plots = 1 + (slope is not None) + (flow_accumulation is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    # DEM
    ax = axes[0]
    im = ax.imshow(dem, cmap='terrain', origin='lower')
    ax.set_title('Высоты (м)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Уклоны
    if slope is not None:
        ax = axes[1]
        im = ax.imshow(slope, cmap='YlOrRd', origin='lower', vmin=0, vmax=45)
        ax.set_title('Уклон (°)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Аккумуляция стока
    if flow_accumulation is not None:
        ax = axes[-1]
        # Логарифмическое масштабирование
        log_acc = np.log10(flow_accumulation + 1)
        im = ax.imshow(log_acc, cmap='Blues', origin='lower')
        ax.set_title('lg(Аккумуляция стока)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_simulation_results(
    results: pd.DataFrame,
    variables: List[str] = None,
    title: str = "Результаты моделирования",
    save_path: str = None
) -> plt.Figure:
    """
    Построение графиков результатов моделирования.

    Args:
        results: DataFrame с результатами
        variables: Список переменных для отображения
        title: Заголовок
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    if variables is None:
        variables = ['precipitation_mm', 'discharge_m3s', 'surface_runoff_mm']

    variables = [v for v in variables if v in results.columns]
    n_vars = len(variables)

    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    time = results['timestamp'] if 'timestamp' in results.columns else results.index

    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))

    labels_ru = {
        'precipitation_mm': 'Осадки (мм)',
        'discharge_m3s': 'Расход (м³/с)',
        'surface_runoff_mm': 'Поверхностный сток (мм)',
        'snowmelt_mm': 'Таяние снега (мм)',
        'temperature_c': 'Температура (°C)',
        'water_input_mm': 'Приток воды (мм)'
    }

    for i, (var, ax) in enumerate(zip(variables, axes)):
        ax.plot(time, results[var], color=colors[i], linewidth=1.5)
        ax.fill_between(time, 0, results[var], color=colors[i], alpha=0.3)
        ax.set_ylabel(labels_ru.get(var, var))
        ax.grid(True, alpha=0.3)

        # Статистика
        mean_val = results[var].mean()
        max_val = results[var].max()
        ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.7,
                  label=f'Среднее: {mean_val:.1f}')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Время')
    axes[0].set_title(title)

    # Форматирование дат
    if isinstance(time.iloc[0] if hasattr(time, 'iloc') else time[0], (datetime, pd.Timestamp)):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cn_sensitivity(
    cn_values: np.ndarray,
    runoff_values: np.ndarray,
    precipitation: float,
    save_path: str = None
) -> plt.Figure:
    """
    График чувствительности модели к параметру CN.

    Args:
        cn_values: Значения CN
        runoff_values: Соответствующие значения стока
        precipitation: Осадки
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cn_values, runoff_values, 'b-', linewidth=2, label='Сток')
    ax.fill_between(cn_values, 0, runoff_values, alpha=0.3)

    # Коэффициент стока
    ax2 = ax.twinx()
    runoff_coeff = runoff_values / precipitation
    ax2.plot(cn_values, runoff_coeff * 100, 'r--', linewidth=2, label='Коэфф. стока')
    ax2.set_ylabel('Коэффициент стока (%)', color='r')

    ax.set_xlabel('Номер кривой стока (CN)')
    ax.set_ylabel('Сток (мм)', color='b')
    ax.set_title(f'Анализ чувствительности (P = {precipitation} мм)')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_flood_events(
    events: List,
    title: str = "Паводковые события",
    save_path: str = None
) -> plt.Figure:
    """
    Визуализация паводковых событий.

    Args:
        events: Список объектов FloodEvent
        title: Заголовок
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    if not events:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Нет паводковых событий', ha='center', va='center',
               fontsize=14)
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Пиковые расходы
    ax = axes[0, 0]
    peak_q = [e.peak_discharge_m3s for e in events]
    x = range(len(events))
    colors = ['green' if e.risk_level.value == 'low' else
              'yellow' if e.risk_level.value == 'moderate' else
              'orange' if e.risk_level.value == 'high' else 'red'
              for e in events]
    ax.bar(x, peak_q, color=colors, edgecolor='black')
    ax.set_xlabel('Событие')
    ax.set_ylabel('Пиковый расход (м³/с)')
    ax.set_title('Пиковые расходы по событиям')

    # 2. Продолжительность
    ax = axes[0, 1]
    durations = [e.duration_hours for e in events]
    ax.bar(x, durations, color='steelblue', edgecolor='black')
    ax.set_xlabel('Событие')
    ax.set_ylabel('Продолжительность (часы)')
    ax.set_title('Продолжительность событий')

    # 3. Осадки vs Сток
    ax = axes[1, 0]
    precip = [e.total_precipitation_mm for e in events]
    ax.scatter(precip, peak_q, s=100, c=colors, edgecolor='black')
    ax.set_xlabel('Общие осадки (мм)')
    ax.set_ylabel('Пиковый расход (м³/с)')
    ax.set_title('Связь осадков и стока')

    # Линия тренда
    if len(precip) > 2:
        z = np.polyfit(precip, peak_q, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(precip), max(precip), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7)

    # 4. Распределение по уровням риска
    ax = axes[1, 1]
    risk_counts = {}
    for e in events:
        risk = e.risk_level.value
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    risk_order = ['low', 'moderate', 'high', 'critical']
    risk_labels = ['Низкий', 'Умеренный', 'Высокий', 'Критический']
    risk_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    values = [risk_counts.get(r, 0) for r in risk_order]
    ax.pie(values, labels=risk_labels, colors=risk_colors, autopct='%1.0f%%',
           startangle=90)
    ax.set_title('Распределение по уровням риска')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_dashboard(
    simulation_results: pd.DataFrame,
    events: List = None,
    terrain_data: Dict = None,
    region: str = "chui",
    save_path: str = None
) -> plt.Figure:
    """
    Создание информационной панели с основными результатами.

    Args:
        simulation_results: Результаты моделирования
        events: Паводковые события
        terrain_data: Данные о рельефе
        region: Код региона
        save_path: Путь для сохранения

    Returns:
        Объект Figure
    """
    setup_plot_style()

    fig = plt.figure(figsize=(16, 12))

    # Сетка для размещения графиков
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Гидрограф (большой график сверху)
    ax1 = fig.add_subplot(gs[0, :])
    time = simulation_results.get('timestamp', simulation_results.index)
    discharge = simulation_results['discharge_m3s']
    precip = simulation_results.get('precipitation_mm', None)

    ax1.plot(time, discharge, 'b-', linewidth=1.5, label='Расход')
    ax1.fill_between(time, 0, discharge, alpha=0.3)
    ax1.set_ylabel('Расход (м³/с)')
    ax1.set_title(f'Гидрограф стока - {REGIONS[region].name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Осадки
    ax2 = fig.add_subplot(gs[1, 0])
    if precip is not None:
        ax2.bar(time, precip, color='steelblue', alpha=0.7)
        ax2.set_ylabel('Осадки (мм)')
        ax2.set_title('Осадки')
    ax2.grid(True, alpha=0.3)

    # 3. Температура (если есть)
    ax3 = fig.add_subplot(gs[1, 1])
    if 'temperature_c' in simulation_results.columns:
        temp = simulation_results['temperature_c']
        ax3.plot(time, temp, 'r-', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Температура (°C)')
        ax3.set_title('Температура')
        ax3.fill_between(time, 0, temp, where=temp > 0, color='red', alpha=0.2)
        ax3.fill_between(time, 0, temp, where=temp < 0, color='blue', alpha=0.2)
    ax3.grid(True, alpha=0.3)

    # 4. Статистика
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    stats_text = f"""
    СТАТИСТИКА МОДЕЛИРОВАНИЯ
    ________________________

    Период: {len(simulation_results)} дней

    Осадки:
      Сумма: {precip.sum():.1f} мм
      Максимум: {precip.max():.1f} мм/день
      Среднее: {precip.mean():.1f} мм/день

    Расход:
      Максимум: {discharge.max():.1f} м³/с
      Среднее: {discharge.mean():.1f} м³/с

    Паводки: {len(events) if events else 0}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. Распределение уровней риска
    ax5 = fig.add_subplot(gs[2, 0])
    if 'risk_level' in simulation_results.columns:
        risk_counts = simulation_results['risk_level'].value_counts()
        colors = {'low': '#2ecc71', 'moderate': '#f1c40f',
                 'high': '#e67e22', 'critical': '#e74c3c'}
        labels = {'low': 'Низкий', 'moderate': 'Умеренный',
                 'high': 'Высокий', 'critical': 'Критический'}

        ax5.bar([labels.get(k, k) for k in risk_counts.index],
               risk_counts.values,
               color=[colors.get(k, 'gray') for k in risk_counts.index])
        ax5.set_title('Уровни риска')
        ax5.set_ylabel('Число дней')

    # 6. Снегозапас (если есть)
    ax6 = fig.add_subplot(gs[2, 1])
    if 'snowmelt_mm' in simulation_results.columns:
        ax6.plot(time, simulation_results['snowmelt_mm'], 'c-', linewidth=1.5)
        ax6.fill_between(time, 0, simulation_results['snowmelt_mm'], alpha=0.3)
        ax6.set_title('Таяние снега')
        ax6.set_ylabel('мм/день')
    ax6.grid(True, alpha=0.3)

    # 7. Информация о регионе
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    region_config = REGIONS[region]
    region_text = f"""
    РЕГИОН: {region_config.name}

    Площадь: {region_config.area_km2:,} км²
    Ср. высота: {region_config.avg_elevation} м

    Основные реки:
    {chr(10).join(['  • ' + r for r in region_config.main_rivers])}
    """
    ax7.text(0.1, 0.9, region_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Информационная панель: Моделирование паводков', fontsize=16, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_report_figures(
    results: pd.DataFrame,
    events: List = None,
    output_dir: str = "results"
) -> List[str]:
    """
    Создание набора рисунков для отчета/диплома.

    Args:
        results: Результаты моделирования
        events: Паводковые события
        output_dir: Директория для сохранения

    Returns:
        Список путей к созданным файлам
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # 1. Гидрограф
    fig = plot_hydrograph(
        results['timestamp'],
        results['discharge_m3s'],
        results.get('precipitation_mm'),
        title='Гидрограф стока и осадки'
    )
    path = os.path.join(output_dir, 'hydrograph.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)

    # 2. Результаты моделирования
    fig = plot_simulation_results(results)
    path = os.path.join(output_dir, 'simulation_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)

    # 3. Паводковые события
    if events:
        fig = plot_flood_events(events)
        path = os.path.join(output_dir, 'flood_events.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved_files.append(path)
        plt.close(fig)

    # 4. Информационная панель
    fig = create_dashboard(results, events)
    path = os.path.join(output_dir, 'dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    saved_files.append(path)
    plt.close(fig)

    return saved_files
