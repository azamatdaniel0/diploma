"""
Модуль создания интерактивных карт.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import html

from ..config import REGIONS

try:
    import folium
    from folium.plugins import HeatMap, TimestampedGeoJson
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


class FloodRiskMapper:
    """
    Создание интерактивных карт риска паводков.
    """

    # Цвета для уровней риска
    RISK_COLORS = {
        'low': '#2ecc71',
        'moderate': '#f1c40f',
        'high': '#e67e22',
        'critical': '#e74c3c'
    }

    def __init__(self, region: str = "chui"):
        """
        Инициализация.

        Args:
            region: Код региона
        """
        if not FOLIUM_AVAILABLE:
            raise ImportError("Библиотека folium не установлена. "
                            "Установите: pip install folium")

        self.region = region
        self.region_config = REGIONS.get(region)

        if self.region_config:
            bounds = self.region_config.bounds
            self.center = [(bounds[1] + bounds[3]) / 2,
                          (bounds[0] + bounds[2]) / 2]
        else:
            self.center = [41.5, 74.5]  # Центр Кыргызстана

    def create_base_map(
        self,
        zoom_start: int = 8,
        tiles: str = 'OpenStreetMap'
    ) -> 'folium.Map':
        """
        Создание базовой карты.

        Args:
            zoom_start: Начальный масштаб
            tiles: Тип подложки

        Returns:
            Объект folium.Map
        """
        m = folium.Map(
            location=self.center,
            zoom_start=zoom_start,
            tiles=tiles
        )

        # Добавление слоев
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)

        # Контроль слоев
        folium.LayerControl().add_to(m)

        return m

    def add_region_boundary(
        self,
        m: 'folium.Map',
        region: str = None
    ) -> 'folium.Map':
        """
        Добавление границ региона на карту.

        Args:
            m: Объект карты
            region: Код региона

        Returns:
            Обновленная карта
        """
        if region is None:
            region = self.region

        if region not in REGIONS:
            return m

        bounds = REGIONS[region].bounds
        name = REGIONS[region].name

        # Прямоугольник границ
        coordinates = [
            [bounds[1], bounds[0]],  # SW
            [bounds[1], bounds[2]],  # SE
            [bounds[3], bounds[2]],  # NE
            [bounds[3], bounds[0]],  # NW
            [bounds[1], bounds[0]]   # SW (замыкание)
        ]

        folium.Polygon(
            locations=coordinates,
            color='blue',
            weight=3,
            fill=False,
            popup=html.escape(name)
        ).add_to(m)

        return m

    def add_flood_risk_layer(
        self,
        m: 'folium.Map',
        risk_data: List[Dict],
        layer_name: str = "Риск паводка"
    ) -> 'folium.Map':
        """
        Добавление слоя риска паводков.

        Args:
            m: Объект карты
            risk_data: Список словарей с ключами lat, lon, risk_level
            layer_name: Название слоя

        Returns:
            Обновленная карта
        """
        feature_group = folium.FeatureGroup(name=layer_name)

        for point in risk_data:
            lat = point['lat']
            lon = point['lon']
            risk = point.get('risk_level', 'low')
            value = point.get('value', 0)

            color = self.RISK_COLORS.get(risk, 'gray')

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"Риск: {html.escape(str(risk))}<br>Значение: {value:.2f}"
            ).add_to(feature_group)

        feature_group.add_to(m)
        return m

    def add_precipitation_heatmap(
        self,
        m: 'folium.Map',
        precip_data: List[List],
        layer_name: str = "Осадки"
    ) -> 'folium.Map':
        """
        Добавление тепловой карты осадков.

        Args:
            m: Объект карты
            precip_data: Список [lat, lon, intensity]
            layer_name: Название слоя

        Returns:
            Обновленная карта
        """
        HeatMap(
            precip_data,
            name=layer_name,
            min_opacity=0.3,
            radius=25,
            blur=15,
            gradient={0.4: 'blue', 0.65: 'cyan', 0.8: 'lime', 1: 'red'}
        ).add_to(m)

        return m

    def add_river_network(
        self,
        m: 'folium.Map',
        rivers: List[Dict] = None
    ) -> 'folium.Map':
        """
        Добавление речной сети на карту.

        Args:
            m: Объект карты
            rivers: Список рек с координатами

        Returns:
            Обновленная карта
        """
        if rivers is None and self.region_config:
            # Используем названия рек из конфига (без координат)
            for river_name in self.region_config.main_rivers:
                # Добавляем маркер (в реальности нужны координаты)
                pass

        if rivers:
            river_group = folium.FeatureGroup(name="Реки")

            for river in rivers:
                name = river.get('name', 'Река')
                coords = river.get('coordinates', [])

                if coords:
                    folium.PolyLine(
                        locations=coords,
                        color='blue',
                        weight=3,
                        opacity=0.8,
                        popup=html.escape(name)
                    ).add_to(river_group)

            river_group.add_to(m)

        return m

    def add_flood_events_markers(
        self,
        m: 'folium.Map',
        events: List,
        layer_name: str = "Паводковые события"
    ) -> 'folium.Map':
        """
        Добавление маркеров паводковых событий.

        Args:
            m: Объект карты
            events: Список объектов FloodEvent
            layer_name: Название слоя

        Returns:
            Обновленная карта
        """
        event_group = folium.FeatureGroup(name=layer_name)

        for i, event in enumerate(events):
            # Получаем координаты центра региона
            if hasattr(event, 'region') and event.region in REGIONS:
                bounds = REGIONS[event.region].bounds
                lat = (bounds[1] + bounds[3]) / 2
                lon = (bounds[0] + bounds[2]) / 2
            else:
                lat, lon = self.center

            # Добавляем небольшое смещение для разных событий
            lat += np.random.uniform(-0.1, 0.1)
            lon += np.random.uniform(-0.1, 0.1)

            color = self.RISK_COLORS.get(event.risk_level.value, 'gray')

            popup_text = f"""
            <b>Паводковое событие</b><br>
            Начало: {html.escape(str(event.start_time))}<br>
            Пик: {html.escape(str(event.peak_time))}<br>
            Длительность: {event.duration_hours:.1f} ч<br>
            Пиковый расход: {event.peak_discharge_m3s:.1f} м³/с<br>
            Уровень риска: {html.escape(event.risk_level.value)}
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red' if event.risk_level.value == 'critical'
                               else 'orange' if event.risk_level.value == 'high'
                               else 'green')
            ).add_to(event_group)

        event_group.add_to(m)
        return m

    def create_flood_risk_map(
        self,
        risk_grid: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        save_path: str = None
    ) -> 'folium.Map':
        """
        Создание интерактивной карты риска паводков.

        Args:
            risk_grid: Сетка уровней риска
            lon_grid: Сетка долгот
            lat_grid: Сетка широт
            save_path: Путь для сохранения HTML

        Returns:
            Объект folium.Map
        """
        m = self.create_base_map()
        self.add_region_boundary(m)

        # Преобразование сетки в точки для тепловой карты
        heat_data = []
        risk_points = []

        # Создание сетки координат
        if lon_grid.ndim == 1:
            LON, LAT = np.meshgrid(lon_grid, lat_grid)
        else:
            LON, LAT = lon_grid, lat_grid

        for i in range(risk_grid.shape[0]):
            for j in range(risk_grid.shape[1]):
                lat = LAT[i, j]
                lon = LON[i, j]
                risk_value = risk_grid[i, j]

                if risk_value > 0:
                    heat_data.append([lat, lon, risk_value / 4])

                    if risk_value >= 2:  # Умеренный и выше
                        risk_level = ('critical' if risk_value >= 4 else
                                     'high' if risk_value >= 3 else
                                     'moderate' if risk_value >= 2 else 'low')
                        risk_points.append({
                            'lat': lat,
                            'lon': lon,
                            'risk_level': risk_level,
                            'value': risk_value
                        })

        # Добавление слоев
        self.add_precipitation_heatmap(m, heat_data, "Уровень риска")
        self.add_flood_risk_layer(m, risk_points, "Точки риска")

        # Легенда
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border-radius: 5px;
                    border: 2px solid gray;">
        <p><strong>Уровень риска</strong></p>
        <p><span style="color: #2ecc71;">●</span> Низкий</p>
        <p><span style="color: #f1c40f;">●</span> Умеренный</p>
        <p><span style="color: #e67e22;">●</span> Высокий</p>
        <p><span style="color: #e74c3c;">●</span> Критический</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        if save_path:
            m.save(save_path)

        return m

    def create_kyrgyzstan_overview_map(
        self,
        save_path: str = None
    ) -> 'folium.Map':
        """
        Создание обзорной карты Кыргызстана со всеми регионами.

        Args:
            save_path: Путь для сохранения

        Returns:
            Объект folium.Map
        """
        # Центр Кыргызстана
        m = folium.Map(
            location=[41.5, 74.5],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        # Добавление всех регионов
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'cadetblue']

        for i, (region_code, config) in enumerate(REGIONS.items()):
            bounds = config.bounds
            color = colors[i % len(colors)]

            # Границы региона
            coordinates = [
                [bounds[1], bounds[0]],
                [bounds[1], bounds[2]],
                [bounds[3], bounds[2]],
                [bounds[3], bounds[0]],
                [bounds[1], bounds[0]]
            ]

            popup_text = f"""
            <b>{html.escape(config.name)}</b><br>
            Площадь: {config.area_km2:,} км²<br>
            Ср. высота: {config.avg_elevation} м<br>
            Реки: {html.escape(', '.join(config.main_rivers))}
            """

            folium.Polygon(
                locations=coordinates,
                color=color,
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.2,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)

            # Маркер в центре региона
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            folium.Marker(
                location=[center_lat, center_lon],
                popup=html.escape(config.name),
                icon=folium.Icon(color=color.replace('dark', '').replace('cadet', ''),
                               icon='info-sign')
            ).add_to(m)

        # Добавление контроля слоев
        folium.LayerControl().add_to(m)

        # Заголовок
        title_html = '''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 1000; background-color: white; padding: 10px;
                    border-radius: 5px; border: 2px solid gray;">
        <h3 style="margin: 0;">Регионы Кыргызстана для моделирования паводков</h3>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        if save_path:
            m.save(save_path)

        return m


def create_sample_maps(output_dir: str = "results"):
    """
    Создание примеров карт.

    Args:
        output_dir: Директория для сохранения
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    mapper = FloodRiskMapper(region="chui")

    # 1. Обзорная карта
    m = mapper.create_kyrgyzstan_overview_map()
    m.save(os.path.join(output_dir, 'kyrgyzstan_overview.html'))
    print(f"Сохранено: {output_dir}/kyrgyzstan_overview.html")

    # 2. Карта риска (с синтетическими данными)
    lon_grid = np.linspace(73.5, 76.5, 30)
    lat_grid = np.linspace(42.0, 43.5, 20)
    risk_grid = np.random.randint(0, 5, size=(20, 30)).astype(float)

    m = mapper.create_flood_risk_map(
        risk_grid, lon_grid, lat_grid,
        save_path=os.path.join(output_dir, 'flood_risk_map.html')
    )
    print(f"Сохранено: {output_dir}/flood_risk_map.html")


if __name__ == "__main__":
    if FOLIUM_AVAILABLE:
        create_sample_maps()
    else:
        print("Folium не установлен")
