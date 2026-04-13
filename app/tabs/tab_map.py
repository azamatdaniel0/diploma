"""
Вкладка 3: Интерактивная карта регионов.
"""

import html

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from src.config import REGIONS
from app.sidebar import SidebarParams
from app.ui_helpers import get_risk_label


def render_tab_map(params: SidebarParams) -> None:
    """Рендеринг вкладки карты."""
    st.subheader(f"🗺️ Карта {REGIONS[params.selected_region].name}")

    map_key = f"map_{params.selected_region}"

    if 'current_map_region' not in st.session_state:
        st.session_state.current_map_region = params.selected_region

    region_changed = st.session_state.current_map_region != params.selected_region
    if region_changed:
        st.session_state.current_map_region = params.selected_region

    col_map1, col_map2 = st.columns([4, 1])
    with col_map2:
        refresh_clicked = st.button("🔄 Обновить карту", key="refresh_map")

    rebuild_map = (map_key not in st.session_state) or region_changed or refresh_clicked

    if rebuild_map:
        region_config = REGIONS[params.selected_region]
        center_lat = (region_config.lat_min + region_config.lat_max) / 2
        center_lon = (region_config.lon_min + region_config.lon_max) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='OpenStreetMap')

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Terrain', overlay=False, control=True
        ).add_to(m)
        folium.TileLayer('CartoDB positron', name='Light', attr='© CartoDB').add_to(m)

        bounds = [
            [region_config.lat_min, region_config.lon_min],
            [region_config.lat_min, region_config.lon_max],
            [region_config.lat_max, region_config.lon_max],
            [region_config.lat_max, region_config.lon_min],
            [region_config.lat_min, region_config.lon_min],
        ]
        folium.Polygon(
            locations=bounds, color='blue', weight=2,
            fill=True, fillColor='blue', fillOpacity=0.1,
            popup=f'Регион: {html.escape(region_config.name)}'
        ).add_to(m)

        for river in region_config.main_rivers:
            lat = np.random.uniform(region_config.lat_min + 0.2, region_config.lat_max - 0.2)
            lon = np.random.uniform(region_config.lon_min + 0.2, region_config.lon_max - 0.2)
            folium.Marker(
                location=[lat, lon],
                popup=f'Река: {html.escape(river)}',
                icon=folium.Icon(color='blue', icon='tint', prefix='fa')
            ).add_to(m)

        if st.session_state.get('simulation_results') is not None:
            results = st.session_state.simulation_results
            high_risk = results[results['risk_level'].isin(['high', 'critical'])]

            if len(high_risk) > 0:
                for idx, row in high_risk.iloc[::24].iterrows():
                    lat = np.random.uniform(region_config.lat_min + 0.1, region_config.lat_max - 0.1)
                    lon = np.random.uniform(region_config.lon_min + 0.1, region_config.lon_max - 0.1)
                    color = 'red' if row['risk_level'] == 'critical' else 'orange'
                    risk_label = html.escape(get_risk_label(row['risk_level']))
                    discharge_value = html.escape(f"{row['discharge_m3s']:.1f}")
                    folium.CircleMarker(
                        location=[lat, lon], radius=10, color=color,
                        fill=True, fillColor=color, fillOpacity=0.5,
                        popup=f"Риск: {risk_label}<br>Расход: {discharge_value} м³/с"
                    ).add_to(m)

        folium.LayerControl().add_to(m)
        st.session_state[map_key] = m
    else:
        m = st.session_state[map_key]

    # returned_objects=[] — отключает отправку данных взаимодействия обратно в Streamlit,
    # что предотвращает бесконечные rerun при каждом клике/зуме/перемещении карты.
    st_folium(m, width=None, height=500, key=f"folium_{map_key}", returned_objects=[])

    st.markdown("""
    **Легенда:**
    - 🔵 Реки региона
    - 🟠 Зоны высокого риска
    - 🔴 Зоны критического риска
    """)
