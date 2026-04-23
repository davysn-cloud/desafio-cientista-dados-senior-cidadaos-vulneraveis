"""Geospatial visualization utilities."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap


# Rio de Janeiro center coordinates
RIO_CENTER = [-22.9068, -43.1729]


def create_heatmap(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    save_path: str = "results/figures/q2_heatmap_demanda.html",
) -> folium.Map:
    """Create a folium heatmap of chamado locations."""
    valid = df.dropna(subset=[lat_col, lon_col])
    coords = valid[[lat_col, lon_col]].values.tolist()

    m = folium.Map(location=RIO_CENTER, zoom_start=11, tiles="cartodbpositron")
    HeatMap(coords, radius=10, blur=15).add_to(m)
    m.save(save_path)
    return m


def create_choropleth(
    geo_df,
    value_col: str,
    title: str = "Chamados por Bairro",
    save_path: str = "results/figures/q2_mapa_coropletico.html",
) -> folium.Map:
    """Create a choropleth map from a GeoDataFrame."""
    m = folium.Map(location=RIO_CENTER, zoom_start=11, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=geo_df.to_json(),
        data=geo_df,
        columns=[geo_df.index.name or "nome", value_col],
        key_on="feature.properties.nome",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
    ).add_to(m)

    m.save(save_path)
    return m
