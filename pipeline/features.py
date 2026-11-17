"""Features compartilhadas entre o treino e o simulador, para os dois calcularem igual."""
import numpy as np
import pandas as pd

from pipeline import geo

CENTRO = (-25.4297, -49.2711)  # Praça Tiradentes, marco zero de Curitiba
KM_PER_DEGREE = 111.32

NUMERIC_FEATURES = [
    "log_area", "quartos", "banheiros", "tem_garagem", "densidade_comodos",
    "dist_centro_km", "latitude", "longitude", "densidade_anuncios_km2",
]
CATEGORICAL_FEATURES = ["bairro_geo"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["area"]


def dist_centro_km(latitude, longitude):
    # Aproximação plana; em 30 km o erro é irrelevante para este uso
    dy = (np.asarray(latitude) - CENTRO[0]) * KM_PER_DEGREE
    dx = (np.asarray(longitude) - CENTRO[1]) * KM_PER_DEGREE * np.cos(np.radians(CENTRO[0]))
    return np.hypot(dx, dy).round(3)


def add_model_columns(df: pd.DataFrame):
    """Colunas que o modelo consome e que dependem só de outras colunas da própria linha."""
    df = df.copy()
    df["log_area"] = np.log(df["area"])
    return df


def bairro_stats(df: pd.DataFrame, bairros):
    """Contagem e densidade de anúncios por bairro, usadas nas features de localização."""
    area_km2 = bairros.set_index("bairro")["area_km2"]
    counts = df["bairro_geo"].value_counts()
    return pd.DataFrame({"anuncios": counts, "area_km2": area_km2}).dropna().assign(
        densidade=lambda t: (t["anuncios"] / t["area_km2"]).round(3)
    )


def simulation_frame(area, quartos, banheiros, garagem, latitude, longitude, bairros, stats):
    """Linha de entrada do modelo para um imóvel hipotético em um ponto do mapa."""
    point = pd.DataFrame({"latitude": [latitude], "longitude": [longitude]})
    bairro = geo.assign_bairro(point, bairros).iloc[0]
    density = stats["densidade"].get(bairro, np.nan)

    frame = pd.DataFrame({
        "area": [float(area)],
        "quartos": [float(quartos)],
        "banheiros": [float(banheiros)],
        "tem_garagem": [int(bool(garagem))],
        "latitude": [float(latitude)],
        "longitude": [float(longitude)],
        "bairro_geo": [bairro],
        "densidade_anuncios_km2": [density],
    })
    frame["densidade_comodos"] = (frame["quartos"] + frame["banheiros"]) / frame["area"]
    frame["dist_centro_km"] = dist_centro_km(frame["latitude"], frame["longitude"])
    return add_model_columns(frame), bairro
