"""Simulador: estimativa de aluguel para uma casa hipotética em um ponto do mapa."""
from dataclasses import dataclass, field

import joblib
import json
import numpy as np
import pandas as pd

from pipeline import config, features

MIN_ADS_PER_BAIRRO = 8
FAR_FROM_DATA_KM = 3.0


@dataclass
class Models:
    regressor: object
    classifier: object
    metadata: dict
    area_range: tuple
    density: pd.DataFrame = field(repr=False)


@dataclass
class Estimate:
    value: float
    low: float
    high: float
    band: str
    band_range: tuple
    bairro: str
    warnings: list


def load_models(data: pd.DataFrame):
    metadata = json.loads((config.MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    density = data.groupby("bairro_geo").agg(
        densidade=("densidade_anuncios_km2", "first"), anuncios=("aluguel", "size")
    )
    return Models(
        regressor=joblib.load(config.MODELS_DIR / "regressor.joblib"),
        classifier=joblib.load(config.MODELS_DIR / "classificador.joblib"),
        metadata=metadata,
        area_range=(float(data["area"].quantile(0.01)), float(data["area"].quantile(0.99))),
        density=density,
    )


def km_between(lat1, lon1, lat2, lon2):
    dy = (np.asarray(lat1) - lat2) * features.KM_PER_DEGREE
    dx = (np.asarray(lon1) - lon2) * features.KM_PER_DEGREE * np.cos(np.radians(lat2))
    return np.hypot(dx, dy)


def band_range(metadata, band_index):
    cuts = metadata["faixas_preco"]["cortes_producao"]
    lows, highs = [0.0] + cuts, cuts + [float("inf")]
    return lows[band_index], highs[band_index]


def estimate(models: Models, bairros, data: pd.DataFrame, area, quartos, banheiros, garagem, latitude, longitude):
    """Estimativa pontual, intervalo e faixa; None se o ponto está fora de Curitiba."""
    frame, bairro = features.simulation_frame(
        area, quartos, banheiros, garagem, latitude, longitude, bairros, models.density
    )
    if not isinstance(bairro, str):
        return None

    row = frame[features.MODEL_FEATURES]
    log_value = float(models.regressor.predict(row)[0])
    half = models.metadata["intervalo"]["meia_largura_log"]
    band_index = int(models.classifier.predict(row)[0])

    warnings = []
    count = int(models.density["anuncios"].get(bairro, 0))
    if count < MIN_ADS_PER_BAIRRO:
        warnings.append(f"{bairro} tem só {count} anúncio(s) na base: a localização pesa pouco e a estimativa é menos confiável.")
    low, high = models.area_range
    if not low <= area <= high:
        warnings.append(f"Área fora da faixa bem coberta pela base ({low:.0f}–{high:.0f} m²): o modelo está extrapolando.")
    nearest = km_between(data["latitude"].to_numpy(), data["longitude"].to_numpy(), latitude, longitude).min()
    if nearest > FAR_FROM_DATA_KM:
        warnings.append(f"O anúncio mais próximo está a {nearest:.1f} km: região sem dados.")

    return Estimate(
        value=float(np.exp(log_value)),
        low=float(np.exp(log_value - half)),
        high=float(np.exp(log_value + half)),
        band=models.metadata["faixas_preco"]["rotulos"][band_index],
        band_range=band_range(models.metadata, band_index),
        bairro=bairro,
        warnings=warnings,
    )


def sensitivity(models: Models, bairros, data, area, quartos, banheiros, garagem, latitude, longitude):
    """Quanto o aluguel estimado muda ao alterar um fator por vez (explicação local)."""
    def price(**changes):
        args = dict(area=area, quartos=quartos, banheiros=banheiros, garagem=garagem, latitude=latitude, longitude=longitude)
        args.update(changes)
        result = estimate(models, bairros, data, **args)
        return None if result is None else result.value

    base = price()
    scenarios = {
        "Área +20%": dict(area=area * 1.2),
        "Área −20%": dict(area=area * 0.8),
        f"+1 quarto ({quartos + 1})": dict(quartos=quartos + 1),
        f"+1 banheiro ({banheiros + 1})": dict(banheiros=banheiros + 1),
        ("Sem garagem" if garagem else "Com garagem"): dict(garagem=not garagem),
        "Mais 3 km do centro": dict(latitude=latitude, longitude=longitude + 3 / (features.KM_PER_DEGREE * np.cos(np.radians(features.CENTRO[0])))),
    }
    rows = []
    for label, changes in scenarios.items():
        value = price(**changes)
        if value is not None and base:
            rows.append({"Cenário": label, "Aluguel estimado": value, "Variação (R$)": value - base, "Variação (%)": (value / base - 1) * 100})
    return pd.DataFrame(rows)


def similar_listings(data: pd.DataFrame, area, quartos, banheiros, latitude, longitude, k=5):
    """Anúncios mais parecidos: perto, de tamanho e número de cômodos próximos."""
    km = km_between(data["latitude"].to_numpy(), data["longitude"].to_numpy(), latitude, longitude)
    area_gap = np.abs(np.log(data["area"].fillna(data["area"].median()) / area))
    rooms_gap = np.abs(data["quartos"].fillna(quartos) - quartos) + 0.5 * np.abs(data["banheiros"].fillna(banheiros) - banheiros)
    # Pesos: 1 km ≈ 25% de diferença de área ≈ 0,5 quarto
    score = km / 1.0 + area_gap / 0.25 + rooms_gap / 0.5
    best = data.assign(distancia_km=km, similaridade=score).nsmallest(k, "similaridade")
    return best[["bairro_geo", "endereco", "area", "quartos", "banheiros", "aluguel", "aluguel_estimado", "distancia_km", "fonte", "url"]]
