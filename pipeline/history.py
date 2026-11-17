"""Comparação temporal: resume cada coleta datada em data/historico/ com as mesmas regras de limpeza.

Só entram as fontes presentes em todas as coletas, para a variação refletir preço e não a
chegada de um portal novo.
"""
import re

import pandas as pd

from pipeline import config, geo, preprocess

HISTORY_DIR = config.DATA_DIR / "historico"
SUMMARY_CSV = config.DATA_DIR / "historico_resumo.csv"


def snapshot_date(path):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    return match.group(1) if match else None


def load_snapshots():
    snapshots = {}
    for path in sorted(HISTORY_DIR.glob("*.csv")):
        date = snapshot_date(path)
        if date:
            snapshots[date] = pd.read_csv(path, dtype={"cep": str})
    return snapshots


def summarize(snapshots: dict):
    if not snapshots:
        return pd.DataFrame()

    canonical = preprocess.canonical_bairros(geo.load_bairros())
    common = set.intersection(*(set(df["fonte"]) for df in snapshots.values()))

    rows = []
    for date, raw in snapshots.items():
        df = preprocess.standardize(raw[raw["fonte"].isin(common)], canonical)
        df = df[df["tipo"].eq("Casa") & df["cidade"].fillna("Curitiba").eq("Curitiba")]
        df = preprocess.drop_incomplete(df)
        df["preco_m2"] = df["aluguel"] / df["area"]
        low, high = preprocess.LIMITS["aluguel"]
        df = df[df["aluguel"].between(low, high)]
        ppm2_low, ppm2_high = preprocess.LIMITS["preco_m2"]
        df = df[df["preco_m2"].isna() | df["preco_m2"].between(ppm2_low, ppm2_high)]

        rows.append({
            "data_coleta": date,
            "fontes": ", ".join(sorted(common)),
            "anuncios": len(df),
            "aluguel_mediano": df["aluguel"].median(),
            "preco_m2_mediano": df["preco_m2"].median(),
            "area_mediana": df["area"].median(),
        })
    return pd.DataFrame(rows).round(1)


def run():
    summary = summarize(load_snapshots())
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False) if len(summary) else "Sem coletas em data/historico/")
    return summary


if __name__ == "__main__":
    run()
