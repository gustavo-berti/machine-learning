import numpy as np
import pandas as pd

from pipeline import features, preprocess
from pipeline.text import bairro_key


def raw_frame(**overrides):
    base = {
        "id": [1, 2], "fonte": ["chavesnamao", "quintoandar"], "url": ["https://x/com-garagem/id-1/", "https://q/2"],
        "tipo": ["Casa", "Casa"], "endereco": ["Rua A, 10", "não disponível"], "bairro": ["Batel ", "Alto da XV"],
        "cidade": ["Curitiba", "Curitiba"], "cep": ["00000-000", "81670390"], "latitude": [0.0, -25.4],
        "longitude": [0.0, -49.3], "area": [120.0, 5.0], "quartos": [0, 3], "banheiros": [2, 0],
        "vagas": [np.nan, 0], "aluguel": [4000, 3000], "condominio_iptu": [np.nan, 100], "custo_total": [np.nan, 3100],
        "mobiliado": [np.nan, False],
    }
    base.update(overrides)
    return pd.DataFrame(base)


CANONICAL = {bairro_key("Batel"): "Batel", "alto da xv": "Alto da Rua XV"}


def test_standardize_turns_sentinels_into_nan():
    df = preprocess.standardize(raw_frame(), CANONICAL)
    assert pd.isna(df.loc[0, "cep"])  # "00000-000"
    assert df.loc[1, "cep"] == "81670-390"
    assert pd.isna(df.loc[1, "endereco"])  # "não disponível"
    assert df[["latitude", "longitude"]].iloc[0].isna().all()  # coordenada 0
    assert pd.isna(df.loc[0, "quartos"]) and pd.isna(df.loc[1, "banheiros"])  # zero implausível
    assert pd.isna(df.loc[1, "area"])  # 5 m² não é uma casa


def test_standardize_unifies_bairro_names():
    df = preprocess.standardize(raw_frame(), CANONICAL)
    assert list(df["bairro"]) == ["Batel", "Alto da Rua XV"]


def test_garage_flag_from_count_or_url_slug():
    df = preprocess.standardize(raw_frame(), CANONICAL)
    df = preprocess.add_garage_flag(df)
    assert df.loc[0, "tem_garagem"] == 1  # chavesnamao: slug "com-garagem"
    assert df.loc[1, "tem_garagem"] == 0  # quintoandar: vagas = 0


def test_drop_incomplete_needs_rent_and_two_key_fields():
    df = preprocess.standardize(raw_frame(), CANONICAL)
    kept = preprocess.drop_incomplete(df)
    assert list(kept["id"]) == [1]  # a linha 2 só tem quartos; falta área e banheiros


def test_cross_source_duplicates_are_merged_keeping_the_richest_row():
    df = raw_frame(
        id=[1, 2], fonte=["chavesnamao", "baggio"], endereco=["Rua A, 10", "A, 10"],
        area=[120.0, 120.0], quartos=[3, 3], aluguel=[4000, 4005],
        latitude=[-25.4, np.nan], longitude=[-49.3, np.nan], cep=[np.nan, "81670-390"], vagas=[np.nan, 2],
    )
    merged, removed = preprocess.merge_cross_source_duplicates(df)
    assert removed == 1 and len(merged) == 1
    row = merged.iloc[0]
    assert row["fonte"] == "baggio"  # mais completa (tem CEP e vagas)
    assert row["latitude"] == -25.4  # lacuna preenchida pela outra fonte
    assert row["vagas"] == 2


def test_rows_without_address_are_never_merged():
    df = raw_frame(endereco=[None, None], area=[120.0, 120.0], quartos=[3, 3], aluguel=[4000, 4000])
    merged, removed = preprocess.merge_cross_source_duplicates(df)
    assert removed == 0 and len(merged) == 2


def test_flag_outliers_marks_domain_violations_but_keeps_rows():
    df = pd.DataFrame({
        "aluguel": [4000, 5000, 4500, 100], "area": [100, 110, 105, 100], "quartos": [3, 3, 3, 3],
        "banheiros": [2, 2, 2, 2], "vagas": [np.nan] * 4, "condominio_iptu": [np.nan] * 4,
    })
    df["preco_m2"] = df["aluguel"] / df["area"]
    flagged = preprocess.flag_outliers(df)
    assert len(flagged) == 4
    assert flagged["outlier"].tolist() == [False, False, False, True]
    assert "aluguel_dominio" in flagged.loc[3, "motivo_outlier"]


def test_dist_centro_is_zero_at_center_and_grows():
    here = features.dist_centro_km(features.CENTRO[0], features.CENTRO[1])
    far = features.dist_centro_km(features.CENTRO[0] + 0.1, features.CENTRO[1])
    assert here == 0 and 10 < far < 12
