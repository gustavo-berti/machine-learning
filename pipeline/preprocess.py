"""Pré-processamento da base bruta (passos 1 a 5 e 7 do claude.md).

Tudo aqui é limpeza *sem estado aprendido*: regras fixas de domínio, padronização e
derivação de colunas. Imputação, codificação e normalização dependem do conjunto de
treino e por isso ficam dentro do Pipeline do scikit-learn (pipeline/modeling.py),
ajustadas só depois da separação treino/teste (passo 8).
"""
import json
import re

import numpy as np
import pandas as pd

from pipeline import config, features, geo
from pipeline.text import BAIRRO_ALIASES, bairro_key

# Limites de domínio para casas de locação em Curitiba. Fora deles o valor é erro de
# digitação ou anúncio de outro tipo de imóvel, não mercado.
LIMITS = {
    "aluguel": (500, 60_000),
    "area": (20, 2_000),
    "quartos": (1, 10),
    "banheiros": (1, 10),
    "vagas": (0, 12),
    "preco_m2": (3, 150),
    "condominio_iptu": (0, 10_000),
}
IQR_FENCE = 3.0  # cerca "extrema" (Tukey usa 1,5; 3 pega só o que é claramente anômalo)
MIN_KEY_FIELDS = 2  # área, quartos e banheiros: abaixo disso o anúncio é incompleto demais

SENTINEL_ADDRESS = {"não disponível", "nao disponivel", "-", ""}


def inspect(df: pd.DataFrame):
    """Passo 1: dimensões, tipos, ausentes, duplicidades e estatísticas básicas."""
    numeric = df.select_dtypes("number")
    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "tipos": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "ausentes_pct": (df.isna().mean() * 100).round(1).to_dict(),
        "duplicados_fonte_id": int(df.duplicated(subset=["fonte", "id"]).sum()),
        "duplicados_linha_inteira": int(df.duplicated().sum()),
        "estatisticas": numeric.describe().round(2).to_dict(),
    }


def canonical_bairros(bairros_gdf):
    """chave normalizada -> nome canônico (o do polígono), mais os apelidos conhecidos."""
    mapping = {bairro_key(name): name for name in bairros_gdf["bairro"]}
    mapping.update({key: value for key, value in BAIRRO_ALIASES.items()})
    return mapping


def clean_address(value):
    if not isinstance(value, str) or value.lower().strip() in SENTINEL_ADDRESS:
        return None
    return value.strip()


def clean_cep(value):
    """'81670390' / '81.670-390' -> '81670-390'; CEP zerado ou incompleto vira nulo."""
    digits = re.sub(r"\D", "", value) if isinstance(value, str) else ""
    if len(digits) != 8 or digits == "00000000":
        return None
    return f"{digits[:5]}-{digits[5:]}"


def standardize(df: pd.DataFrame, canonical: dict):
    """Passo 2: nomes de bairro, CEP, sentinelas e tipos numéricos uniformes."""
    df = df.copy()

    for column in ["area", "quartos", "banheiros", "vagas", "aluguel", "condominio_iptu", "custo_total", "latitude", "longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["bairro_original"] = df["bairro"]
    df["bairro"] = df["bairro"].map(
        lambda name: canonical.get(bairro_key(name), str(name).strip().title()) if isinstance(name, str) else None
    )

    df["endereco"] = df["endereco"].map(clean_address)
    df["cep"] = df["cep"].map(clean_cep)

    # Coordenada 0 é sentinela de "sem localização", não um ponto no Golfo da Guiné
    no_coords = df["latitude"].eq(0) | df["longitude"].eq(0)
    df.loc[no_coords, ["latitude", "longitude"]] = np.nan

    # Casa sem quarto, sem banheiro ou com área ínfima é anúncio incompleto
    for column in ["quartos", "banheiros"]:
        df.loc[df[column] <= 0, column] = np.nan
    df.loc[df["area"] < LIMITS["area"][0], "area"] = np.nan

    df["mobiliado"] = df["mobiliado"].map({True: True, False: False, "True": True, "False": False})
    return df


def garage_from_url(url: str):
    """O chavesnamao não publica vagas, mas o slug do anúncio traz 'com-garagem'."""
    return isinstance(url, str) and "com-garagem" in url


def add_garage_flag(df: pd.DataFrame):
    """tem_garagem unificado: contagem de vagas quando a fonte tem; slug da URL no chavesnamao."""
    has_count = df["vagas"].notna()
    df["tem_garagem"] = np.where(has_count, df["vagas"] > 0, df["url"].map(garage_from_url))
    df["tem_garagem"] = df["tem_garagem"].astype(int)
    return df


def drop_incomplete(df: pd.DataFrame):
    """Remove anúncios sem aluguel ou com quase nenhuma característica física."""
    key_fields = df[["area", "quartos", "banheiros"]].notna().sum(axis=1)
    return df[df["aluguel"].notna() & (key_fields >= MIN_KEY_FIELDS)].copy()


def normalized_address(df: pd.DataFrame):
    street = df["endereco"].fillna("").map(bairro_key).fillna("")
    number = df["endereco"].fillna("").str.extract(r"(\d+)\s*$")[0].fillna("")
    street = street.str.replace(r"[\s,]*\d+\s*$", "", regex=True)
    street = street.str.replace(r"^(r|rua|av|avenida|al|alameda|tv|travessa)\b\.?\s*", "", regex=True)
    return street + "|" + number


def duplicate_keys(df: pd.DataFrame):
    """Chave de identidade do imóvel: logradouro+número, área, quartos e aluguel (arredondado)."""
    parts = pd.DataFrame({
        "end": normalized_address(df),
        "area": df["area"].round(0),
        "quartos": df["quartos"],
        "aluguel": df["aluguel"].round(-1),
    })
    keys = pd.Series(["|".join(map(str, row)) for row in parts.itertuples(index=False)], index=df.index)
    return keys.where(parts["end"].str.len().gt(1))  # sem endereço não dá para afirmar que é o mesmo imóvel


def merge_cross_source_duplicates(df: pd.DataFrame):
    """O mesmo imóvel aparece em mais de um portal (a baggio republica no chavesnamao).

    Em vez de descartar uma das linhas, funde: fica a mais completa e as lacunas dela são
    preenchidas pelas outras (a baggio traz CEP, vagas e condomínio; o chavesnamao, coordenadas).
    """
    keyed = df.assign(_chave=duplicate_keys(df), _riqueza=df.notna().sum(axis=1))
    with_key = keyed[keyed["_chave"].notna()].sort_values("_riqueza", ascending=False, kind="stable")
    without_key = keyed[keyed["_chave"].isna()]

    merged = with_key.groupby("_chave", sort=False).first().reset_index(drop=True)  # 1ª não nula por coluna
    result = pd.concat([merged, without_key], ignore_index=True)
    removed = len(with_key) - len(merged)
    return result.drop(columns=["_chave", "_riqueza"]), removed


def locate(df: pd.DataFrame, bairros, geocode: bool = True):
    """Garante coordenada para todo anúncio e registra de onde ela veio (geo_origem).

    `geocode=False` não consulta a rede, mas ainda aproveita o cache de consultas anteriores.
    """
    df = df.copy()
    inside = [
        geo.inside_curitiba(lat, lon) if pd.notna(lat) and pd.notna(lon) else False
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    df["geo_origem"] = np.where(inside, "fonte", None)
    df.loc[~np.array(inside), ["latitude", "longitude"]] = np.nan  # fora de Curitiba = erro de geocodificação

    missing = df["latitude"].isna()
    if missing.any():
        queries = {
            index: f"{row.endereco}, {row.bairro}, Curitiba, Paraná, Brasil"
            for index, row in df[missing & df["endereco"].notna()].iterrows()
        }
        cache = geo.geocode_addresses(list(queries.values()), offline=not geocode)
        for index, query in queries.items():
            hit = cache.get(query)
            if hit and geo.inside_curitiba(*hit):
                df.loc[index, ["latitude", "longitude", "geo_origem"]] = [hit[0], hit[1], "nominatim"]

    # Último recurso: centroide do bairro declarado (impreciso, mas marcado como tal)
    centroids = {
        row.bairro: row.geometry.representative_point() for row in bairros.itertuples()
    }
    still_missing = df["latitude"].isna() & df["bairro"].isin(centroids)
    for index in df.index[still_missing]:
        point = centroids[df.at[index, "bairro"]]
        df.loc[index, ["latitude", "longitude", "geo_origem"]] = [point.y, point.x, "centroide_bairro"]

    return df


def derive_features(df: pd.DataFrame, bairros):
    """Passo 7: preço/m², custo mensal total, densidades e indicadores de localização."""
    df = df.copy()
    df["bairro_geo"] = geo.assign_bairro(df, bairros)
    df["bairro_geo"] = df["bairro_geo"].fillna(df["bairro"])

    df["preco_m2"] = df["aluguel"] / df["area"]
    df["custo_mensal_total"] = df["aluguel"] + df["condominio_iptu"]
    df["comodos"] = df["quartos"] + df["banheiros"]
    df["densidade_comodos"] = df["comodos"] / df["area"]

    df["dist_centro_km"] = features.dist_centro_km(df["latitude"], df["longitude"])

    stats = features.bairro_stats(df, bairros)
    df["anuncios_bairro"] = df["bairro_geo"].map(stats["anuncios"])
    df["densidade_anuncios_km2"] = df["bairro_geo"].map(stats["densidade"])
    return df


def flag_outliers(df: pd.DataFrame):
    """Passo 4: critério de domínio + cerca IQR extrema sobre log(preço/m²).

    Os anômalos ficam marcados (`outlier`, `motivo_outlier`) em vez de apagados, para a
    EDA poder mostrar o antes e o depois e a modelagem descartar só o que for erro.
    """
    df = df.copy()
    reasons = pd.Series("", index=df.index)

    def flag(mask, reason):
        nonlocal reasons
        reasons = reasons.where(~mask, reasons + reason + ";")

    for column in ["aluguel", "area", "quartos", "banheiros", "vagas", "condominio_iptu", "preco_m2"]:
        low, high = LIMITS[column]
        flag(df[column].notna() & ((df[column] < low) | (df[column] > high)), f"{column}_dominio")

    log_ppm2 = np.log(df["preco_m2"])
    q1, q3 = log_ppm2.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - IQR_FENCE * iqr, q3 + IQR_FENCE * iqr
    flag((log_ppm2 < lower) | (log_ppm2 > upper), "preco_m2_iqr")

    df["motivo_outlier"] = reasons.str.rstrip(";")
    df["outlier"] = df["motivo_outlier"].ne("")
    df.attrs["fences_preco_m2"] = (float(np.exp(lower)), float(np.exp(upper)))
    return df


def clean(raw: pd.DataFrame, geocode: bool = True):
    """Executa a limpeza completa e devolve (DataFrame limpo, relatório)."""
    bairros = geo.load_bairros()
    report = {"inspecao_inicial": inspect(raw)}

    df = standardize(raw, canonical_bairros(bairros))
    # A baggio já filtra por cidade no scraper; as outras fontes não garantem isso
    df = df[df["cidade"].fillna("Curitiba").str.strip().str.lower().eq("curitiba")].copy()
    df = df[df["tipo"].eq("Casa")].copy()
    report["apos_filtro_cidade_tipo"] = len(df)

    df = df.drop_duplicates(subset=["fonte", "id"])
    df = drop_incomplete(df)
    report["apos_incompletos"] = len(df)

    df, merged = merge_cross_source_duplicates(df)
    report["duplicados_entre_fontes_fundidos"] = merged

    df = add_garage_flag(df)
    df = locate(df, bairros, geocode=geocode)
    df = derive_features(df, bairros)
    df = flag_outliers(df)

    report["geo_origem"] = df["geo_origem"].value_counts(dropna=False).to_dict()
    report["outliers"] = int(df["outlier"].sum())
    report["motivos_outlier"] = (
        df.loc[df["outlier"], "motivo_outlier"].str.split(";").explode().value_counts().to_dict()
    )
    report["cerca_preco_m2"] = df.attrs["fences_preco_m2"]
    report["linhas_finais"] = len(df)
    report["linhas_para_modelagem"] = int((~df["outlier"]).sum())
    report["ausentes_pct_final"] = (df.isna().mean() * 100).round(1).to_dict()
    return df.reset_index(drop=True), report


def run(geocode: bool = True):
    raw = pd.read_csv(config.RAW_CSV, dtype={"cep": str})
    df, report = clean(raw, geocode=geocode)

    df.to_csv(config.CLEAN_CSV, index=False)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "qualidade_dados.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"{len(raw)} linhas brutas -> {len(df)} limpas ({report['linhas_para_modelagem']} sem outlier)")
    print(f"Origem das coordenadas: {report['geo_origem']}")
    return df, report


if __name__ == "__main__":
    run()
