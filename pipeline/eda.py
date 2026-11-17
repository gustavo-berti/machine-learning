"""Análise exploratória (claude.md, seção 3): gera figuras e reports/EDA.md."""
import json

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

from pipeline import config, geo
from pipeline.text import bairro_key

MIN_PER_BAIRRO = 8  # abaixo disso a mediana do bairro é pouco confiável
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
COLOR = "#2f6f8f"
ACCENT = "#d1495b"


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name)
    plt.close(fig)


def fmt_brl(value):
    return f"R$ {value:,.0f}".replace(",", ".")


def distributions(df, clean):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(df["aluguel"].clip(upper=df["aluguel"].quantile(0.995)), bins=50, color=COLOR)
    axes[0, 0].set(title="Aluguel (R$/mês)", xlabel="R$")
    axes[0, 1].hist(np.log10(df["aluguel"]), bins=50, color=COLOR)
    axes[0, 1].set(title="Aluguel em escala log10", xlabel="log10(R$)")
    axes[1, 0].hist(df["preco_m2"].dropna().clip(upper=df["preco_m2"].quantile(0.995)), bins=50, color=ACCENT)
    axes[1, 0].set(title="Aluguel por m² (R$/m²)", xlabel="R$/m²")
    box = axes[1, 1].boxplot([clean["preco_m2"].dropna(), df["preco_m2"].dropna()], tick_labels=["sem outliers", "com outliers"], vert=False, patch_artist=True)
    for patch, color in zip(box["boxes"], [COLOR, ACCENT]):
        patch.set_facecolor(color)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(title="Preço/m² antes e depois do tratamento (log)", xlabel="R$/m²")
    fig.suptitle("Distribuição dos preços", fontweight="bold")
    save(fig, "01_distribuicoes.png")


def boxplots_price(clean):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].boxplot(clean["aluguel"], vert=False, patch_artist=True, boxprops={"facecolor": COLOR})
    axes[0].set(title="Boxplot do aluguel (sem outliers)", xlabel="R$/mês", yticks=[])
    axes[1].boxplot(clean["preco_m2"].dropna(), vert=False, patch_artist=True, boxprops={"facecolor": ACCENT})
    axes[1].set(title="Boxplot do preço/m² (sem outliers)", xlabel="R$/m²", yticks=[])
    save(fig, "02_boxplots_preco.png")


def by_bairro(clean):
    stats = (
        clean.groupby("bairro_geo")
        .agg(anuncios=("aluguel", "size"), aluguel_mediano=("aluguel", "median"), preco_m2_mediano=("preco_m2", "median"),
             aluguel_cv=("aluguel", lambda s: s.std() / s.mean()))
        .round(2)
    )
    reliable = stats[stats["anuncios"] >= MIN_PER_BAIRRO]

    top = reliable.sort_values("aluguel_mediano", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    shown = pd.concat([top.head(10), top.tail(10)])
    axes[0].barh(shown.index[::-1], shown["aluguel_mediano"][::-1], color=COLOR)
    axes[0].set(title=f"Aluguel mediano — 10 maiores e 10 menores\n(bairros com ≥ {MIN_PER_BAIRRO} anúncios)", xlabel="R$/mês")
    ppm2 = reliable.sort_values("preco_m2_mediano", ascending=False)
    shown = pd.concat([ppm2.head(10), ppm2.tail(10)])
    axes[1].barh(shown.index[::-1], shown["preco_m2_mediano"][::-1], color=ACCENT)
    axes[1].set(title="Preço/m² mediano — 10 maiores e 10 menores", xlabel="R$/m²")
    save(fig, "03_bairros_preco.png")

    counts = stats["anuncios"].sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index[::-1], counts[::-1], color=COLOR)
    ax.axvline(MIN_PER_BAIRRO, color=ACCENT, ls="--", label=f"mínimo confiável ({MIN_PER_BAIRRO})")
    ax.set(title="Anúncios por bairro (25 maiores)", xlabel="anúncios")
    ax.legend()
    save(fig, "04_contagem_bairros.png")

    variability = reliable["aluguel_cv"].sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(variability.index[::-1], variability[::-1], color=ACCENT)
    ax.set(title="Bairros de maior variabilidade de preço\n(coeficiente de variação do aluguel)", xlabel="desvio padrão / média")
    save(fig, "05_variabilidade_bairros.png")

    top_bairros = clean["bairro_geo"].value_counts().head(12).index
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sample = clean[clean["bairro_geo"].isin(top_bairros)]
    order = sample.groupby("bairro_geo")["aluguel"].median().sort_values().index
    ax.boxplot([sample.loc[sample["bairro_geo"] == b, "aluguel"] for b in order], tick_labels=order, patch_artist=True, boxprops={"facecolor": COLOR})
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="Aluguel nos 12 bairros com mais anúncios (escala log)", ylabel="R$/mês")
    save(fig, "06_boxplot_bairros.png")
    return stats


def correlations(clean):
    columns = ["aluguel", "preco_m2", "area", "quartos", "banheiros", "tem_garagem", "dist_centro_km", "densidade_comodos", "densidade_anuncios_km2"]
    corr = clean[columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    ax.grid(False)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image)
    ax.set_title("Correlação de Spearman")
    save(fig, "07_correlacao.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(clean["area"], clean["aluguel"], s=10, alpha=0.4, color=COLOR)
    axes[0].set(xscale="log", yscale="log", title="Área × aluguel (escala log-log)", xlabel="m²", ylabel="R$/mês")
    axes[1].scatter(clean["dist_centro_km"], clean["preco_m2"], s=10, alpha=0.4, color=ACCENT)
    axes[1].set(title="Distância ao centro × preço/m²", xlabel="km até a Praça Tiradentes", ylabel="R$/m²")
    save(fig, "08_dispersao.png")
    return corr


def spatial_inspection(df, bairros):
    """Aponta pontos possivelmente geocodificados errado: o ponto cai fora do bairro declarado."""
    declared = df["bairro"].map(bairro_key)
    located = df["bairro_geo"].map(bairro_key)
    mismatch = df[declared.ne(located) & df["geo_origem"].ne("centroide_bairro")].copy()
    mismatch["distancia_borda_km"] = distance_to_declared(mismatch, bairros)
    suspects = mismatch[mismatch["distancia_borda_km"].fillna(99) > 1.0]

    fig, ax = plt.subplots(figsize=(9, 9))
    bairros.boundary.plot(ax=ax, color="#999", linewidth=0.5)
    for origin, color, label in [("fonte", COLOR, "coordenada da fonte"), ("nominatim", "#3a9d5d", "geocodificado (Nominatim)"), ("centroide_bairro", "#e0a526", "centroide do bairro")]:
        part = df[df["geo_origem"] == origin]
        ax.scatter(part["longitude"], part["latitude"], s=9, alpha=0.6, color=color, label=f"{label} ({len(part)})")
    ax.scatter(suspects["longitude"], suspects["latitude"], s=60, facecolors="none", edgecolors=ACCENT, label=f"suspeitos ({len(suspects)})")
    ax.set(title="Inspeção espacial dos anúncios", xlabel="longitude", ylabel="latitude", aspect=1 / np.cos(np.radians(25.43)))
    ax.legend(loc="lower right")
    save(fig, "09_inspecao_espacial.png")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["fonte", "id", "url", "endereco", "bairro", "bairro_geo", "latitude", "longitude", "geo_origem", "distancia_borda_km"]
    suspects[columns].to_csv(config.REPORTS_DIR / "pontos_suspeitos.csv", index=False)
    return len(mismatch), len(suspects)


def distance_to_declared(rows, bairros):
    """Distância (km) do ponto até a borda do bairro que o anúncio declara; NaN se o bairro não tem polígono."""
    shapes = bairros.assign(chave=bairros["bairro"].map(bairro_key)).set_index("chave").to_crs("EPSG:31982")["geometry"]
    points = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in zip(rows["latitude"], rows["longitude"])], index=rows.index, crs="EPSG:4326"
    ).to_crs("EPSG:31982")
    distances = []
    for index, point in points.items():
        shape = shapes.get(bairro_key(rows.at[index, "bairro"]))
        distances.append(np.nan if shape is None else round(point.distance(shape) / 1000, 2))
    return pd.Series(distances, index=rows.index)


def source_comparison(clean):
    table = clean.groupby("fonte").agg(
        anuncios=("aluguel", "size"), aluguel_mediano=("aluguel", "median"),
        preco_m2_mediano=("preco_m2", "median"), area_mediana=("area", "median"),
    ).round(1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for source, part in clean.groupby("fonte"):
        ax.hist(np.log10(part["preco_m2"].dropna()), bins=30, alpha=0.55, label=f"{source} (n={len(part)})", density=True)
    ax.set(title="Preço/m² por fonte (log10)", xlabel="log10(R$/m²)", ylabel="densidade")
    ax.legend()
    save(fig, "10_fontes.png")
    return table


def run():
    df = pd.read_csv(config.CLEAN_CSV, dtype={"cep": str})
    clean = df[~df["outlier"]].copy()
    bairros = geo.load_bairros()

    distributions(df, clean)
    boxplots_price(clean)
    stats = by_bairro(clean)
    corr = correlations(clean)
    mismatches, suspects = spatial_inspection(clean, bairros)
    sources = source_comparison(clean)

    reliable = stats[stats["anuncios"] >= MIN_PER_BAIRRO]
    quality = json.loads((config.REPORTS_DIR / "qualidade_dados.json").read_text(encoding="utf-8"))
    summary = {
        "anuncios_total": len(df),
        "anuncios_modelagem": len(clean),
        "outliers_removidos": int(df["outlier"].sum()),
        "aluguel_mediano": float(clean["aluguel"].median()),
        "preco_m2_mediano": float(clean["preco_m2"].median()),
        "bairros_com_dados": int(len(stats)),
        "bairros_confiaveis": int(len(reliable)),
        "bairros_poucos_dados": sorted(stats[stats["anuncios"] < MIN_PER_BAIRRO].index.tolist()),
        "corr_area_aluguel": float(corr.loc["area", "aluguel"]),
        "corr_dist_centro_preco_m2": float(corr.loc["dist_centro_km", "preco_m2"]),
        "pontos_fora_do_bairro_declarado": mismatches,
        "pontos_suspeitos": suspects,
    }
    (config.REPORTS_DIR / "eda_resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(df, clean, stats, reliable, sources, corr, summary, quality)
    print(f"EDA gerada em {config.REPORTS_DIR} ({len(list(config.FIGURES_DIR.glob('*.png')))} figuras)")
    return summary


def write_report(df, clean, stats, reliable, sources, corr, summary, quality):
    top = reliable.sort_values("aluguel_mediano", ascending=False)
    ppm2 = reliable.sort_values("preco_m2_mediano", ascending=False)
    variable = reliable.sort_values("aluguel_cv", ascending=False)

    def listing(table, column, fmt, n=5, ascending=False):
        rows = table.sort_values(column, ascending=ascending).head(n)
        return "\n".join(f"- **{name}** — {fmt(row[column])} ({int(row['anuncios'])} anúncios)" for name, row in rows.iterrows())

    outlier_reasons = ", ".join(f"`{k}` ({v})" for k, v in quality["motivos_outlier"].items()) or "nenhum"
    lines = f"""# Análise exploratória

Base: **{summary['anuncios_total']}** casas de locação em Curitiba após a limpeza, das quais
**{summary['anuncios_modelagem']}** seguem para a modelagem ({summary['outliers_removidos']} marcadas como outlier: {outlier_reasons}).

| Fonte | Anúncios | Aluguel mediano | R$/m² mediano | Área mediana |
| :--- | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {name} | {int(row['anuncios'])} | {fmt_brl(row['aluguel_mediano'])} | {row['preco_m2_mediano']:.1f} | {row['area_mediana']:.0f} m² |"
        for name, row in sources.iterrows()
    ) + f"""

![Distribuições](figures/01_distribuicoes.png)
![Boxplots](figures/02_boxplots_preco.png)

## Perguntas de negócio

**Qual o nível geral de preços?** Aluguel mediano de **{fmt_brl(summary['aluguel_mediano'])}** e
**R$ {summary['preco_m2_mediano']:.1f}/m²**. A distribuição do aluguel é assimétrica à direita (cauda de casas
de alto padrão), por isso a modelagem trabalha em escala logarítmica.

**Quais os bairros mais caros e mais baratos?** (apenas bairros com ≥ {MIN_PER_BAIRRO} anúncios)

Maior aluguel mediano:
{listing(reliable, 'aluguel_mediano', fmt_brl)}

Menor aluguel mediano:
{listing(reliable, 'aluguel_mediano', fmt_brl, ascending=True)}

Maior preço/m²:
{listing(reliable, 'preco_m2_mediano', lambda v: f'R$ {v:.1f}/m²')}

![Bairros](figures/03_bairros_preco.png)
![Boxplot por bairro](figures/06_boxplot_bairros.png)

**Onde há mais anúncios?** {summary['bairros_com_dados']} bairros têm ao menos um anúncio, mas só
{summary['bairros_confiaveis']} passam de {MIN_PER_BAIRRO}. Os demais ({len(summary['bairros_poucos_dados'])}) aparecem no mapa com aviso de
poucos dados: {', '.join(summary['bairros_poucos_dados']) or 'nenhum'}.

![Contagem](figures/04_contagem_bairros.png)

**Onde o preço varia mais?** Bairros com maior coeficiente de variação do aluguel — a mediana do bairro diz pouco ali:
{listing(variable, 'aluguel_cv', lambda v: f'CV {v:.2f}')}

![Variabilidade](figures/05_variabilidade_bairros.png)

**O que se relaciona com o preço?** A área é o fator mais forte (Spearman com aluguel:
**{summary['corr_area_aluguel']:.2f}**); a distância ao centro tem correlação
**{summary['corr_dist_centro_preco_m2']:.2f}** com o preço/m².

![Correlação](figures/07_correlacao.png)
![Dispersão](figures/08_dispersao.png)

## Qualidade espacial

{summary['pontos_fora_do_bairro_declarado']} anúncios caem em um bairro diferente do declarado (os limites do OpenStreetMap e os nomes
usados pelos portais nem sempre coincidem); {summary['pontos_suspeitos']} estão a mais de 1 km do bairro declarado e foram listados em
`reports/pontos_suspeitos.csv` para conferência. A figura mostra a origem de cada coordenada.

![Inspeção espacial](figures/09_inspecao_espacial.png)

## Fontes

![Fontes](figures/10_fontes.png)

As fontes têm perfis diferentes (a tabela do início mostra aluguel e área medianos distintos entre elas), o que justifica
manter `fonte` rastreável e olhar a divergência entre portais antes de tirar conclusões por bairro.
"""
    (config.REPORTS_DIR / "EDA.md").write_text(lines, encoding="utf-8")


if __name__ == "__main__":
    run()
