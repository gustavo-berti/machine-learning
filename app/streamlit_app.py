"""MVP: mapa de aluguéis de casas em Curitiba + simulador de preço (claude.md, seção 5).

Rodar com:  ./venv/bin/streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from pipeline import config, geo, simulate

st.set_page_config(page_title="Aluguel de casas em Curitiba", page_icon="🏠", layout="wide")

MIN_ADS = simulate.MIN_ADS_PER_BAIRRO
BAND_COLORS = {"Baixa": "#3a9d5d", "Média": "#e0a526", "Alta": "#d1495b"}
METRICS = {
    "Aluguel mediano (R$)": ("aluguel_mediano", "R$ {:,.0f}"),
    "Preço mediano por m² (R$/m²)": ("preco_m2_mediano", "R$ {:,.1f}/m²"),
    "Quantidade de anúncios": ("anuncios", "{:,.0f}"),
}


# ------------------------------------------------------------------ dados

@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_DIR / "imoveis_modelo.csv", dtype={"cep": str})


@st.cache_data
def load_bairros():
    bairros = geo.load_bairros()
    bairros["geometry"] = bairros.geometry.simplify(0.0002)
    return bairros


@st.cache_resource
def load_models():
    return simulate.load_models(load_data())


@st.cache_data
def load_json_file(name):
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def brl(value):
    return f"R$ {value:,.0f}".replace(",", ".")


def neighborhood_stats(df):
    return (
        df.groupby("bairro_geo")
        .agg(anuncios=("aluguel", "size"), aluguel_mediano=("aluguel", "median"), preco_m2_mediano=("preco_m2", "median"))
        .round(1)
    )


# ------------------------------------------------------------------ mapas

def base_map(zoom=11):
    return folium.Map(location=config.CURITIBA_CENTER, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)


def choropleth_layer(bairros, stats, metric_key, fmt):
    values = stats.loc[stats["anuncios"] >= MIN_ADS, metric_key]
    colormap = cm.LinearColormap(["#ffffcc", "#fd8d3c", "#800026"], vmin=float(values.min()), vmax=float(values.max()))
    colormap.caption = next(label for label, (key, _) in METRICS.items() if key == metric_key)

    def style(feature):
        name = feature["properties"]["bairro"]
        if name not in stats.index:
            return {"fillColor": "#cccccc", "fillOpacity": 0.15, "color": "#999", "weight": 0.6}
        row = stats.loc[name]
        if row["anuncios"] < MIN_ADS:
            return {"fillColor": "#bbbbbb", "fillOpacity": 0.45, "color": "#777", "weight": 0.8, "dashArray": "4"}
        return {"fillColor": colormap(row[metric_key]), "fillOpacity": 0.65, "color": "#555", "weight": 0.8}

    enriched = bairros.merge(stats, left_on="bairro", right_index=True, how="left")
    enriched["resumo"] = enriched.apply(lambda r: tooltip_text(r, fmt, metric_key), axis=1)
    layer = folium.GeoJson(
        enriched[["bairro", "resumo", "geometry"]].to_json(),
        name="Preço por bairro",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["bairro", "resumo"], aliases=["Bairro", ""], labels=True, sticky=True),
    )
    return layer, colormap


def tooltip_text(row, fmt, metric_key):
    if pd.isna(row.get("anuncios")):
        return "sem anúncios na base"
    text = f"{int(row['anuncios'])} anúncios · mediana {brl(row['aluguel_mediano'])} · {row['preco_m2_mediano']:.1f} R$/m²"
    return text + " ⚠ poucos dados" if row["anuncios"] < MIN_ADS else text


def points_layer(df):
    group = folium.FeatureGroup(name="Anúncios", show=True)
    cuts = df["aluguel"].quantile([1 / 3, 2 / 3]).tolist()
    for row in df.itertuples():
        band = "Baixa" if row.aluguel <= cuts[0] else "Média" if row.aluguel <= cuts[1] else "Alta"
        estimated = f"<br>Estimado pelo modelo: {brl(row.aluguel_estimado)} ({row.desvio_pct:+.0f}%)" if pd.notna(row.aluguel_estimado) else ""
        approx = "<br><i>localização aproximada (centroide do bairro)</i>" if row.geo_origem == "centroide_bairro" else ""
        html = (
            f"<b>{brl(row.aluguel)}</b>/mês · {row.area:.0f} m² · {row.quartos:.0f} quartos"
            f"<br>{row.endereco if isinstance(row.endereco, str) else 'endereço n/d'} — {row.bairro_geo}"
            f"{estimated}{approx}<br><a href='{row.url}' target='_blank'>{row.fonte}</a>"
        ) if pd.notna(row.area) and pd.notna(row.quartos) else f"<b>{brl(row.aluguel)}</b>/mês — {row.bairro_geo}<br><a href='{row.url}' target='_blank'>{row.fonte}</a>"
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=4, color=BAND_COLORS[band], fill=True, fill_opacity=0.8, weight=1,
            popup=folium.Popup(html, max_width=280),
        ).add_to(group)
    return group


def build_market_map(df, bairros, stats, metric_label):
    metric_key, fmt = METRICS[metric_label]
    fmap = base_map()
    layer, colormap = choropleth_layer(bairros, stats, metric_key, fmt)
    layer.add_to(fmap)
    colormap.add_to(fmap)
    points_layer(df).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


# ------------------------------------------------------------------ filtros

def sidebar_filters(df):
    st.sidebar.header("Filtros")
    st.sidebar.selectbox("Finalidade", ["Locação"], help="A base só tem anúncios de aluguel.")
    types = sorted(df["tipo"].dropna().unique())
    chosen_types = st.sidebar.multiselect("Tipo de imóvel", types, default=types)
    bairros = sorted(df["bairro_geo"].dropna().unique())
    chosen_bairros = st.sidebar.multiselect("Bairro", bairros, placeholder="Todos os bairros")

    low, high = int(df["aluguel"].min()), int(df["aluguel"].max())
    price = st.sidebar.slider("Faixa de aluguel (R$/mês)", low, high, (low, high), step=100)
    rooms = st.sidebar.slider("Quartos", 1, int(df["quartos"].max()), (1, int(df["quartos"].max())))
    sources = sorted(df["fonte"].unique())
    chosen_sources = st.sidebar.multiselect("Fonte", sources, default=sources)

    mask = (
        df["tipo"].isin(chosen_types)
        & df["aluguel"].between(*price)
        & (df["quartos"].between(*rooms) | df["quartos"].isna())
        & df["fonte"].isin(chosen_sources)
    )
    if chosen_bairros:
        mask &= df["bairro_geo"].isin(chosen_bairros)
    return df[mask]


# ------------------------------------------------------------------ abas

def tab_map(filtered, bairros):
    if filtered.empty:
        st.warning("Nenhum anúncio com esses filtros.")
        return

    metric_label = st.radio("Camada colorida por bairro", list(METRICS), horizontal=True)
    stats = neighborhood_stats(filtered)
    fmap = build_market_map(filtered, bairros, stats, metric_label)
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[], key="mapa_mercado")

    st.caption(
        "Cada ponto é um anúncio (cor = tercil de aluguel: verde baixo, amarelo médio, vermelho alto). "
        f"Bairros em cinza tracejado têm menos de {MIN_ADS} anúncios nos filtros atuais e ficam fora da escala de cores."
    )
    few = stats[stats["anuncios"] < MIN_ADS]
    approx = int((filtered["geo_origem"] == "centroide_bairro").sum())
    if len(few):
        st.warning(f"**Regiões com poucos dados ({len(few)}):** " + ", ".join(f"{name} ({int(row.anuncios)})" for name, row in few.iterrows()))
    if approx:
        st.info(f"{approx} anúncios não tinham endereço geocodificável e aparecem no centroide do bairro.")

    left, right = st.columns(2)
    left.download_button("Baixar mapa (HTML)", fmap.get_root().render(), "mapa_alugueis_curitiba.html", "text/html")
    right.download_button("Baixar anúncios filtrados (CSV)", filtered.to_csv(index=False), "alugueis_filtrados.csv", "text/csv")

    table = stats.sort_values("aluguel_mediano", ascending=False).reset_index().rename(columns={
        "bairro_geo": "Bairro", "anuncios": "Anúncios", "aluguel_mediano": "Aluguel mediano (R$)", "preco_m2_mediano": "R$/m² mediano"})
    st.subheader("Resumo por bairro")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("Baixar relatório por bairro (CSV)", table.to_csv(index=False), "relatorio_bairros.csv", "text/csv")


def tab_simulator(df, bairros):
    models = load_models()
    st.markdown("**1.** Clique no mapa para escolher a localização · **2.** informe as características · **3.** veja a estimativa.")
    map_col, form_col = st.columns([3, 2])

    point = st.session_state.get("sim_point", config.CURITIBA_CENTER)
    with map_col:
        fmap = base_map(zoom=12 if "sim_point" in st.session_state else 11)
        folium.GeoJson(
            bairros[["bairro", "geometry"]].to_json(), name="Bairros",
            style_function=lambda f: {"fillOpacity": 0.03, "color": "#888", "weight": 0.7},
            tooltip=folium.GeoJsonTooltip(fields=["bairro"], aliases=[""]),
        ).add_to(fmap)
        folium.Marker(point, icon=folium.Icon(color="red", icon="home"), tooltip="Local do imóvel").add_to(fmap)
        fmap.location = list(point)
        clicked = st_folium(fmap, height=520, use_container_width=True, returned_objects=["last_clicked"], key="mapa_simulador")
        if clicked and clicked.get("last_clicked"):
            new_point = (clicked["last_clicked"]["lat"], clicked["last_clicked"]["lng"])
            if new_point != st.session_state.get("sim_point"):
                st.session_state["sim_point"] = new_point
                st.rerun()

    with form_col:
        area = st.number_input("Área (m²)", 20.0, 2000.0, 150.0, step=10.0)
        quartos = st.number_input("Quartos", 1, 10, 3)
        banheiros = st.number_input("Banheiros", 1, 10, 2)
        garagem = st.checkbox("Tem garagem", value=True)
        announced = st.number_input("Aluguel anunciado (opcional, R$)", 0, 100_000, 0, step=100,
                                    help="Informe para comparar com a estimativa do modelo.")
        st.caption(f"Local selecionado: {point[0]:.5f}, {point[1]:.5f}")

    result = simulate.estimate(models, bairros, df, area, quartos, banheiros, garagem, *point)
    if result is None:
        st.error("O ponto escolhido está fora dos limites de Curitiba. Clique dentro do município.")
        return

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Aluguel estimado", brl(result.value))
    c2.metric(f"Faixa provável ({models.metadata['intervalo']['cobertura_alvo']:.0%})", f"{brl(result.low)} – {brl(result.high)}")
    high = "" if np.isinf(result.band_range[1]) else f" a {brl(result.band_range[1])}"
    c3.metric("Faixa de preço", result.band, help=f"{result.band}: a partir de {brl(result.band_range[0])}{high}")
    st.caption(f"Bairro: **{result.bairro}** · modelo: {models.metadata['melhor_regressor']} (regressão) e {models.metadata['melhor_classificador']} (faixa).")

    for warning in result.warnings:
        st.warning(warning)

    if announced:
        if announced > result.high:
            st.error(f"O anunciado ({brl(announced)}) está **acima** da faixa provável — {announced / result.value - 1:+.0%} frente à estimativa.")
        elif announced < result.low:
            st.success(f"O anunciado ({brl(announced)}) está **abaixo** da faixa provável — {announced / result.value - 1:+.0%} frente à estimativa.")
        else:
            st.info(f"O anunciado ({brl(announced)}) está **dentro** da faixa provável ({announced / result.value - 1:+.0%} frente à estimativa).")

    left, right = st.columns(2)
    with left:
        st.subheader("O que move o preço aqui")
        table = simulate.sensitivity(models, bairros, df, area, quartos, banheiros, garagem, *point)
        st.dataframe(
            table, hide_index=True, use_container_width=True,
            column_config={
                "Aluguel estimado": st.column_config.NumberColumn(format="R$ %.0f"),
                "Variação (R$)": st.column_config.NumberColumn(format="%+.0f"),
                "Variação (%)": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )
        st.caption("Cada linha muda um fator e mantém os demais — é uma leitura do modelo, não uma relação causal.")
    with right:
        st.subheader("Imóveis semelhantes")
        similar = simulate.similar_listings(df, area, quartos, banheiros, *point)
        st.dataframe(
            similar.rename(columns={"bairro_geo": "Bairro", "endereco": "Endereço", "area": "m²", "quartos": "Quartos",
                                    "banheiros": "Banh.", "aluguel": "Aluguel", "aluguel_estimado": "Estimado",
                                    "distancia_km": "km", "fonte": "Fonte", "url": "Link"}),
            hide_index=True, use_container_width=True,
            column_config={"Link": st.column_config.LinkColumn(display_text="abrir"),
                           "Aluguel": st.column_config.NumberColumn(format="R$ %.0f"),
                           "Estimado": st.column_config.NumberColumn(format="R$ %.0f"),
                           "km": st.column_config.NumberColumn(format="%.1f")},
        )

    report = (
        f"# Estimativa de aluguel\n\nLocal: {point[0]:.5f}, {point[1]:.5f} ({result.bairro})\n\n"
        f"Casa de {area:.0f} m², {quartos} quartos, {banheiros} banheiros, {'com' if garagem else 'sem'} garagem.\n\n"
        f"- Estimativa: **{brl(result.value)}**/mês\n- Faixa provável: {brl(result.low)} – {brl(result.high)}\n- Faixa de preço: {result.band}\n"
        + "".join(f"- Aviso: {w}\n" for w in result.warnings)
        + "\n" + table.round(1).to_markdown(index=False)
    )
    st.download_button("Baixar relatório da simulação (Markdown)", report, "simulacao.md", "text/markdown")


def tab_market(filtered):
    if filtered.empty:
        st.warning("Nenhum anúncio com esses filtros.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anúncios", f"{len(filtered):,}".replace(",", "."))
    k2.metric("Aluguel mediano", brl(filtered["aluguel"].median()))
    k3.metric("Preço/m² mediano", f"R$ {filtered['preco_m2'].median():.1f}")
    k4.metric("Bairros com dados", filtered["bairro_geo"].nunique())

    left, right = st.columns(2)
    hist = alt.Chart(filtered).mark_bar(color="#2f6f8f").encode(
        alt.X("aluguel:Q", bin=alt.Bin(maxbins=40), title="Aluguel (R$/mês)"), alt.Y("count()", title="Anúncios"))
    left.subheader("Distribuição do aluguel")
    left.altair_chart(hist, use_container_width=True)
    hist2 = alt.Chart(filtered.dropna(subset=["preco_m2"])).mark_bar(color="#d1495b").encode(
        alt.X("preco_m2:Q", bin=alt.Bin(maxbins=40), title="R$/m²"), alt.Y("count()", title="Anúncios"))
    right.subheader("Distribuição do preço por m²")
    right.altair_chart(hist2, use_container_width=True)

    stats = neighborhood_stats(filtered)
    reliable = stats[stats["anuncios"] >= MIN_ADS].sort_values("preco_m2_mediano", ascending=False).reset_index()
    st.subheader(f"Preço/m² mediano por bairro (≥ {MIN_ADS} anúncios)")
    chart = alt.Chart(reliable).mark_bar(color="#d1495b").encode(
        x=alt.X("preco_m2_mediano:Q", title="R$/m² mediano"), y=alt.Y("bairro_geo:N", sort="-x", title=None),
        tooltip=["bairro_geo", "anuncios", "aluguel_mediano", "preco_m2_mediano"])
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Comparação temporal")
    history_path = config.DATA_DIR / "historico_resumo.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if len(history) >= 2:
        st.dataframe(history, hide_index=True, use_container_width=True)
        first, last = history.iloc[0], history.iloc[-1]
        change = (last["aluguel_mediano"] / first["aluguel_mediano"] - 1) * 100
        change_m2 = (last["preco_m2_mediano"] / first["preco_m2_mediano"] - 1) * 100
        st.metric(f"Aluguel mediano: {first['data_coleta']} → {last['data_coleta']}", brl(last["aluguel_mediano"]), f"{change:+.1f}%")
        st.metric("Preço/m² mediano no mesmo intervalo", f"R$ {last['preco_m2_mediano']:.1f}", f"{change_m2:+.1f}%")
        st.caption(f"Compara só as fontes presentes em todas as coletas ({first['fontes']}), após as mesmas regras de limpeza. "
                   "O intervalo é curto e a amostra de cada coleta é pequena: variações de poucos pontos percentuais não são tendência.")
    else:
        st.info("Só há uma coleta. Rode o scraper novamente em outra data para habilitar a comparação (cada coleta é arquivada em data/historico/).")


def tab_models(df, bairros):
    meta = load_json_file("modelos.json")
    if meta is None:
        st.warning("Rode `python -m pipeline` para treinar os modelos.")
        return

    st.subheader("Regressão (aluguel em R$)")
    reg = pd.DataFrame({
        name: {"MAE (R$)": r["teste"]["MAE"], "RMSE (R$)": r["teste"]["RMSE"], "MAPE": r["teste"]["MAPE"] * 100, "R²": r["teste"]["R2"], "RMSE em CV (R$)": r["validacao_cv"]["RMSE"]}
        for name, r in meta["regressao"].items()
    }).T
    st.dataframe(reg.style.format({"MAE (R$)": "{:,.0f}", "RMSE (R$)": "{:,.0f}", "MAPE": "{:.1f}%", "R²": "{:.3f}", "RMSE em CV (R$)": "{:,.0f}"}), use_container_width=True)
    st.caption(f"Teste = {meta['base']['teste']} casas separadas antes de qualquer ajuste. Melhor na validação cruzada: **{meta['melhor_regressor']}**. "
               f"A faixa provável do simulador cobre {meta['intervalo']['cobertura_no_teste']:.0%} do teste (meta {meta['intervalo']['cobertura_alvo']:.0%}).")

    st.subheader("Classificação em faixas de preço (Baixa / Média / Alta)")
    clf = pd.DataFrame({
        name: {"Acurácia": r["teste"]["accuracy"], "F1 macro": r["teste"]["f1_macro"], "F1 macro em CV": r["validacao_cv"]["f1_macro"]}
        for name, r in meta["classificacao"].items()
    }).T
    st.dataframe(clf.style.format("{:.3f}"), use_container_width=True)
    cuts = meta["faixas_preco"]["cortes_producao"]
    st.caption(f"Faixas por tercil do aluguel: Baixa até {brl(cuts[0])} · Média até {brl(cuts[1])} · Alta acima disso.")

    st.subheader("Fatores mais associados à previsão")
    groups = pd.DataFrame({"fator": list(meta["importancia_grupos"]), "importância": list(meta["importancia_grupos"].values())})
    st.altair_chart(alt.Chart(groups).mark_bar(color="#2f6f8f").encode(
        x=alt.X("importância:Q", title="Piora do erro (RMSE em log) ao embaralhar o fator"),
        y=alt.Y("fator:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320))), use_container_width=True)
    st.caption("Importância por permutação em grupo, no conjunto de teste. A localização inclui o preço dos anúncios vizinhos, "
               "que o modelo calcula a partir das coordenadas; por isso as colunas de um mesmo fator são embaralhadas juntas.")
    with st.expander("Importância por coluna isolada"):
        columns = pd.DataFrame({"coluna": list(meta["importancia_features"]), "importância": list(meta["importancia_features"].values())})
        st.dataframe(columns, hide_index=True, use_container_width=True)
        st.caption("Subestima a localização: latitude, longitude, distância ao centro e bairro dividem a mesma informação, "
                   "então embaralhar uma só é compensado pelas outras.")

    st.subheader("Agrupamento DBSCAN")
    view = st.radio("Visão", ["Regiões (concentração geográfica)", "Perfis (área, cômodos, preço/m²)"], horizontal=True)
    column, key = ("cluster_regiao", "regiao") if view.startswith("Regiões") else ("cluster_perfil", "perfil")
    info = meta["dbscan"][key]
    if not info.get("escolhido"):
        st.info("O DBSCAN não encontrou agrupamento útil nessa visão.")
        return
    chosen = info["escolhido"]
    st.caption(f"eps = {chosen['eps']} {chosen['unidade_eps']}, min_samples = {chosen['min_samples']}, silhouette = {chosen['silhouette']:.2f}; "
               f"{info['clusters']} clusters e {info['ruido']} anúncios como ruído (cinza). "
               + ("Regiões = bolsões de anúncios próximos entre si; a tabela mostra o preço de cada um." if key == "regiao"
                  else "Perfis = grupos de casas parecidas em tamanho, cômodos e preço/m², independentemente de onde estão."))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#d62728", "#393b79",
               "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363", "#fdae6b"]
    fmap = base_map()
    for row in df.itertuples():
        label = getattr(row, column)
        color = "#aaaaaa" if label == -1 else palette[label % len(palette)]
        folium.CircleMarker([row.latitude, row.longitude], radius=4, color=color, fill=True, fill_opacity=0.8, weight=0.5,
                            tooltip=f"cluster {label} · {brl(row.aluguel)} · {row.bairro_geo}").add_to(fmap)
    st_folium(fmap, height=520, use_container_width=True, returned_objects=[], key=f"mapa_dbscan_{key}")
    profiles = pd.DataFrame(meta["perfis_cluster_regiao" if key == "regiao" else "perfis_cluster_perfil"])
    st.dataframe(profiles, hide_index=True, use_container_width=True)


def tab_quality(df):
    quality = load_json_file("qualidade_dados.json")
    if quality is None:
        st.warning("Rode `python -m pipeline` para gerar o relatório de qualidade.")
        return

    inspection = quality["inspecao_inicial"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anúncios brutos", inspection["linhas"])
    k2.metric("Após limpeza", quality["linhas_finais"])
    k3.metric("Outliers marcados", quality["outliers"])
    k4.metric("Usados na modelagem", quality["linhas_para_modelagem"])

    st.subheader("Valores ausentes (base final, %)")
    missing = pd.Series(quality["ausentes_pct_final"]).sort_values(ascending=False)
    missing = missing[missing > 0].rename("% ausente").to_frame()
    st.dataframe(missing, use_container_width=True)
    st.caption("`vagas`, `condominio_iptu`, `custo_total` e `mobiliado` só existem nas fontes quintoandar e baggio; por isso a garagem "
               "entra no modelo como `tem_garagem` (contagem de vagas ou o slug do anúncio no chavesnamao).")

    st.subheader("Origem das coordenadas")
    origins = pd.Series(quality["geo_origem"]).rename("anúncios").to_frame()
    st.dataframe(origins, use_container_width=True)

    st.subheader("Limitações conhecidas")
    st.markdown(
        "- **Preço anunciado ≠ preço fechado**: a base é de anúncios, que costumam ficar acima do contratado.\n"
        "- **Sem suítes, idade do imóvel, padrão de acabamento ou condomínio em parte das fontes** — fatores relevantes que o modelo não vê.\n"
        "- **Poucos anúncios em vários bairros**: a estimativa nessas regiões depende da vizinhança, não do bairro.\n"
        "- **Portais com viés de oferta**: casas de alto padrão aparecem mais do que a distribuição real de aluguéis.\n"
        "- **Coordenadas aproximadas** (geocodificadas ou do centroide do bairro) em parte dos anúncios.\n"
        "- O intervalo do simulador é largo de propósito: o erro típico do modelo é alto para esta base."
    )
    st.subheader("Amostra da base")
    st.dataframe(df.head(200), use_container_width=True)


def main():
    st.title("🏠 Aluguel de casas em Curitiba")
    st.caption("Mapa de preços, análise por região e simulador de aluguel a partir de anúncios coletados de chavesnamao, quintoandar e baggio.")

    if not (config.DATA_DIR / "imoveis_modelo.csv").exists() or not (config.MODELS_DIR / "regressor.joblib").exists():
        st.error("Base e modelos não encontrados. Rode `./venv/bin/python -m pipeline` antes de abrir o app.")
        st.stop()

    df = load_data()
    bairros = load_bairros()
    filtered = sidebar_filters(df)
    st.sidebar.metric("Anúncios filtrados", len(filtered))

    tabs = st.tabs(["🗺️ Mapa", "🎯 Simulador", "📊 Mercado", "🤖 Modelos e clusters", "🧹 Dados e qualidade"])
    with tabs[0]:
        tab_map(filtered, bairros)
    with tabs[1]:
        tab_simulator(df, bairros)
    with tabs[2]:
        tab_market(filtered)
    with tabs[3]:
        tab_models(df, bairros)
    with tabs[4]:
        tab_quality(df)


main()
