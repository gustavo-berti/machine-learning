"""Front-end web do projeto: API JSON + página estática (app/web/).

Rodar com:  ./venv/bin/python -m app.server        (abre em http://127.0.0.1:8000)

O app Streamlit (app/streamlit_app.py) continua disponível como painel analítico
(modelos, clusters, qualidade dos dados). Este servidor é a interface do dia a dia:
clicar no mapa, descrever a casa e ver o aluguel estimado.
"""
import json
import math
import sys
import threading
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from pipeline import config, geo, simulate

WEB_DIR = Path(__file__).resolve().parent / "web"
_lock = threading.Lock()


# ------------------------------------------------------------------ dados (carregados uma vez)

@lru_cache(maxsize=1)
def data():
    return pd.read_csv(config.DATA_DIR / "imoveis_modelo.csv", dtype={"cep": str})


@lru_cache(maxsize=1)
def bairros():
    return geo.load_bairros()


@lru_cache(maxsize=1)
def models():
    return simulate.load_models(data())


def clean(value):
    """Converte NaN/inf/numpy em tipos que o JSON aceita."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else round(float(value), 6)
    return value


def records(frame):
    return [{k: clean(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


# ------------------------------------------------------------------ rotas

async def index(request):
    return FileResponse(WEB_DIR / "index.html")


@lru_cache(maxsize=1)
def meta_payload():
    df, meta = data(), models().metadata
    counts = df.groupby("bairro_geo").size()
    polys = bairros()
    centroids = polys.to_crs(31982).geometry.centroid.to_crs(4326)
    bairro_list = sorted(
        ({"nome": name, "lat": round(c.y, 5), "lon": round(c.x, 5), "anuncios": int(counts.get(name, 0))}
         for name, c in zip(polys["bairro"], centroids)),
        key=lambda b: b["nome"],
    )
    reports = config.REPORTS_DIR / "modelos.json"
    reg = json.loads(reports.read_text(encoding="utf-8"))["regressao"] if reports.exists() else {}
    best = reg.get(meta["melhor_regressor"], {}).get("teste", {})
    history_path = config.DATA_DIR / "historico_resumo.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    return {
        "centro": list(config.CURITIBA_CENTER),
        "min_anuncios": simulate.MIN_ADS_PER_BAIRRO,
        "total_anuncios": int(len(df)),
        "fontes": sorted(df["fonte"].unique().tolist()),
        "data_coleta": str(df["data_coleta"].max()),
        "aluguel_mediano": clean(df["aluguel"].median()),
        "preco_m2_mediano": clean(df["preco_m2"].median()),
        "faixas": {"rotulos": meta["faixas_preco"]["rotulos"], "cortes": meta["faixas_preco"]["cortes_producao"]},
        "intervalo_cobertura": meta["intervalo"]["cobertura_alvo"],
        "area_bem_coberta": [round(v) for v in models().area_range],
        "modelo": {
            "regressor": meta["melhor_regressor"],
            "erro_medio": clean(best.get("MAE")),
            "erro_pct": clean(best.get("MAPE", float("nan")) * 100) if best else None,
            "r2": clean(best.get("R2")),
            "importancia": meta.get("importancia_grupos", {}),
        },
        "historico": records(history) if len(history) else [],
        "bairros": bairro_list,
    }


async def meta(request):
    return JSONResponse(meta_payload())


@lru_cache(maxsize=1)
def bairros_geojson():
    polys = bairros()[["bairro", "geometry"]].copy()
    polys["geometry"] = polys.geometry.simplify(0.0002)
    return json.loads(polys.to_json())


async def bairros_geo(request):
    return JSONResponse(bairros_geojson())


@lru_cache(maxsize=1)
def listings_payload():
    cols = ["latitude", "longitude", "aluguel", "area", "quartos", "banheiros", "tem_garagem", "bairro_geo",
            "endereco", "preco_m2", "aluguel_estimado", "desvio_pct", "fonte", "url", "geo_origem"]
    frame = data()[cols].rename(columns={"bairro_geo": "bairro"})
    frame["aproximado"] = frame.pop("geo_origem") == "centroide_bairro"
    frame["endereco"] = frame["endereco"].where(frame["endereco"].notna(), None)
    return records(frame)


async def listings(request):
    return JSONResponse(listings_payload())


async def estimate(request):
    try:
        body = await request.json()
        lat, lon = float(body["lat"]), float(body["lon"])
        area = float(body["area"])
        quartos, banheiros = int(body["quartos"]), int(body["banheiros"])
        garagem = bool(body["garagem"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return JSONResponse({"erro": "Parâmetros inválidos."}, status_code=400)
    if not (10 <= area <= 3000 and 1 <= quartos <= 15 and 1 <= banheiros <= 15):
        return JSONResponse({"erro": "Valores fora do intervalo aceito."}, status_code=400)

    df, polys, mdl = data(), bairros(), models()
    with _lock:  # modelos sklearn não são garantidamente thread-safe
        result = simulate.estimate(mdl, polys, df, area, quartos, banheiros, garagem, lat, lon)
        if result is None:
            return JSONResponse({"fora": True, "erro": "Esse ponto fica fora de Curitiba. Clique dentro do município."})
        factors = simulate.sensitivity(mdl, polys, df, area, quartos, banheiros, garagem, lat, lon)
    similar = simulate.similar_listings(df, area, quartos, banheiros, lat, lon, k=6)
    similar = similar.join(df[["latitude", "longitude"]]).rename(columns={"bairro_geo": "bairro"})
    similar["endereco"] = similar["endereco"].where(similar["endereco"].notna(), None)

    return JSONResponse({
        "fora": False,
        "valor": clean(result.value),
        "minimo": clean(result.low),
        "maximo": clean(result.high),
        "faixa": result.band,
        "faixa_limites": [clean(v) for v in result.band_range],
        "bairro": result.bairro,
        "anuncios_no_bairro": int(mdl.density["anuncios"].get(result.bairro, 0)),
        "avisos": result.warnings,
        "fatores": [
            {"cenario": r["Cenário"], "valor": clean(r["Aluguel estimado"]), "variacao": clean(r["Variação (R$)"]),
             "variacao_pct": clean(r["Variação (%)"])}
            for _, r in factors.iterrows()
        ],
        "semelhantes": records(similar),
    })


async def search(request):
    """Busca de endereço: primeiro bairros (local), depois Nominatim restrito a Curitiba."""
    query = (request.query_params.get("q") or "").strip()
    if len(query) < 3:
        return JSONResponse([])
    try:
        response = requests.get(
            geo.NOMINATIM_URL,
            params={"q": f"{query}, Curitiba, PR", "format": "json", "limit": 5, "countrycodes": "br"},
            headers={"User-Agent": geo.NOMINATIM_USER_AGENT},
            timeout=10,
        )
        hits = response.json() if response.status_code == 200 else []
    except (requests.RequestException, ValueError):
        return JSONResponse({"erro": "Busca de endereço indisponível agora. Clique no mapa ou escolha um bairro."}, status_code=503)
    results = []
    for hit in hits:
        lat, lon = float(hit["lat"]), float(hit["lon"])
        if geo.inside_curitiba(lat, lon):
            results.append({"nome": hit.get("display_name", "").split(", Região")[0], "lat": lat, "lon": lon})
    return JSONResponse(results)


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/bairros", bairros_geo),
    Route("/api/anuncios", listings),
    Route("/api/estimar", estimate, methods=["POST"]),
    Route("/api/buscar", search),
    Mount("/static", StaticFiles(directory=WEB_DIR), name="static"),
])


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Front-end web do simulador de aluguel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("Carregando base e modelos...")
    meta_payload()
    print(f"Pronto: abra http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
