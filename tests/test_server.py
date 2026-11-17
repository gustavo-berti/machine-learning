"""Front-end web (app/server.py): as rotas respondem e a estimativa bate com o simulador."""
import asyncio
import json

import pytest

from pipeline import config

pytest.importorskip("starlette")
pytestmark = pytest.mark.skipif(
    not (config.DATA_DIR / "imoveis_modelo.csv").exists() or not (config.MODELS_DIR / "regressor.joblib").exists(),
    reason="rode `python -m pipeline` antes",
)


def call(handler, method="GET", body=None, query=b""):
    from starlette.requests import Request

    payload = json.dumps(body).encode() if body is not None else b""

    async def receive():
        return {"type": "http.request", "body": payload}

    scope = {"type": "http", "method": method, "headers": [], "query_string": query}
    response = asyncio.run(handler(Request(scope, receive)))
    return response.status_code, json.loads(response.body)


def test_meta_and_listings():
    from app import server

    status, meta = call(server.meta)
    assert status == 200 and meta["total_anuncios"] > 0 and len(meta["bairros"]) > 50
    status, listings = call(server.listings)
    assert status == 200 and len(listings) == meta["total_anuncios"]
    assert all(-26 < row["latitude"] < -25 for row in listings)  # coordenadas não podem ser arredondadas


def test_estimate_inside_and_outside():
    from app import server

    body = dict(lat=-25.43, lon=-49.27, area=150, quartos=3, banheiros=2, garagem=True)
    status, result = call(server.estimate, "POST", body)
    assert status == 200 and not result["fora"]
    assert result["minimo"] < result["valor"] < result["maximo"]
    assert result["fatores"] and len(result["semelhantes"]) == 6

    status, result = call(server.estimate, "POST", {**body, "lat": -25.0})
    assert status == 200 and result["fora"]

    status, _ = call(server.estimate, "POST", {**body, "area": "abc"})
    assert status == 400
