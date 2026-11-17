"""Geoprocessamento: polígonos de bairros, geocodificação e associação ponto -> bairro."""
import json
import time

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.validation import make_valid
from shapely.ops import linemerge, polygonize, unary_union

from pipeline import config
from pipeline.text import BAIRRO_ALIASES, bairro_key

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# A política do Nominatim exige um User-Agent que identifique a aplicação e no máx. 1 req/s
NOMINATIM_USER_AGENT = "ml-curitiba-alugueis/1.0 (projeto academico de ML imobiliario)"
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
OVERPASS_QUERY = """[out:json][timeout:90];
area["name"="Curitiba"]["boundary"="administrative"]["admin_level"="8"]->.a;
relation(area.a)["boundary"="administrative"]["admin_level"="10"];
out geom;"""


def download_bairros(force: bool = False):
    """Baixa as relações administrativas (nível 10) de Curitiba do OpenStreetMap."""
    if config.OVERPASS_BAIRROS.exists() and not force:
        return

    config.GEO_DIR.mkdir(parents=True, exist_ok=True)
    for url in OVERPASS_URLS:
        response = requests.post(
            url,
            data={"data": OVERPASS_QUERY},
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=180,
        )
        if response.status_code == 200 and response.text.lstrip().startswith("{"):
            config.OVERPASS_BAIRROS.write_text(response.text, encoding="utf-8")
            return
    raise RuntimeError("Nenhum servidor Overpass respondeu com os limites dos bairros.")


def relation_to_geometry(relation: dict):
    """Monta o polígono de uma relação do OSM a partir das ways 'outer' e 'inner'."""
    def lines(role):
        return [
            [(point["lon"], point["lat"]) for point in member["geometry"]]
            for member in relation["members"]
            if member["type"] == "way" and member["role"] == role and member.get("geometry")
        ]

    def polygons(role):
        merged = linemerge([line for line in lines(role) if len(line) > 1])
        found = list(polygonize(merged))
        if found:
            return found

        # Anéis que se tocam em um ponto são descartados pelo polygonize; fecha-se o anel e corrige-se
        parts = getattr(merged, "geoms", [merged])
        rings = [Polygon(part.coords) for part in parts if len(part.coords) >= 4]
        return [make_valid(ring) for ring in rings if ring.area > 0]

    outer = unary_union(polygons("outer"))
    inner = polygons("inner")
    if inner:
        outer = outer.difference(unary_union(inner))
    if outer.is_empty:
        return None
    # make_valid pode devolver coleções com linhas soltas; só interessam as áreas
    areas = [part for part in getattr(outer, "geoms", [outer]) if isinstance(part, (Polygon, MultiPolygon))]
    return unary_union(areas) if areas else None


def build_bairros_geojson():
    download_bairros()
    data = json.loads(config.OVERPASS_BAIRROS.read_text(encoding="utf-8"))

    records = []
    for element in data["elements"]:
        geometry = relation_to_geometry(element)
        if geometry is None:
            continue
        name = element["tags"]["name"]
        canonical = BAIRRO_ALIASES.get(bairro_key(name), name)
        records.append({"bairro": canonical, "geometry": geometry})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["area_km2"] = (gdf.to_crs("EPSG:31982").area / 1e6).round(3)
    gdf.to_file(config.BAIRROS_GEOJSON, driver="GeoJSON")
    return gdf


def load_bairros():
    if not config.BAIRROS_GEOJSON.exists():
        return build_bairros_geojson()
    return gpd.read_file(config.BAIRROS_GEOJSON)


def assign_bairro(df: pd.DataFrame, bairros: gpd.GeoDataFrame):
    """Devolve o bairro (polígono) que contém cada ponto; NaN para pontos fora de todos."""
    points = gpd.GeoDataFrame(
        {"idx": df.index},
        geometry=[
            Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
            for lat, lon in zip(df["latitude"], df["longitude"])
        ],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points.dropna(subset="geometry"), bairros[["bairro", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    result = pd.Series(index=df.index, dtype="object")
    result.loc[joined["idx"].values] = joined["bairro"].values
    return result


def inside_curitiba(lat, lon):
    lat_min, lat_max = config.CURITIBA_BBOX["lat"]
    lon_min, lon_max = config.CURITIBA_BBOX["lon"]
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def load_geocode_cache():
    if config.GEOCODE_CACHE.exists():
        return json.loads(config.GEOCODE_CACHE.read_text(encoding="utf-8"))
    return {}


def geocode_query(session: requests.Session, query: str):
    response = session.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 1, "countrycodes": "br"},
        timeout=30,
    )
    if response.status_code != 200 or not response.json():
        return None
    hit = response.json()[0]
    return [float(hit["lat"]), float(hit["lon"])]


def geocode_addresses(queries: list, delay_seconds: float = 1.1, offline: bool = False):
    """Geocodifica endereços via Nominatim com cache em disco (resultado nulo também é cacheado).

    Com `offline=True` só consulta o cache, sem rede.
    """
    cache = load_geocode_cache()
    if offline:
        return cache
    pending = [query for query in dict.fromkeys(queries) if query not in cache]
    if pending:
        print(f"Geocodificando {len(pending)} endereços (Nominatim, 1 req/s)...")
    session = requests.Session()
    session.headers["User-Agent"] = NOMINATIM_USER_AGENT

    for position, query in enumerate(pending, 1):
        try:
            cache[query] = geocode_query(session, query)
        except requests.RequestException:
            continue
        if position % 25 == 0:
            config.GEOCODE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(delay_seconds)

    config.GEOCODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.GEOCODE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache
