from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GEO_DIR = DATA_DIR / "geo"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RAW_CSV = DATA_DIR / "imoveis_curitiba.csv"
CLEAN_CSV = DATA_DIR / "imoveis_limpo.csv"
GEOCODE_CACHE = GEO_DIR / "geocode_cache.json"
OVERPASS_BAIRROS = GEO_DIR / "overpass_bairros.json"
BAIRROS_GEOJSON = GEO_DIR / "bairros_curitiba.geojson"

# Caixa que contém Curitiba; coordenada fora dela é geocodificação errada
CURITIBA_BBOX = {"lat": (-25.65, -25.33), "lon": (-49.40, -49.18)}
CURITIBA_CENTER = (-25.4284, -49.2733)

RANDOM_STATE = 42
TEST_SIZE = 0.2
