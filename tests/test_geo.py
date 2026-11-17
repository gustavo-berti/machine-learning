import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from pipeline import geo


def square_way(x0, y0, x1, y1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return [{"lat": y, "lon": x} for x, y in corners]


def test_relation_with_open_ways_closes_into_polygon():
    # Dois lados em ways separadas, como no OSM
    left = [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}, {"lat": 1, "lon": 1}]
    right = [{"lat": 1, "lon": 1}, {"lat": 1, "lon": 0}, {"lat": 0, "lon": 0}]
    relation = {"members": [
        {"type": "way", "role": "outer", "geometry": left},
        {"type": "way", "role": "outer", "geometry": right},
    ]}
    shape = geo.relation_to_geometry(relation)
    assert shape is not None and round(shape.area, 6) == 1.0


def test_relation_inner_ring_becomes_a_hole():
    relation = {"members": [
        {"type": "way", "role": "outer", "geometry": square_way(0, 0, 4, 4)},
        {"type": "way", "role": "inner", "geometry": square_way(1, 1, 2, 2)},
    ]}
    assert round(geo.relation_to_geometry(relation).area, 6) == 15.0


def test_relation_without_geometry_is_skipped():
    assert geo.relation_to_geometry({"members": [{"type": "node", "role": "label"}]}) is None


def test_assign_bairro_points_inside_and_outside():
    bairros = gpd.GeoDataFrame({"bairro": ["A"]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    df = pd.DataFrame({"latitude": [0.5, 5.0, None], "longitude": [0.5, 5.0, None]})
    result = geo.assign_bairro(df, bairros)
    assert result.iloc[0] == "A"
    assert pd.isna(result.iloc[1]) and pd.isna(result.iloc[2])


def test_inside_curitiba_bbox():
    assert geo.inside_curitiba(-25.43, -49.27)
    assert not geo.inside_curitiba(-23.5, -46.6)
