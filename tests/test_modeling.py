import numpy as np
import pandas as pd

from pipeline import modeling


def spatial_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-25.5, -25.4, n)
    lon = rng.uniform(-49.3, -49.2, n)
    log_area = np.log(rng.uniform(60, 300, n))
    return pd.DataFrame({"latitude": lat, "longitude": lon, "log_area": log_area}), rng.normal(8, 0.5, n)


def test_neighbor_encoder_training_excludes_the_row_itself():
    X, y = spatial_frame()
    encoder = modeling.NeighborPriceEncoder(ks=(5,), adjust_area=False)
    train_values = encoder.fit_transform(X, y)["vizinhos_preco_k5"].to_numpy()
    plain = encoder.transform(X)["vizinhos_preco_k5"].to_numpy()

    # transform() usa o próprio ponto como vizinho mais próximo; fit_transform() não pode
    assert not np.allclose(train_values, plain)

    # Alterar o alvo de UMA linha não pode mudar a feature dela própria no treino
    y_changed = y.copy()
    y_changed[0] += 100
    changed = modeling.NeighborPriceEncoder(ks=(5,), adjust_area=False).fit_transform(X, y_changed)["vizinhos_preco_k5"].to_numpy()
    assert changed[0] == train_values[0]


def test_neighbor_encoder_with_area_adjustment_uses_price_per_m2():
    X, y = spatial_frame()
    encoder = modeling.NeighborPriceEncoder(ks=(5,), adjust_area=True).fit(X, y)
    assert np.isclose(encoder.target_[0], y[0] - X["log_area"][0])


def test_split_has_no_overlap_and_keeps_all_rows():
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"aluguel": rng.uniform(1000, 20000, 200), "id": range(200)})
    train, test = modeling.split(data)
    assert len(train) + len(test) == 200
    assert set(train["id"]).isdisjoint(test["id"])


def test_price_bands_follow_edges():
    bands = modeling.price_bands([1000, 3000, 9000], [2000, 5000])
    assert bands.tolist() == [0, 1, 2]


def test_regression_metrics_back_to_reais():
    metrics = modeling.regression_metrics(np.log([1000.0, 2000.0]), np.log([1100.0, 1800.0]))
    assert round(metrics["MAE"]) == 150  # (|1000-1100| + |2000-1800|) / 2
    assert 0 < metrics["MAPE"] < 0.2


def test_dbscan_search_finds_two_blobs():
    rng = np.random.default_rng(0)
    matrix = np.vstack([rng.normal(0, 0.1, (60, 2)), rng.normal(5, 0.1, (60, 2))])
    trials, best = modeling.dbscan_search(matrix, [0.3, 0.5], [5], min_clusters=2, max_share=0.6)
    assert best is not None and len(set(best[3]) - {-1}) == 2


def test_dbscan_search_rejects_one_giant_block():
    rng = np.random.default_rng(0)
    one_blob = np.vstack([rng.normal(0, 0.1, (200, 2)), rng.normal(9, 0.1, (6, 2))])
    _, best = modeling.dbscan_search(one_blob, [0.3], [5], min_clusters=2, max_share=0.5)
    assert best is None  # silhouette alto, mas 97% dos pontos num cluster só


def test_grouped_importance_ranks_the_informative_group_first():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.normal(size=300)})
    y = 3 * X["a"] + 0.1 * X["c"]
    model = LinearRegression().fit(X, y)
    ranking = modeling.grouped_importance(model, X, y, groups={"G1": ["a"], "G2": ["b", "c"]}, n_repeats=10)
    assert list(ranking)[0] == "G1" and ranking["G1"] > ranking["G2"] >= 0


def test_neighbor_encoder_emits_one_column_per_scale():
    X, y = spatial_frame()
    out = modeling.NeighborPriceEncoder(ks=(3, 10), adjust_area=False).fit_transform(X, y)
    assert {"vizinhos_preco_k3", "vizinhos_preco_k10"} <= set(out.columns)
    # a escala longa suaviza mais: varia menos entre linhas que a curta
    assert out["vizinhos_preco_k10"].std() < out["vizinhos_preco_k3"].std()
