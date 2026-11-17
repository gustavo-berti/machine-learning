"""Modelagem (claude.md, seções 2.5-2.8 e 4): DBSCAN, Decision Tree, MLP e SVM.

Ordem que evita vazamento: separa treino/teste primeiro; imputação, escala, one-hot e a
feature espacial de vizinhança entram num Pipeline e só veem o treino (e, dentro do
GridSearchCV, só a parte de treino de cada dobra).
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_absolute_percentage_error, r2_score, root_mean_squared_error, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pipeline import config, features
from pipeline.encoders import NEIGHBOR_SCALES, NeighborPriceEncoder, neighbor_columns

BAND_LABELS = ["Baixa", "Média", "Alta"]
CV_FOLDS = 5
INTERVAL_COVERAGE = 0.8


def build_preprocessor(adjust_area: bool, ks: tuple = NEIGHBOR_SCALES):
    numeric_columns = features.NUMERIC_FEATURES + neighbor_columns(ks)
    numeric = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=8, sparse_output=False)
    columns = ColumnTransformer(
        [("num", numeric, numeric_columns), ("bairro", categorical, features.CATEGORICAL_FEATURES)],
        remainder="drop",
    )
    return [("vizinhos", NeighborPriceEncoder(ks=ks, adjust_area=adjust_area)), ("colunas", columns)]


def make_pipeline(estimator, adjust_area: bool, ks: tuple = NEIGHBOR_SCALES):
    return Pipeline(build_preprocessor(adjust_area, ks) + [("modelo", estimator)])


def regression_candidates():
    return {
        "Baseline (mediana)": (DummyRegressor(strategy="median"), {}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=config.RANDOM_STATE),
            {"modelo__max_depth": [4, 6, 8, 12], "modelo__min_samples_leaf": [3, 5, 10, 20]},
        ),
        "MLP": (
            MLPRegressor(solver="lbfgs", max_iter=2000, random_state=config.RANDOM_STATE),
            {"modelo__hidden_layer_sizes": [(16,), (32,), (32, 16)], "modelo__alpha": [0.1, 1.0, 3.0, 10.0]},
        ),
        "SVM (SVR)": (
            SVR(kernel="rbf"),
            {"modelo__C": [1, 3, 10, 30], "modelo__epsilon": [0.05, 0.1, 0.2], "modelo__gamma": ["scale", 0.02, 0.05]},
        ),
    }


def classification_candidates():
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            {"modelo__max_depth": [3, 5, 8, 12], "modelo__min_samples_leaf": [3, 5, 10, 20]},
        ),
        "MLP": (
            MLPClassifier(solver="lbfgs", max_iter=2000, random_state=config.RANDOM_STATE),
            {"modelo__hidden_layer_sizes": [(16,), (32,), (32, 16)], "modelo__alpha": [0.1, 1.0, 3.0, 10.0]},
        ),
        "SVM (SVC)": (
            SVC(kernel="rbf", random_state=config.RANDOM_STATE),
            {"modelo__C": [1, 3, 10, 30], "modelo__gamma": ["scale", 0.02, 0.05]},
        ),
    }


def price_bands(values, edges):
    return np.digitize(values, edges)  # 0 = Baixa, 1 = Média, 2 = Alta


def regression_metrics(y_true_log, y_pred_log):
    true, pred = np.exp(y_true_log), np.exp(y_pred_log)
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(root_mean_squared_error(true, pred)),
        "MAPE": float(mean_absolute_percentage_error(true, pred)),
        "R2": float(r2_score(true, pred)),
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
    }


def prepare(df: pd.DataFrame):
    """Só entram na modelagem anúncios sem outlier e com as features-chave conhecidas."""
    data = df[~df["outlier"]].copy()
    data = features.add_model_columns(data)
    data = data.dropna(subset=["latitude", "longitude", "aluguel", "bairro_geo"]).reset_index(drop=True)
    return data


def split(data: pd.DataFrame):
    """Passo 8: separação feita antes de qualquer transformação ajustada.

    Estratifica por tercil de aluguel só para manter as faixas equilibradas nos dois lados;
    as faixas "oficiais" do classificador são recalculadas apenas com o treino.
    """
    strata = pd.qcut(data["aluguel"], 3, labels=False)
    train_idx, test_idx = train_test_split(
        data.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=strata
    )
    return data.loc[train_idx].reset_index(drop=True), data.loc[test_idx].reset_index(drop=True)


def fit_search(name, estimator, grid, adjust_area, X, y, cv, scoring):
    pipeline = make_pipeline(estimator, adjust_area)
    if not grid:
        return pipeline.fit(X, y), {}
    search = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X, y)
    return search.best_estimator_, {key.replace("modelo__", ""): value for key, value in search.best_params_.items()}


def run_regression(train, test):
    X_train, X_test = train[features.MODEL_FEATURES], test[features.MODEL_FEATURES]
    y_train, y_test = np.log(train["aluguel"]), np.log(test["aluguel"])
    cv = KFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, (estimator, grid) in regression_candidates().items():
        model, best = fit_search(name, estimator, grid, True, X_train, y_train, cv, "neg_root_mean_squared_error")
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv)
        results[name] = {
            "melhores_parametros": best,
            "validacao_cv": regression_metrics(y_train, oof),
            "teste": regression_metrics(y_test, model.predict(X_test)),
        }
        fitted[name] = model
        print(f"  {name:20s} teste: MAE R$ {results[name]['teste']['MAE']:,.0f} | R² {results[name]['teste']['R2']:.3f}")
    return results, fitted


def run_classification(train, test):
    edges = np.quantile(train["aluguel"], [1 / 3, 2 / 3])
    X_train, X_test = train[features.MODEL_FEATURES], test[features.MODEL_FEATURES]
    y_train, y_test = price_bands(train["aluguel"], edges), price_bands(test["aluguel"], edges)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, (estimator, grid) in classification_candidates().items():
        model, best = fit_search(name, estimator, grid, False, X_train, y_train, cv, "f1_macro")
        predicted = model.predict(X_test)
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv)
        results[name] = {
            "melhores_parametros": best,
            "validacao_cv": {"accuracy": float(accuracy_score(y_train, oof)), "f1_macro": float(f1_score(y_train, oof, average="macro"))},
            "teste": {
                "accuracy": float(accuracy_score(y_test, predicted)),
                "f1_macro": float(f1_score(y_test, predicted, average="macro")),
                "matriz_confusao": confusion_matrix(y_test, predicted).tolist(),
            },
        }
        fitted[name] = model
        print(f"  {name:20s} teste: acc {results[name]['teste']['accuracy']:.3f} | F1 {results[name]['teste']['f1_macro']:.3f}")
    return results, fitted, [float(edge) for edge in edges]


def residual_quantile(model, X, y, coverage=INTERVAL_COVERAGE):
    """Metade da largura do intervalo, em log: quantil dos erros fora da amostra (conformal)."""
    cv = KFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(clone(model), X, y, cv=cv)
    return float(np.quantile(np.abs(y - oof), coverage)), oof


FEATURE_GROUPS = {
    "Localização (coordenadas, bairro, vizinhança)": ["latitude", "longitude", "dist_centro_km", "bairro_geo", "densidade_anuncios_km2"],
    "Tamanho (área)": ["log_area", "area"],
    "Cômodos (quartos, banheiros)": ["quartos", "banheiros", "densidade_comodos"],
    "Garagem": ["tem_garagem"],
}


def grouped_importance(model, X, y, groups=FEATURE_GROUPS, n_repeats=30):
    """Permutação em grupo: embaralha juntas as colunas de um mesmo fator.

    A feature de vizinhança é calculada dentro do Pipeline a partir de latitude/longitude, então
    ela só entra na conta se essas colunas forem embaralhadas — por isso o agrupamento.
    Devolve quanto o RMSE (em log) piora quando o fator deixa de ser informativo.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    baseline = root_mean_squared_error(y, model.predict(X))
    result = {}
    for name, columns in groups.items():
        losses = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))
            shuffled = X.copy()
            shuffled[columns] = X[columns].iloc[order].to_numpy()
            losses.append(root_mean_squared_error(y, model.predict(shuffled)) - baseline)
        result[name] = float(np.mean(losses))
    return dict(sorted(result.items(), key=lambda item: -item[1]))


def feature_importance(model, X_test, y_test):
    """Importância por coluna (permutação simples); a de localização aparece diluída, ver grouped_importance."""
    result = permutation_importance(
        model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=20, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.Series(result.importances_mean, index=X_test.columns)
    importance = importance.drop(labels=["area"], errors="ignore")  # `area` é redundante com log_area
    return {name: float(value) for name, value in importance.sort_values(ascending=False).items()}


# ---------------------------------------------------------------- DBSCAN

def planar_km(df: pd.DataFrame):
    return np.column_stack([
        (df["longitude"] - features.CENTRO[1]) * features.KM_PER_DEGREE * np.cos(np.radians(features.CENTRO[0])),
        (df["latitude"] - features.CENTRO[0]) * features.KM_PER_DEGREE,
    ])


def dbscan_search(matrix, eps_grid, min_samples_grid, max_noise=0.35, min_clusters=5, max_share=0.5):
    """Escolhe eps/min_samples pelo maior silhouette entre as combinações aceitáveis.

    O silhouette sozinho premia soluções triviais (um bloco enorme + dois grupinhos), então a
    escolha exige: ao menos `min_clusters` clusters, no máximo `max_noise` de ruído e nenhum
    cluster com mais de `max_share` dos pontos agrupados.
    """
    trials, best = [], None
    for min_samples in min_samples_grid:
        for eps in eps_grid:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
            grouped = labels[labels != -1]
            clusters = len(set(grouped))
            noise = float((labels == -1).mean())
            largest = float(np.bincount(grouped).max() / len(grouped)) if clusters else 1.0
            silhouette = float(silhouette_score(matrix[labels != -1], grouped)) if clusters >= 2 else None
            trials.append({"eps": eps, "min_samples": min_samples, "clusters": clusters, "ruido": round(noise, 3),
                           "maior_cluster": round(largest, 3), "silhouette": silhouette})
            acceptable = clusters >= min_clusters and noise <= max_noise and largest <= max_share
            if silhouette is not None and acceptable and (best is None or silhouette > best[0]):
                best = (silhouette, eps, min_samples, labels)
    return trials, best


def run_dbscan(data: pd.DataFrame):
    """Duas visões: regiões (concentração geográfica de anúncios) e perfis (características do imóvel).

    Região usa só a posição em km (sem padronizar, para eps ter significado físico). Misturar
    preço/m² à posição foi testado e colapsa em um bloco único, porque a cidade é contínua.
    """
    # Anúncio sem área não tem preço/m²; usa-se a mediana para ele não virar ruído por falta de dado
    log_ppm2 = np.log(data["preco_m2"])
    log_ppm2 = log_ppm2.fillna(log_ppm2.median())

    views = {
        "regiao": (planar_km(data), False),
        "perfil": (np.column_stack([
            np.log(data["area"].fillna(data["area"].median())),
            data["quartos"].fillna(data["quartos"].median()),
            data["banheiros"].fillna(data["banheiros"].median()),
            log_ppm2,
        ]), True),
    }
    grids = {
        "regiao": dict(eps_grid=[0.4, 0.6, 0.8, 1.0, 1.2, 1.5], min_samples_grid=[6, 8, 12], max_noise=0.25),
        "perfil": dict(eps_grid=[0.3, 0.4, 0.5, 0.6, 0.8, 1.0], min_samples_grid=[5, 8, 12], max_noise=0.35),
    }

    labels_by_view, outputs = {}, {}
    for view, (raw, scale) in views.items():
        matrix = StandardScaler().fit_transform(raw) if scale else raw
        trials, best = dbscan_search(matrix, **grids[view])
        if best is None:
            labels_by_view[view] = np.full(len(data), -1)
            outputs[view] = {"escolhido": None, "tentativas": trials}
            continue
        silhouette, eps, min_samples, labels = best
        labels_by_view[view] = labels
        sizes = pd.Series(labels[labels != -1]).value_counts().sort_index()
        outputs[view] = {
            "escolhido": {"eps": eps, "min_samples": min_samples, "silhouette": silhouette,
                          "unidade_eps": "desvios-padrão" if scale else "km"},
            "clusters": int(len(sizes)),
            "ruido": int((labels == -1).sum()),
            "tamanhos": {int(k): int(v) for k, v in sizes.items()},
            "tentativas": trials,
        }
        print(f"  DBSCAN {view}: {len(sizes)} clusters, {(labels == -1).sum()} ruído, silhouette {silhouette:.2f} (eps={eps}, min={min_samples})")
    return labels_by_view, outputs


def cluster_profiles(data: pd.DataFrame, label_column: str):
    grouped = data[data[label_column] != -1].groupby(label_column)
    return grouped.agg(
        anuncios=("aluguel", "size"),
        aluguel_mediano=("aluguel", "median"),
        preco_m2_mediano=("preco_m2", "median"),
        area_mediana=("area", "median"),
        quartos_medianos=("quartos", "median"),
        dist_centro_km=("dist_centro_km", "median"),
        bairro_dominante=("bairro_geo", lambda s: s.mode().iat[0]),
    ).round(1)


# ---------------------------------------------------------------- orquestração

def run():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.CLEAN_CSV, dtype={"cep": str})
    data = prepare(df)
    train, test = split(data)
    print(f"Base de modelagem: {len(data)} casas | treino {len(train)} | teste {len(test)}\n")

    print("Regressão (aluguel contínuo, alvo em log):")
    regression, reg_models = run_regression(train, test)
    contenders = {name: result for name, result in regression.items() if name != "Baseline (mediana)"}
    best_reg = min(contenders, key=lambda name: contenders[name]["validacao_cv"]["RMSE"])

    print("\nClassificação (faixas de preço por tercil):")
    classification, clf_models, edges = run_classification(train, test)
    best_clf = max(classification, key=lambda name: classification[name]["validacao_cv"]["f1_macro"])

    print(f"\nMelhor regressor (validação cruzada): {best_reg} | melhor classificador: {best_clf}")

    # Intervalo de previsão e importâncias, usando o melhor regressor
    X_train, X_test = train[features.MODEL_FEATURES], test[features.MODEL_FEATURES]
    y_train, y_test = np.log(train["aluguel"]), np.log(test["aluguel"])
    half_width, _ = residual_quantile(reg_models[best_reg], X_train, y_train)
    test_pred = reg_models[best_reg].predict(X_test)
    coverage = float((np.abs(y_test - test_pred) <= half_width).mean())
    importance = feature_importance(reg_models[best_reg], X_test, y_test)
    group_importance = grouped_importance(reg_models[best_reg], X_test, y_test)

    # Modelos de produção: reajustados com treino + teste (a avaliação acima já foi feita no teste)
    X_all, y_all = data[features.MODEL_FEATURES], np.log(data["aluguel"])
    final_reg = clone(reg_models[best_reg]).fit(X_all, y_all)
    all_edges = np.quantile(data["aluguel"], [1 / 3, 2 / 3])
    final_clf = clone(clf_models[best_clf]).fit(X_all, price_bands(data["aluguel"], all_edges))

    cv = KFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_all = cross_val_predict(clone(reg_models[best_reg]), X_all, y_all, cv=cv)
    data["aluguel_estimado"] = np.exp(oof_all).round(0)
    data["desvio_pct"] = ((data["aluguel"] - data["aluguel_estimado"]) / data["aluguel_estimado"] * 100).round(1)

    print("\nDBSCAN:")
    labels, dbscan = run_dbscan(data)
    data["cluster_regiao"] = labels["regiao"]
    data["cluster_perfil"] = labels["perfil"]

    joblib.dump(final_reg, config.MODELS_DIR / "regressor.joblib")
    joblib.dump(final_clf, config.MODELS_DIR / "classificador.joblib")
    data.to_csv(config.DATA_DIR / "imoveis_modelo.csv", index=False)

    metadata = {
        "melhor_regressor": best_reg,
        "melhor_classificador": best_clf,
        "base": {"total": len(data), "treino": len(train), "teste": len(test)},
        "features": features.MODEL_FEATURES,
        "faixas_preco": {"rotulos": BAND_LABELS, "cortes_treino": edges, "cortes_producao": [float(e) for e in all_edges]},
        "intervalo": {"cobertura_alvo": INTERVAL_COVERAGE, "meia_largura_log": half_width, "cobertura_no_teste": coverage},
        "importancia_features": importance,
        "importancia_grupos": group_importance,
        "regressao": regression,
        "classificacao": classification,
        "dbscan": dbscan,
        "perfis_cluster_regiao": cluster_profiles(data, "cluster_regiao").reset_index().to_dict("records"),
        "perfis_cluster_perfil": cluster_profiles(data, "cluster_perfil").reset_index().to_dict("records"),
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.REPORTS_DIR / "modelos.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nIntervalo de {INTERVAL_COVERAGE:.0%}: ±{(np.exp(half_width) - 1):.0%} | cobertura no teste: {coverage:.0%}")
    return metadata


if __name__ == "__main__":
    run()
