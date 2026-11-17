"""Transformadores próprios. Ficam fora do modeling.py para o joblib conseguir recarregar o modelo
salvo mesmo quando o treino rodou como `python -m pipeline.modeling` (classe em __main__ não é importável)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

NEIGHBOR_SCALES = (5, 25)


def neighbor_columns(ks):
    return [f"vizinhos_preco_k{k}" for k in ks]


class NeighborPriceEncoder(BaseEstimator, TransformerMixin):
    """Feature de localização: preço médio dos k anúncios mais próximos do treino, em várias escalas.

    Uma escala curta (k pequeno) capta a rua/bairro; uma longa suaviza o ruído e capta a região.
    No treino cada linha é excluída da própria vizinhança (leave-one-out); do contrário o alvo
    vazaria para a feature e o modelo aprenderia a copiá-lo.
    `adjust_area=True` usa log(aluguel) - log(área), ou seja, o preço por m² em log, para a
    feature não confundir "vizinhança cara" com "casa grande". Com alvo ordinal (faixas)
    usa-se `adjust_area=False` e a feature vira a faixa média dos vizinhos.
    """

    def __init__(self, ks: tuple = NEIGHBOR_SCALES, adjust_area: bool = True):
        self.ks = ks
        self.adjust_area = adjust_area

    def _coords(self, X):
        return X[["latitude", "longitude"]].to_numpy(dtype=float)

    def fit(self, X, y):
        coords = self._coords(X)
        target = np.asarray(y, dtype=float)
        if self.adjust_area:
            self.area_fallback_ = float(np.nanmedian(X["log_area"]))
            target = target - X["log_area"].fillna(self.area_fallback_).to_numpy()

        self.global_mean_ = float(target.mean())
        self.target_ = target
        self.index_ = NearestNeighbors(n_neighbors=min(max(self.ks) + 1, len(coords))).fit(coords)
        return self

    def transform(self, X):
        reach = min(max(self.ks), len(self.target_))
        _, neighbors = self.index_.kneighbors(self._coords(X), n_neighbors=reach)
        return self._attach(X, {k: self.target_[neighbors[:, :k]].mean(axis=1) for k in self.ks})

    def fit_transform(self, X, y=None, **params):
        self.fit(X, y)
        _, neighbors = self.index_.kneighbors(self._coords(X))
        reach = max(self.ks)
        values = {k: np.empty(len(X)) for k in self.ks}
        for row, found in enumerate(neighbors):
            others = found[found != row][:reach]
            for k in self.ks:
                values[k][row] = self.target_[others[:k]].mean() if len(others) else self.global_mean_
        return self._attach(X, values)

    def _attach(self, X, values):
        X = X.copy()
        for k, column in zip(self.ks, neighbor_columns(self.ks)):
            X[column] = values[k]
        return X
