"""
Die Modelle und die beiden Schätzverfahren.

Alle Modelle haben dieselben zwei Befehle: `fit(X, y)` (trainieren) und
`predict(X)` (prognostizieren). Deshalb lassen sie sich in den Verfahren unten
beliebig austauschen.

Schätzverfahren:
- fama_macbeth_forecast: ein Modell pro Monat, Prognose = Mittel der
  Monatsmodelle im Fenster (Lewellen, 2015)
- stacked_forecast: ein Modell auf allen untereinandergestapelten Monaten des
  Fensters (Gu, Kelly & Xiu, 2020; Bali, Kelly, Mörke & Rahman, 2026)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class Investors(BaseEstimator, RegressorMixin):
    """K simulierte Investoren für die Machine Forecast Disagreement (Bali, Kelly, Mörke & Rahman, 2026).

    Jeder Investor ist ein eigener Entscheidungsbaum und unterscheidet sich von
    den anderen auf zwei Arten:
    1. Information: er sieht nur eine feste, zufällig gewählte Teilmenge der
       Charakteristika (`feature_share`, Paper: 50 %), in jeder Schätzung dieselbe
    2. Modell: er wird auf einer eigenen Zufallsstichprobe der Aktien trainiert
       (Bootstrap, wie bei einem Random Forest)
    `predict` liefert die mittlere Prognose aller Investoren, `predict_each`
    die Prognose jedes einzelnen Investors (eine Zeile pro Investor). Die
    Disagreement einer Aktie ist die Standardabweichung dieser Prognosen.
    """

    def __init__(self, n_investors=100, feature_share=0.5, max_depth=3, min_samples_leaf=50, random_state=42):
        self.n_investors = n_investors
        self.feature_share = feature_share
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_rows, n_features = X.shape
        n_seen = max(1, round(n_features * self.feature_share))
        # 1. Informationsmenge jedes Investors (bei gleichem random_state in jeder Schätzung dieselbe)
        self.features_ = [np.sort(rng.choice(n_features, n_seen, replace=False)) for _ in range(self.n_investors)]
        # 2. Jeder Investor schätzt seinen Baum auf einer eigenen Bootstrap-Stichprobe
        self.trees_ = []
        for k, cols in enumerate(self.features_):
            rows = rng.integers(0, n_rows, n_rows)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                         random_state=k)
            self.trees_.append(tree.fit(X[rows][:, cols], y[rows]))
        return self

    def predict_each(self, X):
        """Prognose jedes Investors: Tabelle Investoren x Zeilen von X."""
        return np.vstack([tree.predict(X[:, cols]) for cols, tree in zip(self.features_, self.trees_)])

    def predict(self, X):
        return self.predict_each(X).mean(axis=0)


# Hyperparameter begrenzen, wie fein ein Baum die Aktien aufteilen darf - je feiner, desto eher
# lernt er Zufallsrauschen statt echter Zusammenhänge:
#   max_depth         maximale Anzahl Ja/Nein-Fragen hintereinander (Tiefe des Baums)
#   min_samples_leaf  mindestens so viele Aktien pro Endknoten (Blatt); dessen Prognose ist ihre Durchschnittsrendite
#   n_estimators      Anzahl der Bäume
#   max_features      Random Forest: bei jeder Frage nur aus einer zufälligen Auswahl der Charakteristika
#                     wählen ("sqrt": Wurzel aus 14, also 3) - macht die Bäume verschieden
#   learning_rate     XGBoost: mit welchem Gewicht jeder neue Baum in die Prognose eingeht
#   subsample, colsample_bytree  XGBoost: zufälliger Anteil der Aktien bzw. Charakteristika je Baum
#   n_investors, feature_share   Investors: Anzahl der Investoren und Anteil der Charakteristika, den jeder sieht
#   n_jobs=-1         alle Prozessorkerne nutzen (ändert nichts am Ergebnis)
#   random_state      Startwert des Zufallsgenerators: gleiche Ergebnisse bei jedem Lauf
MODELS = {
    # Benchmark: die lineare Regression aus dem Paper
    "OLS": LinearRegression(),
    # ein einzelner Entscheidungsbaum
    "DecisionTree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=42),
    # Mittel vieler unabhängiger Bäume (Bagging)
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                          max_features="sqrt", n_jobs=-1, random_state=42),
    # Bäume nacheinander, jeder korrigiert die Fehler der vorigen (Boosting)
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42),
    # 100 simulierte Investoren, je ein Baum der Tiefe 3 (wie im MFD-Paper) auf der Hälfte der Charakteristika;
    # liefert zusätzlich die Disagreement (Standardabweichung der Investorprognosen)
    "Investors": Investors(n_investors=100, feature_share=0.5, max_depth=3, min_samples_leaf=50, random_state=42),
}


def predict(fitted, X):
    """Prognose eines trainierten Modells; bei Investors eine Zeile pro Investor."""
    return fitted.predict_each(X) if hasattr(fitted, "predict_each") else fitted.predict(X)


def empty_forecast(model, n_rows):
    """Leere Prognosetabelle: eine Zahl pro Zeile, bei Investors eine Zeile pro Investor."""
    return np.zeros((model.n_investors, n_rows)) if isinstance(model, Investors) else np.zeros(n_rows)


def fama_macbeth_forecast(model, X, y, month_nr, first, last, window, expanding=False, label=""):
    """Verfahren 1 (Lewellen): ein Modell pro Monat trainieren und die Prognosen der Monatsmodelle mitteln.

    Lewellen schätzt jeden Monat eine Querschnittsregression, mittelt die
    Koeffizienten der letzten 10 Jahre und prognostiziert damit. Hier gibt jedes
    der Monatsmodelle eine eigene Prognose ab, und diese werden gemittelt - für
    eine Regression ist das exakt dasselbe, und es funktioniert genauso für Bäume.

    Monatsmodell j lernt den Zusammenhang zwischen den Charakteristika am Ende
    von Monat j und der Rendite im Monat j + 1. Die Prognose für Monat t ist das
    Mittel der Modelle der Monate t - `window` bis t - 1 (bei `expanding` aller
    Monate seit Beginn der Daten, im Paper "cumulative"). Deren Renditen sind in
    t alle bekannt: Es werden keine Zukunftsdaten verwendet.

    `month_nr` ist die Nummer des Monats jeder Zeile von X (0 = erster Monat),
    `first` und `last` sind die Nummern des ersten und letzten Prognosemonats.
    Rückgabe: (Prognose je Zeile, NaN außerhalb der Prognosemonate - bei
    Investors eine Zeile pro Investor; Feature Importances aller Monatsmodelle).
    """
    forecast_sum = empty_forecast(model, len(y))  # Summe der Prognosen je Aktie und Monat
    n_models = np.zeros(len(y))                   # Anzahl der Monatsmodelle, die diese Prognose enthält
    importances = []

    first_model = 0 if expanding else max(first - window, 0)
    for j in range(first_model, last):
        # 1. Modell auf dem Querschnitt von Monat j trainieren (ca. 500 Aktien x 14 Charakteristika)
        in_month_j = month_nr == j
        fitted = clone(model).fit(X[in_month_j], y[in_month_j])

        # 2. Damit alle Aktien der Monate j + 1 ... j + window prognostizieren (expanding: alle späteren)
        last_month = last if expanding else min(j + window, last)
        in_forecast_months = (month_nr > j) & (month_nr >= first) & (month_nr <= last_month)
        forecast_sum[..., in_forecast_months] += predict(fitted, X[in_forecast_months])
        n_models[in_forecast_months] += 1

        if hasattr(fitted, "feature_importances_"):
            importances.append(fitted.feature_importances_)
        print(f"\r  {label:<13} month {j - first_model + 1}/{last - first_model}", end="", flush=True)
    print()

    # 3. Mittelwert: Summe der Prognosen / Anzahl der Modelle
    with np.errstate(invalid="ignore"):  # 0 / 0 = NaN für Zeilen außerhalb des Prognosezeitraums
        return forecast_sum / n_models, importances


def stacked_forecast(model, X, y, month_nr, first, last, window, expanding=False, refit_every=12, label=""):
    """Verfahren 2 (Gu, Kelly & Xiu): ein Modell auf allen gestapelten Monaten des Fensters.

    Für den Prognosemonat t werden die Querschnitte der Monate t - `window` bis
    t - 1 untereinandergestapelt (120 Monate x 500 Aktien = 60.000 Zeilen; bei
    `expanding` alle Monate seit Beginn der Daten). Ein einziges Modell lernt
    darauf und prognostiziert die Monate t bis t + `refit_every` - 1; danach
    wird es neu trainiert (Paper: alle 12 Monate). Es fließen dieselben
    Informationen ein wie bei fama_macbeth_forecast - nur lernt hier ein
    Modell aus allen Monaten zugleich, statt 120 Monatsmodelle zu mitteln.

    Rückgabe wie fama_macbeth_forecast.
    """
    forecast = empty_forecast(model, len(y)) * np.nan
    importances = []

    refit_months = range(first, last + 1, refit_every)
    for i, t in enumerate(refit_months, start=1):
        # 1. Trainingsdaten: alle Zeilen der Monate im Fenster vor t
        in_window = (month_nr >= (0 if expanding else t - window)) & (month_nr < t)
        fitted = clone(model).fit(X[in_window], y[in_window])

        # 2. Prognose für die Monate t ... t + refit_every - 1
        in_forecast_months = (month_nr >= t) & (month_nr <= min(t + refit_every - 1, last))
        forecast[..., in_forecast_months] = predict(fitted, X[in_forecast_months])

        if hasattr(fitted, "feature_importances_"):
            importances.append(fitted.feature_importances_)
        print(f"\r  {label:<13} fit {i}/{len(refit_months)} ({in_window.sum():,} rows)", end="", flush=True)
    print()
    return forecast, importances
