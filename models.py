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

import itertools

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

    def __init__(self, n_investors=100, feature_share=0.5, max_depth=3, min_samples_leaf=25, random_state=42):
        self.n_investors = n_investors
        self.feature_share = feature_share
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        n_rows, n_features = X.shape
        n_seen = max(1, int(n_features * self.feature_share))  # wie sklearn bei max_features als Anteil
        # 1. Informationsmenge jedes Investors (bei gleichem random_state in jeder Schätzung dieselbe)
        feature_rng = np.random.default_rng(self.random_state)
        self.features_ = [np.sort(feature_rng.choice(n_features, n_seen, replace=False))
                          for _ in range(self.n_investors)]
        # 2. Bootstrap genau wie sklearns RandomForestRegressor: je Baum ein eigener Startwert, daraus
        #    n Zeilen mit Zurücklegen; gezogene Zeilen gehen mit ihrer Häufigkeit als Gewicht ein
        #    (min_samples_leaf zählt also verschiedene Aktien, nicht Kopien derselben Aktie)
        seed_rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        for cols in self.features_:
            seed = seed_rng.randint(np.iinfo(np.int32).max)
            rows = np.random.RandomState(seed).randint(0, n_rows, n_rows)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                         random_state=seed)
            self.trees_.append(tree.fit(X[:, cols], y, sample_weight=np.bincount(rows, minlength=n_rows)))
        return self

    def predict_each(self, X):
        """Prognose jedes Investors: Tabelle Investoren x Zeilen von X."""
        return np.vstack([tree.predict(X[:, cols]) for cols, tree in zip(self.features_, self.trees_)])

    def predict(self, X):
        return self.predict_each(X).mean(axis=0)


class HuberBoosting(BaseEstimator, RegressorMixin):
    """Gradient Boosting (XGBoost) mit Huber-Verlust wie "GBRT+H" bei Gu, Kelly & Xiu (2020).

    Der Huber-Verlust ist für kleine Fehler der quadratische Fehler und wächst
    für große Fehler nur linear; einzelne extreme Monatsrenditen verbiegen das
    Modell so weniger. Die Grenze `delta` liegt wie im Paper beim 99,9-%-Quantil
    der absoluten Abweichungen im Training (XGBoost nutzt die glatte
    "Pseudo-Huber"-Form). Gestartet wird beim Median der Renditen.
    """

    def __init__(self, n_estimators=300, max_depth=2, learning_rate=0.1, huber_quantile=0.999,
                 n_jobs=-1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.huber_quantile = huber_quantile
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        center = np.median(y)
        delta = np.quantile(np.abs(y - center), self.huber_quantile)
        self.model_ = XGBRegressor(objective="reg:pseudohubererror", huber_slope=delta, base_score=center,
                                   n_estimators=self.n_estimators, max_depth=self.max_depth,
                                   learning_rate=self.learning_rate, n_jobs=self.n_jobs,
                                   random_state=self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


# Hyperparameter begrenzen, wie fein ein Baum die Aktien aufteilen darf - je feiner, desto eher
# lernt er Zufallsrauschen statt echter Zusammenhänge:
#   max_depth         maximale Anzahl Ja/Nein-Fragen hintereinander (Tiefe des Baums)
#   min_samples_leaf  mindestens so viele Aktien pro Endknoten (Blatt); dessen Prognose ist ihre Durchschnittsrendite
#   n_estimators      Anzahl der Bäume
#   max_features      Random Forest: bei jeder Frage nur aus einer zufälligen Auswahl der Charakteristika wählen
#   learning_rate     Boosting: mit welchem Gewicht jeder neue Baum in die Prognose eingeht
#   n_investors, feature_share   Investors: Anzahl der Investoren und Anteil der Charakteristika, den jeder sieht
#   n_jobs=-1         alle Prozessorkerne nutzen (ändert nichts am Ergebnis)
#   random_state      Startwert des Zufallsgenerators: gleiche Ergebnisse bei jedem Lauf
# Die hier eingetragenen Werte sind die Standardwerte aus den Papern, angepasst an unsere Daten. Sie gelten
# ohne Validierung (TUNE_HYPERPARAMETERS = False in main.py); sonst werden Tiefe, Anzahl der Charakteristika
# je Frage usw. aus GRIDS gewählt.
#   - Tiefe 3: das MFD-Paper (Tabelle B1) nutzt Tiefe 3; bei Gu, Kelly & Xiu wählt der Random Forest
#     "im Mittel ein bis fünf Ebenen" - 3 liegt in der Mitte. Gilt auch für den einzelnen Baum (dafür
#     nennen die Paper keinen Wert).
#   - max_features = 3: die Paper lassen bei jeder Frage nur einen kleinen Teil der Charakteristika zu
#     (MFD-Paper 5 %, Gu, Kelly & Xiu 3 bis 50 von 920); bei 14 Charakteristika wären 5 % weniger als eine,
#     daher 3 (etwa die Wurzel aus 14, der übliche Standard).
#   - XGBoost: Tiefe 2, 500 Bäume, Lernrate 0,01 wie die Boosting-Variante im MFD-Paper (Tabelle C1,
#     Konfiguration 18); alle drei Werte liegen im Raster von Gu, Kelly & Xiu. Huber-Verlust wie dort.
#   - min_samples_leaf = 25 steht in keinem der Paper: Ein Monatsmodell sieht nur rund 500 Aktien, mit
#     sehr wenigen Aktien pro Blatt würde es vor allem Rauschen lernen. (output/return_fixed/ ist mit 25
#     gerechnet, output/return/ und output/rank/ noch mit 50.)
MODELS = {
    # Benchmark: die lineare Regression aus dem Paper (ohne Hyperparameter)
    "OLS": LinearRegression(),
    # ein einzelner Entscheidungsbaum
    "DecisionTree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=25, random_state=42),
    # Mittel vieler unabhängiger Bäume (Bagging); 300 Bäume wie bei Gu, Kelly & Xiu
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=25,
                                          max_features=3, n_jobs=-1, random_state=42),
    # Bäume nacheinander, jeder korrigiert die Fehler der vorigen (Boosting), mit Huber-Verlust
    "XGBoost": HuberBoosting(n_estimators=500, max_depth=2, learning_rate=0.01, random_state=42),
    # 100 simulierte Investoren, je ein Baum (Tiefe 3 wie im MFD-Paper) auf der Hälfte der Charakteristika;
    # liefert zusätzlich die Disagreement (Standardabweichung der Investorprognosen). Im Paper ist jeder
    # Investor ein Random Forest aus 2.000 Bäumen, jeder auf 5 % der Beobachtungen: Bei rund 500 Aktien
    # pro Monat wären das 25 Aktien pro Baum - zu wenig für Tiefe 3. Daher ein Baum auf allen Aktien.
    "Investors": Investors(n_investors=100, feature_share=0.5, max_depth=3, min_samples_leaf=25, random_state=42),
}

# Mögliche Hyperparameter, aus denen auf der Validierung gewählt wird (Gu, Kelly & Xiu 2020,
# Internet Appendix Tabelle A.5; dort 920 statt 14 Eingaben, daher max_features nur bis 10;
# XGBoost dort bis 1000 Bäume, hier höchstens 300).
# Alle übrigen Einstellungen (z. B. min_samples_leaf = 25) bleiben wie in MODELS.
GRIDS = {
    "DecisionTree": {"max_depth": [1, 2, 3, 4, 5, 6]},
    "RandomForest": {"max_depth": [1, 2, 3, 4, 5, 6], "max_features": [3, 5, 10]},
    "XGBoost": {"max_depth": [1, 2], "learning_rate": [0.01, 0.1], "n_estimators": [100, 300]},
    "Investors": {"max_depth": [1, 2, 3, 4, 5, 6]},
}


def grid_points(grid, n_features):
    """Alle Kombinationen eines Rasters als Liste von Dictionaries (max_features höchstens n_features)."""
    points = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    points = [{k: min(v, n_features) if k == "max_features" else v for k, v in p.items()} for p in points]
    return [dict(t) for t in dict.fromkeys(tuple(p.items()) for p in points)]  # Doppelte entfernen


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


# =============================================================================
# Hyperparameter auf einer Validierung wählen (Gu, Kelly & Xiu 2020)
# =============================================================================
# Beide Verfahren bekommen dieselben Daten: die `window` Monate vor dem Prognosemonat. Einmal im
# Jahr (alle `retune_every` Monate, wie im Paper) wird dieses Fenster geteilt: Auf den ersten Monaten
# wird trainiert, auf den letzten `validation` Monaten wird für jede Kombination aus GRIDS der mittlere
# quadratische Prognosefehler gemessen. Die Kombination mit dem kleinsten Fehler gilt für die
# nächsten 12 Prognosemonate; deren Modelle lernen dann wieder auf dem ganzen Fenster.
# Auch hier gibt es keine Zukunftsdaten: Die Validierung liegt vollständig vor dem Prognosemonat.


def tuning_months(first, last, retune_every):
    """Nummern der Monate, in denen die Hyperparameter neu gewählt werden."""
    return list(range(first, last + 1, retune_every))


def validation_error(forecast, y):
    """Mittlerer quadratischer Fehler; bei Investors zählt die mittlere Prognose."""
    return np.mean((y - (forecast.mean(axis=0) if forecast.ndim == 2 else forecast)) ** 2)


def tuned_fama_macbeth_forecast(model, grid, X, y, month_nr, first, last, window, validation=36,
                                retune_every=12, expanding=False, label=""):
    """Verfahren 1 (Lewellen) mit Hyperparametern, die jedes Jahr auf einer Validierung gewählt werden.

    Für jede Kombination aus `grid` wird jedes Monatsmodell nur einmal trainiert. Es geht ein
    - in die Prognose der Testmonate, genau wie in fama_macbeth_forecast, und
    - in die Validierung jedes Auswahltermins s, für den es zum Trainingsteil gehört
      (Monate s - window bis s - validation - 1): Die Validierungsprognose für die Monate
      s - validation bis s - 1 ist das Mittel dieser Monatsmodelle.
    Danach erhält jeder Block von 12 Testmonaten die Prognose der Kombination mit dem kleinsten
    Validierungsfehler.

    Rückgabe: (Prognose wie fama_macbeth_forecast, Feature Importances, gewählte Hyperparameter
    je Auswahltermin).
    """
    points = grid_points(grid, X.shape[1])
    starts = tuning_months(first, last, retune_every)
    validation_rows = {s: np.flatnonzero((month_nr >= s - validation) & (month_nr < s)) for s in starts}
    is_test = (month_nr >= first) & (month_nr <= last)
    errors = np.zeros((len(starts), len(points)))
    test_forecasts, importances_by_point = [], []

    first_model = 0 if expanding else max(first - window, 0)
    for p, params in enumerate(points):
        candidate = clone(model).set_params(**params)
        forecast_sum = empty_forecast(model, len(y)).astype(np.float32)
        n_models = np.zeros(len(y), dtype=np.float32)
        valid_sum = {s: np.zeros(len(validation_rows[s])) for s in starts}  # Summe der (mittleren) Prognosen
        n_valid = {s: 0 for s in starts}
        importances = []
        for j in range(first_model, last):
            in_month_j = month_nr == j
            fitted = clone(candidate).fit(X[in_month_j], y[in_month_j])
            if hasattr(fitted, "feature_importances_"):
                importances.append(fitted.feature_importances_)

            # alle Zeilen, für die Modell j eine Prognose abgibt (Testmonate und Validierungsmonate)
            last_month = last if expanding else min(j + window, last)
            rows = np.flatnonzero((month_nr > j) & (month_nr <= last_month) & (month_nr >= first - validation))
            forecast_j = predict(fitted, X[rows])

            in_test = is_test[rows]
            forecast_sum[..., rows[in_test]] += forecast_j[..., in_test]
            n_models[rows[in_test]] += 1
            mean_j = forecast_j.mean(axis=0) if forecast_j.ndim == 2 else forecast_j
            for s in starts:  # Validierung: Modell j gehört zum Trainingsteil des Fensters vor s
                if (expanding or j >= s - window) and j < s - validation:
                    valid_sum[s] += mean_j[np.searchsorted(rows, validation_rows[s])]
                    n_valid[s] += 1
            print(f"\r  {label:<13} {p + 1}/{len(points)} {params} month {j - first_model + 1}/{last - first_model}"
                  + " " * 10, end="", flush=True)
        for i, s in enumerate(starts):
            errors[i, p] = validation_error(valid_sum[s] / n_valid[s], y[validation_rows[s]])
        with np.errstate(invalid="ignore"):
            test_forecasts.append(forecast_sum / n_models)
        importances_by_point.append(importances)
    print()
    return combine_blocks(model, test_forecasts, importances_by_point, errors, points, starts, month_nr,
                          last, retune_every, len(y))


def tuned_stacked_forecast(model, grid, X, y, month_nr, first, last, window, validation=36, retune_every=12,
                           refit_every=1, expanding=False, label=""):
    """Verfahren 2 (Gu, Kelly & Xiu) mit Hyperparametern, die jedes Jahr auf einer Validierung gewählt werden.

    Für jeden Auswahltermin s lernt jede Kombination aus `grid` ein Modell auf den gestapelten
    Monaten s - window bis s - validation - 1 und prognostiziert die Monate s - validation bis
    s - 1. Mit der besten Kombination prognostiziert stacked_forecast die nächsten 12 Monate,
    neu trainiert alle `refit_every` Monate auf dem ganzen Fenster.

    Rückgabe wie tuned_fama_macbeth_forecast.
    """
    points = grid_points(grid, X.shape[1])
    starts = tuning_months(first, last, retune_every)
    errors = np.zeros((len(starts), len(points)))
    forecast = empty_forecast(model, len(y)) * np.nan
    importances, chosen = [], []
    for i, s in enumerate(starts):
        in_train = (month_nr >= (0 if expanding else s - window)) & (month_nr < s - validation)
        in_validation = (month_nr >= s - validation) & (month_nr < s)
        for p, params in enumerate(points):
            fitted = clone(model).set_params(**params).fit(X[in_train], y[in_train])
            errors[i, p] = validation_error(predict(fitted, X[in_validation]), y[in_validation])
            print(f"\r  {label:<13} tuning {i + 1}/{len(starts)}: {p + 1}/{len(points)} {params}" + " " * 10,
                  end="", flush=True)
        best = points[int(np.argmin(errors[i]))]
        chosen.append(best)
        block_end = min(s + retune_every - 1, last)
        block_forecast, block_importances = stacked_forecast(clone(model).set_params(**best), X, y, month_nr, s,
                                                             block_end, window, expanding, refit_every, label="")
        in_block = (month_nr >= s) & (month_nr <= block_end)
        forecast[..., in_block] = block_forecast[..., in_block]
        importances += block_importances
        print(f"\r  {label:<13} block {i + 1}/{len(starts)} done, chosen {best}" + " " * 20, end="", flush=True)
    print()
    return forecast, importances, chosen_table(chosen, starts, errors, points)


def combine_blocks(model, test_forecasts, importances_by_point, errors, points, starts, month_nr, last,
                   retune_every, n_rows):
    """Jeder Block von Testmonaten erhält die Prognose der Kombination mit dem kleinsten Validierungsfehler."""
    forecast = empty_forecast(model, n_rows) * np.nan
    importances, chosen = [], []
    for i, s in enumerate(starts):
        best = int(np.argmin(errors[i]))
        in_block = (month_nr >= s) & (month_nr <= min(s + retune_every - 1, last))
        forecast[..., in_block] = test_forecasts[best][..., in_block]
        importances += importances_by_point[best]
        chosen.append(points[best])
    return forecast, importances, chosen_table(chosen, starts, errors, points)


def chosen_table(chosen, starts, errors, points):
    """Gewählte Hyperparameter je Auswahltermin (Nummer des ersten Prognosemonats) mit Validierungsfehler."""
    rows = []
    for i, (s, params) in enumerate(zip(starts, chosen)):
        rows.append({"first_month_nr": s, **params, "validation_mse": errors[i].min(),
                     "validation_mse_worst": errors[i].max()})
    return rows
