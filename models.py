"""
Modelldefinitionen und die beiden Schätzverfahren:

- Fama-MacBeth (wie Lewellen, 2015): in jedem Monat ein Modell auf dem Querschnitt,
  die Prognose ist das Mittel der Prognosen aller Monatsmodelle im Fenster
- Pooled: ein Modell auf allen untereinandergestapelten Querschnitten des Fensters

Beide Verfahren haben dieselbe Schnittstelle und dieselbe Rückgabe
(Prognosen, Feature Importances, gewählte Hyperparameter).

Beide Funktionen sind wie ein Skript von oben nach unten lesbar: Vorbereitung,
Schleife über die Modelle, Schleife über die Zeit, Ergebnis.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import util

MODEL_CLASSES = {
    "OLS": LinearRegression,  # Benchmark: mit run_fama_macbeth genau das Modell aus dem Paper
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
}


def get_models(names, params=None, random_state=None):
    """Erstellt für jeden Namen in `names` ein (noch untrainiertes) Modell.

    `params` ordnet jedem Modellnamen die Argumente seines Konstruktors zu.
    """
    params = params or {}
    models = {}
    for name in names:
        if name not in MODEL_CLASSES:
            raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_CLASSES)}")
        model_class = MODEL_CLASSES[name]
        kwargs = dict(params.get(name, {}))
        if "random_state" in model_class().get_params():  # OLS hat keinen Zufallsanteil
            kwargs.setdefault("random_state", random_state)
        models[name] = model_class(**kwargs)
    return models


# =============================================================================
# Fama-MacBeth-Schätzung (Lewellen, 2015)
# =============================================================================

def run_fama_macbeth(df, models, feature_cols, target_col, date_col, id_col, train_window,
                     expanding=False, test_start=None, test_end=None, size_col=None,
                     param_grids=None, validation_window=12, verbose=True):
    """Schätzt die erwarteten Renditen wie Lewellen (2015), mit beliebigen Modellen statt OLS.

    Lewellen schätzt in jedem Monat eine Querschnittsregression der Renditen auf
    die Charakteristika (Fama-MacBeth), mittelt die Steigungen der letzten 10
    Jahre und prognostiziert damit die Rendite des nächsten Monats. Hier wird in
    jedem Monat stattdessen ein Modell auf dem Querschnitt (n Aktien x T
    Charakteristika) trainiert und die Prognosen der Monatsmodelle der letzten
    `train_window` Monate werden gemittelt (bei `expanding` aller Monate seit
    Beginn der Daten, im Paper "cumulative"). Für OLS ist das identisch mit dem
    Paper. Prognostiziert wird jeder Monat zwischen `test_start` und `test_end`.

    Monatsmodell j wird mit den Charakteristika in j und der Rendite in j + 1
    trainiert und prognostiziert die Monate j + 1 bis j + `train_window` (bei
    `expanding` alle folgenden Monate). Die Prognose für einen Monat ist das
    Mittel aller Modelle, die ihn prognostizieren - für lineare Modelle ist das
    genau die Prognose mit den gemittelten Fama-MacBeth-Steigungen wie im Paper.

    Ist `param_grids` für ein Modell gesetzt, werden dessen Hyperparameter jeden
    Monat neu gewählt: Für jede Kombination aus dem Grid werden die Prognosen
    schon ab `validation_window` Monaten vor dem ersten Prognosemonat berechnet.
    In Monat t wird dann die Kombination verwendet, deren Prognosen in den
    `validation_window` Monaten vor t den kleinsten MSE hatten. Diese Prognosen
    sind selbst out-of-sample und ihre Renditen sind in t bekannt - es werden
    also keine Zukunftsdaten genutzt.

    Rückgabe
    --------
    predictions : DataFrame
        Eine Zeile pro (Datum, Aktie) in den Testmonaten mit der realisierten
        Zielgröße und einer Prognosespalte pro Modell (Spaltenname = Modellname).
    importances : DataFrame
        Feature Importances pro Modell, gemittelt über alle Monatsmodelle.
    best_params : DataFrame
        Gewählte Hyperparameter pro Monat und Modell (leer ohne Grid Search).
    """
    # --- Vorbereitung: Monate als Blöcke von Zeilen ----------------------------
    df = df.sort_values([date_col, id_col]).reset_index(drop=True)
    dates = np.sort(df[date_col].unique())
    bounds = np.searchsorted(df[date_col].to_numpy(), dates, side="left")
    bounds = np.append(bounds, len(df))  # Zeilen von Monat k: bounds[k]:bounds[k + 1]
    first, last = util.test_period(dates, train_window, test_start, test_end)
    param_grids = param_grids or {}

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # Ausgabetabelle: alle Zeilen der Testmonate (first ... last)
    output_cols = [c for c in (date_col, id_col, size_col, target_col) if c]
    predictions = df.iloc[bounds[first]:bounds[last + 1]][output_cols].reset_index(drop=True)
    importances = {}
    best_params = []

    # --- Schleife über die Modelle ---------------------------------------------
    for name, model in models.items():
        param_grid = param_grids.get(name)
        combos = list(ParameterGrid(param_grid)) if param_grid else [{}]  # ohne Grid: eine leere Kombination

        # Ohne Grid Search starten die Prognosen bei `first`, mit Grid Search schon früher (Validierung)
        val_first = first
        if param_grid:
            val_first = max(first - validation_window, 1)
            if val_first >= first:
                raise ValueError("Grid search needs at least one month of data before the first forecast month.")
        start = 0 if expanding else max(val_first - train_window, 0)
        month = np.repeat(np.arange(val_first, last + 1), np.diff(bounds[val_first:last + 2]))  # Monat jeder Zeile

        forecasts = []         # eine Prognosereihe (val_first ... last) pro Kombination
        all_importances = []   # Feature Importances pro Kombination
        val_mse = {}           # MSE der Prognosen in den `validation_window` Monaten VOR jedem Monat

        # --- Schleife über die Hyperparameter-Kombinationen ---------------------
        for c, params in enumerate(combos):
            label = f"{name} {c + 1}/{len(combos)}" if param_grid else name
            sums = np.zeros(len(y))    # Summe der Prognosen je Zeile
            counts = np.zeros(len(y))  # Anzahl der Modelle, die diese Zeile prognostizieren
            importances_c = []

            # --- Schleife über die Monatsmodelle --------------------------------
            for j in range(start, last):
                fitted = clone(model).set_params(**params).fit(X[bounds[j]:bounds[j + 1]],
                                                               y[bounds[j]:bounds[j + 1]])

                # Monate, die dieses Modell prognostiziert (nur innerhalb von val_first ... last)
                last_forecast_month = last if expanding else min(j + train_window, last)
                forecast = slice(bounds[max(j + 1, val_first)], bounds[last_forecast_month + 1])
                sums[forecast] += fitted.predict(X[forecast])
                counts[forecast] += 1

                if hasattr(fitted, "feature_importances_"):
                    importances_c.append(fitted.feature_importances_)
                if verbose:
                    print(f"\r  {label:<13} month {j - start + 1}/{last - start}", end="", flush=True)
            if verbose:
                print()

            rows = slice(bounds[val_first], bounds[last + 1])
            forecasts.append(sums[rows] / counts[rows])
            all_importances.append(importances_c)

            if param_grid:
                # quadrierter Fehler pro Monat, dann gleitende Summe über `validation_window` Monate
                sq_err = pd.Series((forecasts[c] - y[rows]) ** 2).groupby(month)
                val_mse[c] = (sq_err.sum().rolling(validation_window, min_periods=1).sum()
                              / sq_err.count().rolling(validation_window, min_periods=1).sum()).shift(1)

        # --- Ergebnis dieses Modells ---------------------------------------------
        if not param_grid:
            predictions[name] = forecasts[0]
            importances[name] = all_importances[0]
        else:
            # beste Kombination pro Prognosemonat (kleinster MSE in den Monaten davor)
            best = pd.DataFrame(val_mse).loc[first:last].idxmin(axis=1)
            test_rows = np.flatnonzero(month >= first)
            predictions[name] = np.vstack(forecasts)[best.loc[month[test_rows]].to_numpy(), test_rows]
            importances[name] = all_importances[best.mode()[0]]  # der am häufigsten gewählten Kombination
            best_params += [{date_col: date, "model": name, "params": combos[c]}
                            for date, c in zip(dates[first:last + 1], best)]

    # Feature Importances pro Modell, gemittelt über alle trainierten Modelle (OLS hat keine)
    mean_importances = pd.DataFrame(
        {name: np.mean(values, axis=0) for name, values in importances.items() if values},
        index=feature_cols)
    return predictions, mean_importances, pd.DataFrame(best_params)


# =============================================================================
# Pooled-Schätzung (gestapelte Querschnitte)
# =============================================================================

def run_pooled(df, models, feature_cols, target_col, date_col, id_col, train_window,
               retrain_every=1, expanding=False, test_start=None, test_end=None, size_col=None,
               universe_size=None, param_grids=None, validation_window=12, verbose=True):
    """Trainiert jedes Modell auf einem Rolling Window und prognostiziert die folgende(n) Periode(n).

    Jedes Datum ist ein Querschnitt: n Aktien x T Charakteristika. Pro Fenster
    werden die letzten `train_window` Querschnitte (bei `expanding` alle seit
    Beginn der Daten) zu einer Trainingsmatrix X_train der Dimension
    (train_window * n) x T untereinandergestapelt. Damit prognostizieren die
    Modelle out-of-sample die Querschnitte der nächsten `retrain_every` Perioden
    (X_test: n x T pro Datum). Danach rückt das Fenster um `retrain_every`
    Perioden weiter. Prognostiziert wird nur zwischen `test_start` und `test_end`.

    Ist `universe_size` gesetzt, werden in jedem Fenster nur die `universe_size`
    größten Aktien zum ersten Testdatum verwendet (siehe `util.fixed_universe`).

    Ist `param_grids` für ein Modell gesetzt, werden dessen Hyperparameter in
    jedem Fenster optimiert: Jede Kombination wird auf den früheren Perioden
    trainiert und auf den letzten `validation_window` Perioden des Fensters
    bewertet (MSE, zeitliche Aufteilung statt zufälliger Folds, damit nicht mit
    Zukunftsdaten validiert wird). Die beste Kombination wird anschließend auf
    allen Perioden des Fensters neu trainiert.

    Rückgabe wie `run_fama_macbeth`.
    """
    # --- Vorbereitung ----------------------------------------------------------
    windows = list(util.rolling_windows(df[date_col], train_window, retrain_every, test_start, test_end,
                                        expanding))
    param_grids = param_grids or {}
    if param_grids and validation_window >= train_window:
        raise ValueError("validation_window must be smaller than train_window.")

    output_cols = [c for c in (date_col, id_col, size_col, target_col) if c]
    predictions = []
    importances = {name: [] for name in models}
    best_params = []

    # --- Schleife über die Fenster ---------------------------------------------
    for i, (train_dates, test_dates) in enumerate(windows, start=1):
        train = df[df[date_col].isin(train_dates)]
        test = df[df[date_col].isin(test_dates)]
        if universe_size:
            train, test = util.fixed_universe(train, test, date_col, id_col, size_col, universe_size)
        X_train, y_train = train[feature_cols], train[target_col]

        out = test[output_cols].copy()

        # --- Schleife über die Modelle -------------------------------------------
        for name, model in models.items():
            # in jedem Fenster ein frisches Modell (clone), damit nichts aus früheren Fenstern übernommen wird
            if name in param_grids:
                # Validierung auf den letzten `validation_window` Perioden des Trainingsfensters
                val_dates = np.sort(pd.unique(train[date_col].to_numpy()))[-validation_window:]
                test_fold = np.where(np.isin(train[date_col].to_numpy(), val_dates), 0, -1)  # 0 = Validierung, -1 = Training
                search = GridSearchCV(clone(model), param_grids[name], cv=PredefinedSplit(test_fold),
                                      scoring="neg_mean_squared_error", refit=True)
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                best_params.append({date_col: test_dates[0], "model": name, "params": search.best_params_})
            else:
                fitted = clone(model).fit(X_train, y_train)

            out[name] = fitted.predict(test[feature_cols])
            if hasattr(fitted, "feature_importances_"):
                importances[name].append(fitted.feature_importances_)
        predictions.append(out)

        if verbose:
            print(f"\r  Window {i}/{len(windows)}  "
                  f"train {util.fmt_date(train_dates[0])} - {util.fmt_date(train_dates[-1])} "
                  f"X_train {X_train.shape[0]}x{X_train.shape[1]}  "
                  f"test {util.fmt_date(test_dates[0])}", end="", flush=True)
    if verbose:
        print()

    # Feature Importances pro Modell, gemittelt über alle trainierten Modelle (OLS hat keine)
    mean_importances = pd.DataFrame(
        {name: np.mean(values, axis=0) for name, values in importances.items() if values},
        index=feature_cols)
    return pd.concat(predictions, ignore_index=True), mean_importances, pd.DataFrame(best_params)
