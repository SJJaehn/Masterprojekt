"""
Schritt 2: Modelle schätzen und die erwarteten Renditen out-of-sample prognostizieren.

Zwei Verfahren mit derselben Schnittstelle und derselben Rückgabe
(Prognosen, Feature Importances, gewählte Hyperparameter):

- Fama-MacBeth (wie Lewellen, 2015): in jedem Monat ein Modell auf dem
  Querschnitt; die Prognose ist das Mittel der Prognosen aller Monatsmodelle
  im Fenster der letzten 10 Jahre
- Pooled: ein Modell auf allen untereinandergestapelten Querschnitten des Fensters

Die Modelle sind austauschbar: OLS (lineare Regression, genau das Modell aus dem
Paper), Decision Tree, Random Forest und XGBoost. Alle haben dieselben zwei
Befehle: `fit(X, y)` (trainieren) und `predict(X)` (prognostizieren).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from report import fmt_date

MODEL_CLASSES = {
    "OLS": LinearRegression,              # Benchmark: mit run_fama_macbeth genau das Modell aus dem Paper
    "DecisionTree": DecisionTreeRegressor,  # ein einzelner Entscheidungsbaum
    "RandomForest": RandomForestRegressor,  # Mittel vieler unabhängiger Bäume (Bagging)
    "XGBoost": XGBRegressor,              # Bäume nacheinander, jeder korrigiert die Fehler der vorigen (Boosting)
}


def get_models(names, params=None, random_state=None):
    """Erstellt für jeden Namen in `names` ein (noch untrainiertes) Modell.

    `params` ordnet jedem Modellnamen seine Hyperparameter zu, z.B.
    {"DecisionTree": {"max_depth": 5}}.
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


def test_period(dates, min_history, test_start=None, test_end=None):
    """Nummern (first, last) des ersten und letzten Prognosemonats in den sortierten `dates`.

    Vor dem ersten Prognosemonat liegen mindestens `min_history` Monate.
    Prognostiziert wird nur zwischen `test_start` und `test_end` (None = ohne Grenze).
    """
    first = min_history
    if test_start is not None:
        first = max(first, np.searchsorted(dates, pd.Timestamp(test_start).to_datetime64()))
    last = len(dates) - 1
    if test_end is not None:
        last = np.searchsorted(dates, pd.Timestamp(test_end).to_datetime64(), side="right") - 1
    if first > last:
        raise ValueError(f"No test periods: {min_history} periods of history needed before the first "
                         f"forecast, test from {test_start} to {test_end}, data has {len(dates)} periods.")
    return first, last


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
    jedem Monat stattdessen ein Modell auf dem Querschnitt trainiert, und die
    Prognosen der Monatsmodelle der letzten `train_window` Monate werden
    gemittelt (bei `expanding` aller Monate seit Beginn der Daten, im Paper
    "cumulative"). Für OLS ist das identisch mit dem Paper (siehe README).
    Prognostiziert wird jeder Monat zwischen `test_start` und `test_end`.

    Ist `param_grids` für ein Modell gesetzt, werden dessen Hyperparameter jeden
    Monat neu gewählt (siehe `choose_params_each_month`).

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
    # --- Vorbereitung: Monate durchnummerieren (0, 1, 2, ...) -------------------
    df = df.sort_values([date_col, id_col]).reset_index(drop=True)
    dates = np.sort(df[date_col].unique())
    month_nr = np.searchsorted(dates, df[date_col].to_numpy())  # Nummer des Monats jeder Zeile
    first, last = test_period(dates, train_window, test_start, test_end)

    X = df[feature_cols].to_numpy(dtype=float)  # Charakteristika im Monat t
    y = df[target_col].to_numpy(dtype=float)    # Rendite im Monat t + 1

    # Ausgabetabelle: alle Aktien der Testmonate first ... last
    is_test_row = (month_nr >= first) & (month_nr <= last)
    output_cols = [c for c in (date_col, id_col, size_col, target_col) if c]
    predictions = df.loc[is_test_row, output_cols].reset_index(drop=True)
    importances = {}
    best_params = []
    param_grids = param_grids or {}

    for name, model in models.items():
        param_grid = param_grids.get(name)

        # --- Fall 1: feste Hyperparameter -------------------------------------------
        if not param_grid:
            forecast, importances[name] = average_of_monthly_models(
                model, X, y, month_nr, first, last, train_window, expanding, name, verbose)
            predictions[name] = forecast[is_test_row]
            continue

        # --- Fall 2: Grid Search - jede Kombination rechnen, pro Monat die beste wählen ---
        # Die Prognosen beginnen `validation_window` Monate früher, damit es schon im
        # ersten Testmonat Prognosefehler gibt, nach denen ausgewählt werden kann.
        val_first = max(first - validation_window, 1)
        if val_first >= first:
            raise ValueError("Grid search needs at least one month of data before the first forecast month.")
        combos = list(ParameterGrid(param_grid))
        forecasts, combo_importances = [], []
        for c, params in enumerate(combos):
            forecast, imp = average_of_monthly_models(
                clone(model).set_params(**params), X, y, month_nr, val_first, last, train_window,
                expanding, f"{name} {c + 1}/{len(combos)}", verbose)
            forecasts.append(forecast)
            combo_importances.append(imp)

        best = choose_params_each_month(forecasts, y, month_nr, val_first, first, last, validation_window)
        chosen = best.loc[month_nr[is_test_row]].to_numpy()  # gewählte Kombination für jede Testzeile
        predictions[name] = np.vstack(forecasts)[chosen, np.flatnonzero(is_test_row)]
        importances[name] = combo_importances[best.mode()[0]]  # der am häufigsten gewählten Kombination
        best_params += [{date_col: dates[t], "model": name, "params": combos[c]} for t, c in best.items()]

    # Feature Importances pro Modell, gemittelt über alle Monatsmodelle (OLS hat keine)
    mean_importances = pd.DataFrame(
        {name: np.mean(values, axis=0) for name, values in importances.items() if values},
        index=feature_cols)
    return predictions, mean_importances, pd.DataFrame(best_params)


def average_of_monthly_models(model, X, y, month_nr, first, last, train_window, expanding,
                              label="", verbose=True):
    """Kern des Verfahrens: ein Modell pro Monat trainieren und die Prognosen der Monatsmodelle mitteln.

    Monatsmodell j lernt den Zusammenhang zwischen den Charakteristika am Ende
    von Monat j und der Rendite im Monat j + 1. Es fließt in die Prognosen der
    Monate j + 1 bis j + `train_window` ein (bei `expanding` aller späteren
    Monate). Die Prognose für Monat t ist damit das Mittel der Modelle der
    Monate t - `train_window` bis t - 1. Deren Renditen sind in t alle bekannt:
    Es werden keine Zukunftsdaten verwendet.

    Rückgabe: (Prognose je Zeile von `X`, NaN außerhalb der Monate `first` ...
    `last`; Liste der Feature Importances aller Monatsmodelle).
    """
    forecast_sum = np.zeros(len(y))  # Summe der Prognosen je Aktie und Monat
    n_models = np.zeros(len(y))      # Anzahl der Monatsmodelle, die diese Prognose enthält
    importances = []

    first_model = 0 if expanding else max(first - train_window, 0)
    for j in range(first_model, last):
        # 1. Modell auf dem Querschnitt von Monat j trainieren (ca. 500 Aktien x 14 Charakteristika)
        in_month_j = month_nr == j
        fitted = clone(model).fit(X[in_month_j], y[in_month_j])

        # 2. Damit alle Aktien der Monate j + 1 ... j + train_window prognostizieren
        last_month = last if expanding else min(j + train_window, last)
        in_forecast_months = (month_nr > j) & (month_nr >= first) & (month_nr <= last_month)
        forecast_sum[in_forecast_months] += fitted.predict(X[in_forecast_months])
        n_models[in_forecast_months] += 1

        if hasattr(fitted, "feature_importances_"):
            importances.append(fitted.feature_importances_)
        if verbose:
            print(f"\r  {label:<13} month {j - first_model + 1}/{last - first_model}", end="", flush=True)
    if verbose:
        print()

    # 3. Mittelwert: Summe der Prognosen / Anzahl der Modelle
    with np.errstate(invalid="ignore"):  # 0 / 0 = NaN für Zeilen außerhalb des Prognosezeitraums
        return forecast_sum / n_models, importances


def choose_params_each_month(forecasts, y, month_nr, val_first, first, last, validation_window):
    """Wählt für jeden Prognosemonat t die Hyperparameter-Kombination mit dem kleinsten Fehler davor.

    `forecasts` enthält eine Prognosereihe pro Kombination (ab Monat `val_first`).
    Verglichen wird der mittlere quadratische Fehler (MSE) in den
    `validation_window` Monaten VOR t. Diese Prognosen sind selbst
    out-of-sample, und ihre Renditen sind in t bekannt - die Auswahl nutzt also
    keine Zukunftsdaten.

    Rückgabe: Series Monatsnummer -> Nummer der gewählten Kombination.
    """
    rows = (month_nr >= val_first) & (month_nr <= last)
    squared_errors = pd.DataFrame({c: (f[rows] - y[rows]) ** 2 for c, f in enumerate(forecasts)})
    by_month = squared_errors.groupby(month_nr[rows])

    # MSE über die letzten `validation_window` Monate; shift(1): nur Monate VOR t
    sse = by_month.sum().rolling(validation_window, min_periods=1).sum()
    count = by_month.size().rolling(validation_window, min_periods=1).sum()
    mse_before = sse.div(count, axis=0).shift(1)
    return mse_before.loc[first:last].idxmin(axis=1)


# =============================================================================
# Pooled-Schätzung (gestapelte Querschnitte) - Alternative zu Fama-MacBeth
# =============================================================================

def rolling_windows(dates, train_window, retrain_every=1, test_start=None, test_end=None, expanding=False):
    """Liefert (train_dates, test_dates) für ein Rolling Window über alle Datumswerte.

    Das Trainingsset enthält die `train_window` Monate direkt vor dem/den
    Testmonat(en) (bei `expanding` alle Monate seit Beginn der Daten, aber
    mindestens `train_window`); das Testset enthält die nächsten `retrain_every`
    Monate. Prognostiziert wird nur zwischen `test_start` und `test_end`.
    """
    dates = np.sort(pd.unique(dates))
    first, last = test_period(dates, train_window, test_start, test_end)
    for start in range(first, last + 1, retrain_every):
        train_from = 0 if expanding else start - train_window
        yield dates[train_from:start], dates[start:min(start + retrain_every, last + 1)]


def run_pooled(df, models, feature_cols, target_col, date_col, id_col, train_window,
               retrain_every=1, expanding=False, test_start=None, test_end=None, size_col=None,
               param_grids=None, validation_window=12, verbose=True):
    """Trainiert jedes Modell auf einem Rolling Window und prognostiziert den/die folgenden Monat(e).

    Pro Fenster werden die letzten `train_window` Querschnitte (bei `expanding`
    alle seit Beginn der Daten) zu einer Trainingsmatrix untereinandergestapelt
    (z.B. 120 Monate x 500 Aktien = 60.000 Zeilen). Ein einziges Modell lernt
    darauf und prognostiziert die Querschnitte der nächsten `retrain_every`
    Monate. Danach rückt das Fenster um `retrain_every` Monate weiter.

    Ist `param_grids` für ein Modell gesetzt, werden dessen Hyperparameter in
    jedem Fenster optimiert: Jede Kombination wird auf den früheren Monaten
    trainiert und auf den letzten `validation_window` Monaten des Fensters
    bewertet (MSE, zeitliche Aufteilung statt zufälliger Folds, damit nicht mit
    Zukunftsdaten validiert wird). Die beste Kombination wird anschließend auf
    allen Monaten des Fensters neu trainiert.

    Rückgabe wie `run_fama_macbeth`.
    """
    windows = list(rolling_windows(df[date_col], train_window, retrain_every, test_start, test_end, expanding))
    param_grids = param_grids or {}
    if param_grids and validation_window >= train_window:
        raise ValueError("validation_window must be smaller than train_window.")

    output_cols = [c for c in (date_col, id_col, size_col, target_col) if c]
    predictions = []
    importances = {name: [] for name in models}
    best_params = []

    for i, (train_dates, test_dates) in enumerate(windows, start=1):
        train = df[df[date_col].isin(train_dates)]
        test = df[df[date_col].isin(test_dates)]
        X_train, y_train = train[feature_cols], train[target_col]
        out = test[output_cols].copy()

        for name, model in models.items():
            # in jedem Fenster ein frisches Modell (clone), damit nichts aus früheren Fenstern übernommen wird
            if name in param_grids:
                # Validierung auf den letzten `validation_window` Monaten des Trainingsfensters
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
                  f"train {fmt_date(train_dates[0])} - {fmt_date(train_dates[-1])} "
                  f"X_train {X_train.shape[0]}x{X_train.shape[1]}  "
                  f"test {fmt_date(test_dates[0])}", end="", flush=True)
    if verbose:
        print()

    # Feature Importances pro Modell, gemittelt über alle trainierten Modelle (OLS hat keine)
    mean_importances = pd.DataFrame(
        {name: np.mean(values, axis=0) for name, values in importances.items() if values},
        index=feature_cols)
    return pd.concat(predictions, ignore_index=True), mean_importances, pd.DataFrame(best_params)
