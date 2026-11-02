"""
Die vier Modelle und das Schätzverfahren von Lewellen (2015).

Alle Modelle haben dieselben zwei Befehle: `fit(X, y)` (trainieren) und
`predict(X)` (prognostizieren). Deshalb lassen sie sich im Verfahren unten
beliebig austauschen.
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Hyperparameter begrenzen, wie fein ein Baum die Aktien aufteilen darf - je feiner, desto eher
# lernt er Zufallsrauschen statt echter Zusammenhänge:
#   max_depth         maximale Anzahl Ja/Nein-Fragen hintereinander (Tiefe des Baums)
#   min_samples_leaf  mindestens so viele Aktien pro Endknoten (Blatt); dessen Prognose ist ihre Durchschnittsrendite
#   n_estimators      Anzahl der Bäume
#   max_features      Random Forest: bei jeder Frage nur aus einer zufälligen Auswahl der Charakteristika
#                     wählen ("sqrt": Wurzel aus 14, also 3) - macht die Bäume verschieden
#   learning_rate     XGBoost: mit welchem Gewicht jeder neue Baum in die Prognose eingeht
#   subsample, colsample_bytree  XGBoost: zufälliger Anteil der Aktien bzw. Charakteristika je Baum
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
}


def fama_macbeth_forecast(model, X, y, month_nr, first, last, window, label=""):
    """Kern des Verfahrens: ein Modell pro Monat trainieren und die Prognosen der Monatsmodelle mitteln.

    Lewellen schätzt jeden Monat eine Querschnittsregression, mittelt die
    Koeffizienten der letzten 10 Jahre und prognostiziert damit. Hier gibt jedes
    der Monatsmodelle eine eigene Prognose ab, und diese werden gemittelt - für
    eine Regression ist das exakt dasselbe, und es funktioniert genauso für Bäume.

    Monatsmodell j lernt den Zusammenhang zwischen den Charakteristika am Ende
    von Monat j und der Rendite im Monat j + 1. Die Prognose für Monat t ist das
    Mittel der Modelle der Monate t - `window` bis t - 1. Deren Renditen sind in
    t alle bekannt: Es werden keine Zukunftsdaten verwendet.

    `month_nr` ist die Nummer des Monats jeder Zeile von X (0 = erster Monat),
    `first` und `last` sind die Nummern des ersten und letzten Prognosemonats.
    Rückgabe: (Prognose je Zeile, NaN außerhalb der Prognosemonate;
    Feature Importances aller Monatsmodelle).
    """
    forecast_sum = np.zeros(len(y))  # Summe der Prognosen je Aktie und Monat
    n_models = np.zeros(len(y))      # Anzahl der Monatsmodelle, die diese Prognose enthält
    importances = []

    first_model = max(first - window, 0)
    for j in range(first_model, last):
        # 1. Modell auf dem Querschnitt von Monat j trainieren (ca. 500 Aktien x 14 Charakteristika)
        in_month_j = month_nr == j
        fitted = clone(model).fit(X[in_month_j], y[in_month_j])

        # 2. Damit alle Aktien der Monate j + 1 ... j + window prognostizieren
        in_forecast_months = (month_nr > j) & (month_nr >= first) & (month_nr <= min(j + window, last))
        forecast_sum[in_forecast_months] += fitted.predict(X[in_forecast_months])
        n_models[in_forecast_months] += 1

        if hasattr(fitted, "feature_importances_"):
            importances.append(fitted.feature_importances_)
        print(f"\r  {label:<13} month {j - first_model + 1}/{last - first_model}", end="", flush=True)
    print()

    # 3. Mittelwert: Summe der Prognosen / Anzahl der Modelle
    with np.errstate(invalid="ignore"):  # 0 / 0 = NaN für Zeilen außerhalb des Prognosezeitraums
        return forecast_sum / n_models, importances
