"""
Masterprojekt - Prognose von Aktienrenditen mit baumbasierten Modellen.

Vorgehen wie bei Lewellen (2015), "The Cross-Section of Expected Stock Returns",
mit Machine-Learning-Modellen anstelle der linearen Regression:
    1. Daten: JKP-Daten laden (eine Zeile pro Monat und Aktie), pro Monat die
       500 größten Aktien mit Werten für alle Charakteristika
    2. Schätzung: in jedem Monat ein Modell auf dem Querschnitt trainieren
       (500 Aktien x 14 Charakteristika -> Rendite im Folgemonat); die erwartete
       Rendite einer Aktie ist das Mittel der Prognosen der Monatsmodelle der
       letzten 10 Jahre. Bei der linearen Regression (OLS) ist das genau das
       Verfahren aus dem Paper - OLS läuft daher als Benchmark mit.
    3. Das Modell aus dem Paper: Fama-MacBeth-Regressionen (Tabelle 2)
    4. Güte der Prognosen: Verteilung und Out-of-Sample-Regression der
       realisierten Renditen auf die Prognosen (Tabelle 3)
    5. Nach der Prognose sortierte Dezilportfolios (Tabelle 6)
    6. Handelsstrategie (Long Top 20 %) im Vergleich zu Equal Weight

Jeder Schritt ist eine Funktion unten in dieser Datei; main() ganz unten ruft
sie nacheinander auf. Die eigentliche Rechnung steht in:
    dataset.py     Schritt 1  (Daten laden, Stichprobe)
    models.py      Schritt 2  (Modelle, Fama-MacBeth-Schätzung)
    evaluation.py  Schritte 3-6  (Regressionen, Tabellen, Strategie)
    report.py      Ausgabe von Tabellen und Grafik
"""

import os

import pandas as pd

import dataset
import evaluation
import models
import report

# Ordner von main.py: alle Pfade beziehen sich darauf, egal von wo das Skript gestartet wird
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Einstellungen (die wichtigsten - weitere Details stehen direkt in den Schritten unten)
# =============================================================================

USE_DUMMY_DATA = False       # True: Zufallsdaten zum schnellen Testen der Pipeline
FEATURE_SET = "Model 3"      # Prädiktoren nach Lewellen (2015): "Model 1" (Größe, B/M, Momentum),
                             # "Model 2" (+ Emissionen, Accruals, Profitabilität, Asset Growth),
                             # "Model 3" (alle 14); None: alle Spalten außer den vier Spalten unten
UNIVERSE_SIZE = 500          # pro Monat nur die 500 größten Aktien verwenden (None: alle im Datensatz)

ESTIMATION = "fama_macbeth"  # "fama_macbeth": pro Monat ein Modell auf dem Querschnitt, Prognose = Mittel
                             #   der Monatsmodelle im Fenster (wie die gemittelten Steigungen im Paper)
                             # "pooled": ein Modell auf allen gestapelten Querschnitten des Fensters
WINDOW_TYPE = "rolling"      # "rolling": die letzten TRAIN_WINDOW Monate (Paper: Hauptergebnis)
                             # "expanding": alle Monate seit Beginn der Daten (Paper: "cumulative")
TRAIN_WINDOW = 120           # Fensterlänge in Monaten (Paper: 10 Jahre);
                             # bei "expanding" die Mindestlänge vor der ersten Prognose
TEST_START = "1974-01-01"    # erster Prognosemonat (Paper: 1974)
TEST_END = "2013-12-31"      # letzter Prognosemonat (Paper: 2013; None: bis zum Ende der Daten, 2024)

MODELS_TO_RUN = ["OLS", "DecisionTree", "RandomForest", "XGBoost"]
                             # "OLS" = Benchmark: mit ESTIMATION = "fama_macbeth" genau das Modell aus dem Paper
GRID_SEARCH = False          # True: Hyperparameter jeden Monat (bzw. in jedem Fenster) neu wählen (langsamer)

LONG_PCT = 0.20              # Kauf der besten 20 % der Aktien nach prognostizierter Rendite
SHORT_PCT = 0.0              # Leerverkauf der schlechtesten x % der Aktien (0.0 = kein Shorting)

SAVE_RESULTS = True          # Ergebnisse (CSV-Dateien und Grafik) im Ordner output speichern
SHOW_PLOT = True             # Grafik zusätzlich in einem Fenster anzeigen

# --- Feste Werte (normalerweise nicht ändern) ---------------------------------
DATE_COL = "eom"             # Monatsende
ID_COL = "id"                # JKP-ID der Aktie
TARGET_COL = "ret_exc_lead1m"  # Überrendite über den NÄCHSTEN Monat (das wird prognostiziert)
SIZE_COL = "me"              # Marktkapitalisierung (untransformiert): Universum und wertgewichtete Portfolios
TARGET_IS_EXCESS = True      # Zielgröße ist bereits eine Überrendite -> rf wird nicht nochmal abgezogen
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def check_settings():
    """Prüft die Einstellungen auf Kombinationen, die nicht zusammenpassen."""
    if ESTIMATION not in ("fama_macbeth", "pooled"):
        raise ValueError("ESTIMATION must be 'fama_macbeth' or 'pooled'.")
    if WINDOW_TYPE not in ("rolling", "expanding"):
        raise ValueError("WINDOW_TYPE must be 'rolling' or 'expanding'.")


def window_label():
    """Beschreibung des Schätzfensters für Überschriften und Grafik."""
    if WINDOW_TYPE == "rolling":
        return f"{TRAIN_WINDOW}-month rolling window"
    return f"expanding window (min. {TRAIN_WINDOW} months)"


# =============================================================================
# Ablauf
# =============================================================================

def load_sample():
    """Schritt 1: Daten laden und die Stichprobe bilden.

    Rückgabe: (Daten, Liste der Charakteristika, risikofreier Zins oder None).
    """
    report.print_section("1. Data")

    # Import-Einstellungen: pro Monat die `top_k` größten Aktien nach Marktkapitalisierung, mit ihren Daten
    # der `years_before` Jahre davor (max. Trainingszeitraum) und der `years_after` Jahre danach (max.
    # Prognosezeitraum). Charakteristika werden pro Monat transformiert: "rank" (wie Tutor), "z" oder "none".
    top_k, years_before, years_after, transform = 1000, 25, 5, "rank"
    if UNIVERSE_SIZE and not USE_DUMMY_DATA and UNIVERSE_SIZE > top_k:
        raise ValueError(f"UNIVERSE_SIZE must not be larger than the imported top_k ({top_k}).")
    data_path = os.path.join(BASE_DIR, "Thema 3 - Stock Return Prediction", "DATA", "USA.parquet")
    # vorgefilterte Daten (klein genug für Git): werden geladen, falls vorhanden, sonst einmalig aus
    # data_path erstellt. Der Dateiname enthält die Import-Einstellungen - ändert man diese, entsteht eine neue Datei.
    prepared_path = os.path.join(BASE_DIR, "data", f"jkp_top{top_k}_{years_before}y_{years_after}y_{transform}.csv.gz")

    # Stichprobe: erster Monat der Schätzung (Paper: 1964, Beginn der Compustat-Daten); wie im Paper nur
    # Aktien mit Werten für alle Charakteristika
    sample_start = "1964-01-01"
    risk_free_path = os.path.join(BASE_DIR, "data", "risk_free.csv")  # nur bei TARGET_IS_EXCESS = False:
                                 # CSV (date, rf pro Periode); fehlt die Datei, wird rf ignoriert

    # --- Daten laden (Dummy-Daten, vorbereitete CSV oder einmalig aus der Parquet-Datei bauen) ---
    if USE_DUMMY_DATA:
        df = dataset.make_dummy_data(seed=42, date_col=DATE_COL, id_col=ID_COL,
                                     target_col=TARGET_COL, size_col=SIZE_COL)
    elif os.path.exists(prepared_path):
        print(f"  Loading prepared data '{os.path.relpath(prepared_path, BASE_DIR)}'")
        df = dataset.load_prepared_data(prepared_path, DATE_COL)
    else:
        print(f"  Building prepared data from '{os.path.relpath(data_path, BASE_DIR)}' (about 1 minute) ...")
        df = dataset.load_data(data_path, top_k, years_before, years_after, transform)
        dataset.save_prepared_data(df, prepared_path)
        print(f"  Saved to '{os.path.relpath(prepared_path, BASE_DIR)}'")

    # --- Charakteristika wählen (die Dummy-Daten haben eigene Spaltennamen) ---
    use_feature_set = FEATURE_SET and not USE_DUMMY_DATA
    if use_feature_set:
        feature_cols = dataset.LEWELLEN_MODELS[FEATURE_SET]
    else:
        exclude = [DATE_COL, ID_COL, TARGET_COL, SIZE_COL]
        feature_cols = [c for c in df.columns if c not in exclude]

    # --- Stichprobe bilden (die Dummy-Daten haben eigene Zeiträume ab 2010, daher kein Startdatum) ---
    if USE_DUMMY_DATA:
        sample_start = None
    df = dataset.prepare_data(df, DATE_COL, ID_COL, TARGET_COL, feature_cols)
    print(f"  {len(df):,} rows | {df[ID_COL].nunique():,} stocks in total | {df[DATE_COL].nunique()} periods "
          f"({report.fmt_date(df[DATE_COL].min())} - {report.fmt_date(df[DATE_COL].max())})")
    df = dataset.select_sample(df, DATE_COL, SIZE_COL, feature_cols, sample_start, True, UNIVERSE_SIZE)
    print(f"  Sample from {report.fmt_date(df[DATE_COL].min())}: {len(feature_cols)} characteristics "
          f"({FEATURE_SET if use_feature_set else 'all columns'}), stocks with missing values removed")
    if UNIVERSE_SIZE:
        print(f"  Universe: {UNIVERSE_SIZE} largest stocks per month by '{SIZE_COL}'")

    # --- Risikofreier Zins (nur nötig, wenn die Zielgröße keine Überrendite ist) ---
    rf = None
    if TARGET_IS_EXCESS:
        print("  Risk-free rate: not needed - the target already is an excess return")
    else:
        rf = dataset.load_risk_free(risk_free_path, df[DATE_COL].unique())
        if rf is None:
            print(f"  Risk-free rate: not available ('{risk_free_path}') - Sharpe ratios use raw returns")
        else:
            print(f"  Risk-free rate: loaded from '{risk_free_path}' "
                  f"(mean {rf.mean() * 12:.2%} p.a.)")
    return df, feature_cols, rf


def estimate(df, feature_cols):
    """Schritt 2: Modelle schätzen und out-of-sample prognostizieren (Fama-MacBeth oder pooled).

    Rückgabe: (Prognosen, Feature Importances, gewählte Hyperparameter).
    """
    # Hyperparameter pro Modell (fest, solange GRID_SEARCH = False). Sie begrenzen, wie fein ein Baum die
    # Aktien aufteilen darf - je feiner, desto eher lernt er Zufallsrauschen statt echter Zusammenhänge:
    #   max_depth         maximale Anzahl Ja/Nein-Fragen hintereinander (Tiefe des Baums)
    #   min_samples_leaf  mindestens so viele Aktien pro Endknoten (Blatt); dessen Prognose ist ihre Durchschnittsrendite
    #   n_estimators      Anzahl der Bäume (Random Forest, XGBoost)
    #   max_features      Random Forest: Anteil der Charakteristika, aus denen bei jeder Frage gewählt wird
    #                     ("sqrt": Wurzel aus 14, also 3 zufällige) - macht die Bäume verschieden
    #   learning_rate     XGBoost: mit welchem Gewicht jeder neue Baum in die Prognose eingeht
    #   subsample, colsample_bytree  XGBoost: zufälliger Anteil der Aktien bzw. Charakteristika je Baum
    # random_state ist der Startwert des Zufallsgenerators: gleiche Ergebnisse bei jedem Lauf.
    random_state = 42
    model_params = {
        "OLS": {},
        "DecisionTree": {"max_depth": 5, "min_samples_leaf": 50},
        "RandomForest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50,
                         "max_features": "sqrt", "n_jobs": -1},
        "XGBoost": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1},
    }

    # Grid Search: zu testende Werte (alle übrigen Parameter kommen aus model_params). Verglichen wird auf den
    # `validation_window` Monaten vor dem Prognosemonat.
    validation_window = 12
    param_grids = {
        "DecisionTree": {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50, 100]},
        "RandomForest": {"max_depth": [3, 6], "max_features": ["sqrt", 0.5]},
        "XGBoost": {"max_depth": [2, 3, 5], "learning_rate": [0.01, 0.05, 0.1]},
    }

    # nur "pooled": Monate, die mit einem Modell prognostiziert werden, bevor neu trainiert wird
    retrain_every = 1

    # die Dummy-Daten haben eigene Zeiträume (ab 2010), daher keine Testgrenzen
    test_start = None if USE_DUMMY_DATA else TEST_START
    test_end = None if USE_DUMMY_DATA else TEST_END

    args = dict(
        models=models.get_models(MODELS_TO_RUN, model_params, random_state), feature_cols=feature_cols,
        target_col=TARGET_COL, date_col=DATE_COL, id_col=ID_COL, train_window=TRAIN_WINDOW,
        expanding=WINDOW_TYPE == "expanding", test_start=test_start, test_end=test_end, size_col=SIZE_COL,
        param_grids=param_grids if GRID_SEARCH else None, validation_window=validation_window,
    )
    grid = ", grid search" if GRID_SEARCH else ""
    if ESTIMATION == "fama_macbeth":
        report.print_section(f"2. Fama-MacBeth estimation: one model per month, "
                             f"forecasts averaged over a {window_label()}{grid}")
        preds, importances, best_params = models.run_fama_macbeth(df, **args)
    else:
        report.print_section(f"2. Pooled estimation: one model per {window_label()}{grid}")
        preds, importances, best_params = models.run_pooled(df, **args, retrain_every=retrain_every)

    print(f"  Forecasts: {report.fmt_date(preds[DATE_COL].min())} - {report.fmt_date(preds[DATE_COL].max())} | "
          f"{preds[DATE_COL].nunique()} months | {preds.groupby(DATE_COL).size().mean():.0f} stocks per month")
    if not best_params.empty:
        print("\n  Most frequently chosen hyperparameters:")
        for name, g in best_params.groupby("model"):
            counts = g["params"].astype(str).value_counts()
            print(f"  {name:<13} {counts.index[0]}  ({counts.iloc[0]} of {len(g)})")
    if not importances.empty:
        print("\n  Top 10 features (mean importance over all models of the tree methods):")
        report.print_table(importances.loc[importances.mean(axis=1).nlargest(10).index], decimals=4)
    return preds, importances, best_params


def paper_regressions(df, feature_cols, preds):
    """Schritt 3: das Modell aus dem Paper - Fama-MacBeth-Regressionen (Lewellen, Tabelle 2).

    Verwendet alle Monate, deren Regressionen in die Prognosen eingehen (Paper: 1964-2013).
    Rückgabe: (Tabelle der mittleren Koeffizienten, monatliche Koeffizienten).
    """
    report.print_section("3. Fama-MacBeth regressions - the model of the paper (Lewellen Table 2)")
    table, monthly = evaluation.fama_macbeth(df[df[DATE_COL] < preds[DATE_COL].max()], TARGET_COL,
                                             feature_cols, DATE_COL)
    table[["Coef", "S.E."]] *= 100  # in % pro Monat, wie im Paper
    report.print_table(table)
    print(f"  {len(monthly)} monthly regressions ({report.fmt_date(monthly.index[0])} - "
          f"{report.fmt_date(monthly.index[-1])}) | mean R2 {monthly['R2'].mean():.3f} | "
          f"mean N {monthly['N'].mean():.0f}\n"
          "  Coef in % per month (with rank characteristics: return difference between the lowest and\n"
          "  the highest ranked stock) | S.E. and t-stat with Newey-West standard errors")
    if ESTIMATION == "fama_macbeth" and "OLS" in MODELS_TO_RUN:
        print(f"  The OLS forecasts below average exactly these monthly coefficients over the {window_label()}.")
    return table, monthly


def evaluate_forecasts(preds):
    """Schritt 4: Prognosegüte (Lewellen, Tabelle 3) und weitere Kennzahlen.

    Rückgabe: (Tabelle 3, weitere Prognosekennzahlen, Korrelationen der Prognosen oder None).
    """
    report.print_section("4. Expected-return estimates (Lewellen Table 3)")
    table3 = pd.DataFrame({name: evaluation.forecast_evaluation(preds, name, TARGET_COL, DATE_COL)
                           for name in MODELS_TO_RUN}).T
    report.print_table(table3, report.EVALUATION_FORMATS)
    print("  Avg/Std/p10/p90: cross-sectional distribution of the forecasts (% per month)\n"
          "  Slope/S.E./t-stat/R2: out-of-sample FM regression of realized returns on the forecasts\n"
          "  (slope 1 = unbiased forecast, slope > 0 = the forecast separates high and low returns)")

    print("\n  Further prediction metrics:")
    metrics = pd.DataFrame({name: evaluation.prediction_metrics(preds, name, TARGET_COL, DATE_COL)
                            for name in MODELS_TO_RUN}).T
    report.print_table(metrics, decimals=4)

    correlations = None
    if len(MODELS_TO_RUN) > 1:
        correlations = evaluation.forecast_correlations(preds, MODELS_TO_RUN, DATE_COL)
        print("\n  Mean cross-sectional correlation of the forecasts (cf. Lewellen Table 4):")
        report.print_table(correlations)
    return table3, metrics, correlations


def portfolio_sorts(preds):
    """Schritt 5: nach der Prognose sortierte Portfolios (Lewellen, Tabelle 6). Rückgabe: Tabelle pro Modell."""
    n_portfolios = 10  # Anzahl der nach Prognose sortierten Portfolios (Paper: Dezile)
    report.print_section(f"5. Portfolios sorted by expected return (Lewellen Table 6, {n_portfolios} groups)")
    portfolios = {name: evaluation.sorted_portfolios(preds, name, TARGET_COL, DATE_COL, SIZE_COL, n_portfolios)
                  for name in MODELS_TO_RUN}
    for name, table in portfolios.items():
        print(f"\n  {name}")
        report.print_table(table)
    print("\n  EW/VW: equal-/value-weighted | Pred/Avg/Std in % per month | Sharpe annualized | "
          "t-stat with Newey-West SE")
    return portfolios


def trading_strategy(preds, rf):
    """Schritt 6: Handelsstrategie pro Modell im Vergleich zu Equal Weight.

    Rückgabe: (Renditen pro Monat und Strategie, Performance-Kennzahlen).
    """
    report.print_section(f"6. Trading strategy: {evaluation.strategy_label(LONG_PCT, SHORT_PCT)}")
    returns = pd.DataFrame({name: evaluation.strategy_returns(preds, name, TARGET_COL, DATE_COL,
                                                              LONG_PCT, SHORT_PCT)
                            for name in MODELS_TO_RUN})
    # Benchmark: alle Aktien des Universums gleichgewichtet
    returns["Equal Weight"] = evaluation.equal_weight_returns(preds, TARGET_COL, DATE_COL)

    # ein Long-Short-Portfolio ist selbstfinanzierend, seine Rendite ist daher bereits eine Überrendite
    zero_cost = LONG_PCT > 0 and SHORT_PCT > 0
    perf = pd.DataFrame({name: evaluation.performance_metrics(
                             returns[name], 12, rf=None if zero_cost and name in MODELS_TO_RUN else rf)
                         for name in returns}).T
    report.print_table(perf, report.PERFORMANCE_FORMATS)
    return returns, perf


def save_results(results):
    """Speichert alle Tabellen als CSV-Dateien in OUTPUT_DIR (leere und fehlende Tabellen entfallen)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, table in results.items():
        if table is not None and not table.empty:
            table.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"))
    print(f"\n  Results saved to '{OUTPUT_DIR}'")


def plot_strategy(returns, perf):
    """Grafik: Wertentwicklung der Strategien mit Kennzahlentabelle (gespeichert und/oder angezeigt)."""
    subtitle = [f"{evaluation.strategy_label(LONG_PCT, SHORT_PCT)} by predicted return"]
    if UNIVERSE_SIZE:
        subtitle.append(f"top {UNIVERSE_SIZE} stocks")
    subtitle.append(f"{'Fama-MacBeth' if ESTIMATION == 'fama_macbeth' else 'pooled'}, {window_label()}")
    if TARGET_IS_EXCESS:
        subtitle.append("excess returns")
    subtitle.append(f"{report.fmt_date(returns.index[0])} - {report.fmt_date(returns.index[-1])}")
    report.plot_performance(
        returns, perf,
        title="Strategy performance vs. equal weight",
        subtitle=" | ".join(subtitle),
        save_path=os.path.join(OUTPUT_DIR, "performance.png") if SAVE_RESULTS else None,
        show=SHOW_PLOT,
    )


def main():
    check_settings()
    df, feature_cols, rf = load_sample()                       # Schritt 1
    preds, importances, best_params = estimate(df, feature_cols)  # Schritt 2
    fm_table, fm_monthly = paper_regressions(df, feature_cols, preds)  # Schritt 3
    table3, metrics, correlations = evaluate_forecasts(preds)  # Schritt 4
    portfolios = portfolio_sorts(preds)                        # Schritt 5
    strategy, perf = trading_strategy(preds, rf)               # Schritt 6

    if SAVE_RESULTS:
        save_results({
            "predictions": preds.set_index([DATE_COL, ID_COL]),
            "feature_importance": importances,
            "best_params": best_params.set_index([DATE_COL, "model"]) if not best_params.empty else None,
            "fama_macbeth_regressions": fm_table,
            "fama_macbeth_monthly_coefficients": fm_monthly,
            "forecast_evaluation": table3,
            "prediction_metrics": metrics,
            "forecast_correlations": correlations,
            "sorted_portfolios": pd.concat(portfolios, names=["model", "portfolio"]),
            "strategy_returns": strategy,
            "performance": perf,
        })
    plot_strategy(strategy, perf)


if __name__ == "__main__":
    main()
