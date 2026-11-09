"""
Masterprojekt - Prognose von Aktienrenditen mit baumbasierten Modellen.

Replikation von Lewellen (2015), "The Cross-Section of Expected Stock Returns",
mit Machine-Learning-Modellen anstelle der linearen Regression. Das Skript läuft
von oben nach unten durch:
    1. Daten: pro Monat die 500 größten Aktien (oder alle "large stocks") mit Werten für alle Charakteristika
    2. Schätzung: in jedem Monat ein Modell auf dem Querschnitt trainieren; die
       erwartete Rendite ist das Mittel der Prognosen der Monatsmodelle der
       letzten 10 Jahre (für OLS genau das Verfahren des Papers)
    3. Fama-MacBeth-Regressionen (Tabelle 2)
    4. Güte der Prognosen (Tabelle 3)
    5. Nach der Prognose sortierte Dezilportfolios (Tabelle 6)
    6. Handelsstrategie (Long Top 20 %) im Vergleich zu Equal Weight
    7. Entwicklung über die Zeit (Abschnitt 3.6, Fig. 2) mit Test auf einen Rückgang
    8. Machine Forecast Disagreement (Bali, Kelly, Mörke & Rahman, 2026): wie
       stark sich 100 simulierte Investoren über eine Aktie uneinig sind, und
       ob das die Renditen vorhersagt
Ergebnisse (CSV-Dateien und Grafiken) landen in output/single bzw. output/stacked, jeweils aufgeteilt in
predicted_return/ (Schritte 3-7) und investor_disagreement/ (Schritt 8).

models.py enthält die Modelle und das Schätzverfahren, util.py die Rechenbausteine.
"""

import os

import numpy as np
import pandas as pd

import models
import util
from util import DATE, ID, RET, SIZE  # Spaltennamen: Monatsende, Aktie, Rendite im Folgemonat, Marktwert

# =============================================================================
# Einstellungen
# =============================================================================

FEATURE_SET = "Model 3"      # Prädiktoren nach Lewellen (2015): "Model 1" (Größe, B/M, Momentum),
                             # "Model 2" (+ Emissionen, Accruals, Profitabilität, Asset Growth), "Model 3" (alle 14)
UNIVERSE = "top500"          # "top500": pro Monat die UNIVERSE_SIZE größten Aktien
                             # "large": alle Stammaktien über dem NYSE-Median (Lewellens "large stocks", Tabelle 3
                             #   Panel C, ca. 300-1.000 pro Monat); braucht beim ersten Mal die JKP-Rohdaten
UNIVERSE_SIZE = 500          # nur "top500": Anzahl der größten Aktien pro Monat (max. 1000)
ESTIMATION = "single"        # "single": ein Modell pro Monat, Prognose = Mittel der Monatsmodelle (Lewellen,
                             #   Fama-MacBeth); "stacked": ein Modell auf allen gestapelten Monaten des
                             #   Fensters (Gu, Kelly & Xiu)
WINDOW_TYPE = "rolling"      # "rolling": die letzten TRAIN_WINDOW Monate (Paper: Hauptergebnis)
                             # "expanding": alle Monate seit 1964 (Paper: "cumulative")
TRAIN_WINDOW = 120           # Fensterlänge in Monaten (Paper: 10 Jahre); bei "expanding" die Mindestlänge
REFIT_EVERY = 1              # nur "stacked": Modell jeden Monat neu trainieren wie bei "single" (Gu, Kelly & Xiu: 12)
TUNE_HYPERPARAMETERS = False # Hyperparameter der Bäume einmal im Jahr auf einer Validierung wählen (Gu, Kelly & Xiu),
                             #   in beiden Verfahren gleich; False: Standardwerte der Paper aus models.MODELS
                             #   (Ergebnisordner siehe unten, vor Schritt 2)
TARGET = "return"            # Zielgröße der Bäume: "return" (Rendite, wie Gu, Kelly & Xiu) oder "rank" (Rang der Rendite
                             #   im Monat auf [-0.5, 0.5]); OLS lernt immer die Rendite (Lewellen)
VALIDATION_SHARE = 0.3       # Anteil des Fensters für die Validierung: die jüngsten 30 % (bei 120 Monaten: 36),
                             #   die übrigen 70 % (84 Monate) zum Training
RETUNE_EVERY = 12            # alle wie viele Monate die Hyperparameter neu gewählt werden (Paper: jedes Jahr)
TEST_START = "1974-01-01"    # erster Prognosemonat (Paper: 1974)
TEST_END = "2013-12-31"      # letzter Prognosemonat (Paper: 2013; None: bis zum Ende der Daten, 2024)
MODELS_TO_RUN = ["OLS", "DecisionTree", "RandomForest", "XGBoost", "Investors"]
                             # "Investors": 100 simulierte Investoren, liefert die Disagreement für Schritt 8
LONG_PCT = 0.10              # Strategien: Kauf der besten 10 % der Aktien (höchste Prognose bzw. geringste Disagreement)
SHORT_PCT = 0.10             #   und Leerverkauf der schlechtesten 10 %; gerechnet werden Long, Short und beides
ROLLING_MONTHS = 12          # Schritt 7: gleitender Mittelwert (zusätzlich immer 120 Monate wie in Fig. 2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. Daten
# =============================================================================
util.print_section("1. Data")

# Vorbereitete Daten (eine Zeile pro Monat und Aktie, Charakteristika bereits in Ränge umgerechnet).
# Fehlt die Datei, wird sie einmalig aus den 6-GB-Rohdaten des Tutors erzeugt
# (Thema 3 - Stock Return Prediction/DATA/USA.parquet, nicht im Repository).
prepared_file = {"top500": "jkp_top1000_25y_5y_rank.csv.gz", "large": "jkp_large_rank.csv.gz"}[UNIVERSE]
prepared_path = os.path.join(BASE_DIR, "data", prepared_file)
if not os.path.exists(prepared_path):
    raw_path = os.path.join(BASE_DIR, "Thema 3 - Stock Return Prediction", "DATA", "USA.parquet")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"'{prepared_file}' is missing; to build it, put the raw JKP data at '{raw_path}'")
    print(f"  Building '{prepared_file}' from the raw JKP data (a few minutes) ...")
    util.build_prepared_data(raw_path, prepared_path, universe="large" if UNIVERSE == "large" else "top1000")
df = pd.read_csv(prepared_path, parse_dates=[DATE])
features = util.LEWELLEN_MODELS[FEATURE_SET]

# Stichprobe wie im Paper: ab 1964, nur Monate mit is_included = 1 (die übrigen Zeilen der Datei sind
# nur die Historie der Aktien), mit Rendite und allen Charakteristika; bei "top500" davon pro Monat die
# 500 größten. Die Ränge der Datei gelten für alle Aktien und werden innerhalb der Stichprobe neu gebildet.
df = df.sort_values([DATE, ID])
df = df[(df[DATE] >= "1964-01-01") & (df["is_included"] == 1)].dropna(subset=[RET, *features])
if UNIVERSE == "top500":
    size_rank = df.groupby(DATE)[SIZE].rank(ascending=False, method="first")  # 1 = größte Aktie
    df = df[size_rank <= UNIVERSE_SIZE]
df = df.reset_index(drop=True)
df[util.JKP_CHARACTERISTICS] = util.rank_characteristics(df)
print(f"  {len(df):,} rows | {df[ID].nunique():,} stocks | {df[DATE].nunique()} months "
      f"({df[DATE].min():%Y-%m} - {df[DATE].max():%Y-%m}) | {len(features)} characteristics ({FEATURE_SET})")


# =============================================================================
# 2. Schätzung: Fama-MacBeth (Mittel der Monatsmodelle) oder gestapelt (ein Modell auf dem ganzen Fenster)
# =============================================================================
window_label = f"{TRAIN_WINDOW}-month rolling window" if WINDOW_TYPE == "rolling" else "expanding window"
if ESTIMATION == "single":
    method_label = f"single: one model per month, forecasts averaged over a {window_label}"
else:
    method_label = f"stacked: one model on a {window_label}, refit every {REFIT_EVERY} months"
util.print_section(f"2. Estimation ({method_label})")

# Monate durchnummerieren (0 = Januar 1964) und Nummer des ersten und letzten Prognosemonats bestimmen.
# Vor der ersten Prognose müssen TRAIN_WINDOW Monate liegen.
dates = np.sort(df[DATE].unique())
month_nr = np.searchsorted(dates, df[DATE].to_numpy())
first = max(TRAIN_WINDOW, np.searchsorted(dates, np.datetime64(TEST_START)))
last = len(dates) - 1 if TEST_END is None else np.searchsorted(dates, np.datetime64(TEST_END), side="right") - 1

# Ergebnisordner: output/results/<Hyperparameter>/<Zielgröße>/<Prognosejahre>/<Universum>/<Schätzverfahren>
#   Hyperparameter: "grid" (jährlich auf der Validierung gewählt) oder "fixed_leaf<min_samples_leaf>" (feste Werte)
#   z. B. output/results/fixed_leaf25/return/1974-2013/top500/single
# darin je ein Ordner für die beiden Signale:
#   predicted_return/       Auswertung der Renditeprognosen wie im Paper (Tabellen 2, 3, 6, Strategie, Zeitverlauf)
#   investor_disagreement/  dieselbe Auswertung für die Disagreement der Investoren (Schritt 8)
setup = "grid" if TUNE_HYPERPARAMETERS else f"fixed_leaf{models.MODELS['DecisionTree'].min_samples_leaf}"
years = f"{pd.Timestamp(dates[first]):%Y}-{pd.Timestamp(dates[last]):%Y}"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "results", setup, TARGET, years, UNIVERSE, ESTIMATION)
PRED_DIR = os.path.join(OUTPUT_DIR, "predicted_return")
MFD_DIR = os.path.join(OUTPUT_DIR, "investor_disagreement")
for folder in [PRED_DIR, MFD_DIR]:
    os.makedirs(folder, exist_ok=True)

X = df[features].to_numpy(dtype=float)  # Charakteristika am Ende von Monat t
y = df[RET].to_numpy(dtype=float)       # Rendite im Monat t + 1
# Zielgröße der Bäume: die Rendite selbst oder ihr Rang im Monat. Ausgewertet wird immer mit der Rendite.
y_trees = util.rank_returns(df) if TARGET == "rank" else y
print(f"  Target of the trees: {'rank of the return within the month' if TARGET == 'rank' else 'return'}"
      " (OLS: return)")

# Ergebnistabelle: eine Zeile pro Prognosemonat und Aktie, eine Prognosespalte pro Modell
is_test_row = (month_nr >= first) & (month_nr <= last)
preds = df.loc[is_test_row, [DATE, ID, SIZE, RET]].reset_index(drop=True)
estimate = models.fama_macbeth_forecast if ESTIMATION == "single" else models.stacked_forecast
tuned_estimate = models.tuned_fama_macbeth_forecast if ESTIMATION == "single" else models.tuned_stacked_forecast
options = {"refit_every": REFIT_EVERY} if ESTIMATION == "stacked" else {}
importances, chosen_hyperparameters = {}, []
for name in MODELS_TO_RUN:
    target = y if name == "OLS" else y_trees
    if TUNE_HYPERPARAMETERS and name in models.GRIDS:
        # Hyperparameter jedes Jahr auf den jüngsten VALIDATION_SHARE des Fensters wählen (in Monaten gerundet)
        forecast, model_importances, chosen = tuned_estimate(
            models.MODELS[name], models.GRIDS[name], X, target, month_nr, first, last, TRAIN_WINDOW,
            validation=round(VALIDATION_SHARE * TRAIN_WINDOW), retune_every=RETUNE_EVERY,
            expanding=WINDOW_TYPE == "expanding",
            label=name, **options)
        chosen_hyperparameters += [{"model": name, "first_forecast_month": pd.Timestamp(dates[row["first_month_nr"]]),
                                    **row} for row in chosen]
    else:
        forecast, model_importances = estimate(models.MODELS[name], X, target, month_nr, first, last, TRAIN_WINDOW,
                                               expanding=WINDOW_TYPE == "expanding", label=name, **options)
    if name == "Investors":
        # eine Prognose pro Investor: deren Mittel ist die Prognose, deren Streuung die Disagreement
        investor_forecasts = forecast[:, is_test_row]
        preds[name] = investor_forecasts.mean(axis=0)
        preds["MFD"] = investor_forecasts.std(axis=0)  # Standardabweichung über die 100 Investoren
    else:
        preds[name] = forecast[is_test_row]
    if model_importances:  # OLS und Investors haben keine Feature Importances
        importances[name] = np.mean(model_importances, axis=0)
importances = pd.DataFrame(importances, index=features)
if chosen_hyperparameters:
    # gewählte Hyperparameter je Modell und Jahr; Übersicht: wie oft jede Kombination gewählt wurde
    chosen_hyperparameters = pd.DataFrame(chosen_hyperparameters).drop(columns="first_month_nr")
    chosen_hyperparameters.to_csv(os.path.join(PRED_DIR, "hyperparameters.csv"), index=False)
    print("\n  Chosen hyperparameters (number of years, chosen on the validation months):")
    for name, table in chosen_hyperparameters.groupby("model", sort=False):
        params = [c for c in table.columns if c not in ("model", "first_forecast_month", "validation_mse",
                                                       "validation_mse_worst") and table[c].notna().any()]
        table = table.astype({c: int for c in params if (table[c] % 1 == 0).all()})  # 100.0 -> 100
        counts = table.groupby(params).size().sort_values(ascending=False)
        print(f"  {name:<13} " + " | ".join(f"{dict(zip(params, k if isinstance(k, tuple) else (k,)))}: {n}"
                                            for k, n in counts.items()))

print(f"  Forecasts: {preds[DATE].min():%Y-%m} - {preds[DATE].max():%Y-%m} | {preds[DATE].nunique()} months | "
      f"{preds.groupby(DATE).size().mean():.0f} stocks per month")
print("\n  Top 10 features (mean importance over all estimated models):")
util.print_table(importances.loc[importances.mean(axis=1).nlargest(10).index], decimals=4)


# =============================================================================
# 3. Fama-MacBeth-Regressionen - das Modell aus dem Paper (Tabelle 2)
# =============================================================================
util.print_section("3. Fama-MacBeth regressions (Lewellen Table 2)")

# alle Monate, deren Regressionen in die Prognosen eingehen
table2, fm_monthly = util.fama_macbeth(df[df[DATE] < preds[DATE].max()], RET, features)
table2[["Coef", "S.E."]] *= 100  # in % pro Monat, wie im Paper
util.print_table(table2)
print(f"  {len(fm_monthly)} monthly regressions | mean R2 {fm_monthly['R2'].mean():.3f}\n"
      "  Coef in % per month: return difference between the lowest and the highest ranked stock\n"
      "  t-stat with Newey-West standard errors (4 lags)")


# =============================================================================
# 4. Güte der Prognosen (Tabelle 3)
# =============================================================================
util.print_section("4. Expected-return estimates (Lewellen Table 3)")

table3 = pd.DataFrame({name: util.forecast_evaluation(preds, name) for name in MODELS_TO_RUN}).T
util.print_table(table3, decimals=3)
print("  Avg/Std/p10/p90: distribution of the forecasts (% per month)\n"
      "  Slope: FM regression of realized returns on the forecasts (1 = unbiased, > 0 = useful)")

metrics = pd.DataFrame({name: util.prediction_metrics(preds, name) for name in MODELS_TO_RUN}).T
print("\n  Further prediction metrics:")
util.print_table(metrics, decimals=4)

# mittlere Korrelation der Prognosen zweier Modelle im Querschnitt (vgl. Lewellen, Tabelle 4)
correlations = preds.groupby(DATE)[MODELS_TO_RUN].corr().groupby(level=1).mean().loc[MODELS_TO_RUN, MODELS_TO_RUN]
print("\n  Mean cross-sectional correlation of the forecasts:")
util.print_table(correlations)


# =============================================================================
# 5. Dezilportfolios (Tabelle 6)
# =============================================================================
util.print_section("5. Portfolios sorted by expected return (Lewellen Table 6)")

portfolios = {name: util.sorted_portfolios(preds, name) for name in MODELS_TO_RUN}
for name, table in portfolios.items():
    print(f"\n  {name}")
    util.print_table(table)
print("\n  EW/VW: equal-/value-weighted | Pred/Avg/Std in % per month | Sharpe annualized | Newey-West t-stat")


# =============================================================================
# 6. Handelsstrategien: Long, Short und Long-Short gegen Equal Weight
# =============================================================================
util.print_section(f"6. Trading strategies: long top {LONG_PCT:.0%}, short bottom {SHORT_PCT:.0%}, both")

# Name: (Anteil long, Anteil short). Long-Short ist selbstfinanzierend: der Leerverkauf bezahlt den Kauf.
STRATEGIES = {"Long": (LONG_PCT, 0.0), "Short": (0.0, SHORT_PCT), "Long-Short": (LONG_PCT, SHORT_PCT)}
strategy = util.backtest(preds, MODELS_TO_RUN, STRATEGIES)  # Spalten: (Strategie, Modell) plus Equal Weight
performance = pd.DataFrame({column: util.performance_metrics(strategy[column]) for column in strategy}).T
util.print_table(performance, decimals=4)


# =============================================================================
# 7. Entwicklung über die Zeit (Lewellen, Abschnitt 3.6 und Fig. 2)
# =============================================================================
util.print_section("7. Evolution through time (Lewellen Section 3.6 / Fig. 2)")

# Vier Kennzahlen für jeden Monat einzeln:
# - Steigung der Regression realisierte Rendite auf Prognose (ihr Mittelwert ist die Steigung aus Tabelle 3)
# - Rendite High-Dezil minus Low-Dezil, gleichgewichtet (vgl. Tabelle 6)
# - Long-Seite: Rendite der gekauften Aktien minus Equal Weight
# - Short-Seite: Equal Weight minus Rendite der leerverkauften Aktien (wie viel schlechter sie als der Markt waren;
#   die Short-Rendite ist minus deren Rendite, daher Equal Weight + Short)
equal_weight = strategy[("Benchmark", "Equal Weight")]
series = {
    "Predictive slope": pd.DataFrame(
        {name: util.fama_macbeth(preds, RET, [name])[1][name] for name in MODELS_TO_RUN}),
    "H-L return EW (%)": pd.DataFrame(
        {name: util.portfolio_returns(preds, name)[1]["H-L"] for name in MODELS_TO_RUN}) * 100,
    "Long minus Equal Weight (%)": strategy["Long"].sub(equal_weight, axis=0) * 100,
    "Equal Weight minus shorted stocks (%)": strategy["Short"].add(equal_weight, axis=0) * 100,
}

# Test auf einen Rückgang: zwei gleich lange Hälften (Standard: 1974-1993 und 1994-2013) und linearer Trend
months = series["Predictive slope"].index
split_date = months[len(months) // 2]
trends = pd.concat({label: pd.DataFrame({name: util.trend_test(monthly[name], split_date) for name in monthly}).T
                    for label, monthly in series.items()}, names=["series", "model"])
util.print_table(trends)
print(f"  2nd half from {split_date:%Y-%m} | Difference: 2nd minus 1st half | Trend: change per 10 years\n"
      "  t-stats with Newey-West SE; a negative value with |t| > 2 indicates a significant decline")


# =============================================================================
# Ergebnisse der Renditeprognosen speichern: output/<ESTIMATION>/predicted_return
# =============================================================================
pd.DataFrame(preds.set_index([DATE, ID])).to_csv(os.path.join(OUTPUT_DIR, "predictions.csv.gz"))  # alle Prognosen
results = {
    "feature_importance.csv": importances,
    "fama_macbeth_regressions.csv": table2,
    "fama_macbeth_monthly_coefficients.csv": fm_monthly,
    "forecast_evaluation.csv": table3,
    "prediction_metrics.csv": metrics,
    "forecast_correlations.csv": correlations,
    "sorted_portfolios.csv": pd.concat(portfolios, names=["model", "portfolio"]),
    "strategy_returns.csv": strategy,
    "performance.csv": performance,
    "time_variation_monthly.csv": pd.concat(series, axis=1),
    "time_variation_trends.csv": trends,
}
for filename, table in results.items():
    table.to_csv(os.path.join(PRED_DIR, filename))

universe_label = f"top {UNIVERSE_SIZE} stocks" if UNIVERSE == "top500" else f"{UNIVERSE} stocks"
period = (f"{preds[DATE].min():%Y} - {preds[DATE].max():%Y}, {universe_label}, {ESTIMATION}, {WINDOW_TYPE}"
          + (", trees on return ranks" if TARGET == "rank" else ""))
for window in [ROLLING_MONTHS, 120]:
    util.plot_time_variation(series, window, f"Predictive ability through time, {window}-month rolling mean "
                             f"({period})", os.path.join(PRED_DIR, f"time_variation_{window}m.png"))
print(f"\n  Results saved to '{PRED_DIR}'")


# =============================================================================
# 8. Machine Forecast Disagreement (Bali, Kelly, Mörke & Rahman, 2026)
#    Dieselbe Auswertung wie in den Schritten 4-7, nur mit der Disagreement statt der Prognose.
#    Das Paper findet: je größer die Disagreement, desto NIEDRIGER die Rendite im Folgemonat.
#    Deshalb wird hier umgekehrt sortiert: High = geringste Disagreement, Low = höchste Disagreement.
#    Ein positives H-L heißt also: der Effekt aus dem Paper ist da.
# =============================================================================
if "MFD" in preds:
    util.print_section("8. Machine forecast disagreement (MFD)")

    # 8a. Verteilung und Fama-MacBeth-Regression (wie Tabelle 3): Verteilung der MFD im Querschnitt
    #     (in % pro Monat) und Regression der Renditen auf die MFD, allein und zusammen mit der Prognose
    mfd_distribution = preds.groupby(DATE)["MFD"].describe(percentiles=[0.1, 0.5, 0.9]).mean() * 100
    mfd_regressions = pd.concat({
        "MFD only": util.fama_macbeth(preds, RET, ["MFD"])[0],
        "MFD + forecast": util.fama_macbeth(preds, RET, ["MFD", "Investors"])[0],
    }, names=["regression", "variable"])
    print(f"  MFD in % per month: mean {mfd_distribution['mean']:.3f} | p10 {mfd_distribution['10%']:.3f} | "
          f"median {mfd_distribution['50%']:.3f} | p90 {mfd_distribution['90%']:.3f}")
    print("\n  Fama-MacBeth regressions of returns on MFD (paper: negative MFD slope):")
    util.print_table(mfd_regressions, decimals=3)

    # 8b. Dezilportfolios nach der Disagreement (wie Tabelle 6)
    mfd_portfolios = util.sorted_portfolios(preds, "MFD", high_is_good=False, value_label="MFD (%)")
    print("\n  Deciles sorted by MFD: High = lowest disagreement, Low = highest disagreement")
    util.print_table(mfd_portfolios)

    # 8c. Strategien (wie Schritt 6): Long = Kauf der 10 % mit der geringsten Disagreement,
    #     Short = Leerverkauf der 10 % mit der höchsten Disagreement (laut Paper überbewertet), Long-Short = beides
    mfd_strategy = util.backtest(preds, ["MFD"], STRATEGIES, high_is_good=False)
    mfd_performance = pd.DataFrame({column: util.performance_metrics(mfd_strategy[column])
                                    for column in mfd_strategy}).T
    print(f"\n  Strategies: long the {LONG_PCT:.0%} with the lowest, short the {SHORT_PCT:.0%} with the highest "
          "disagreement")
    util.print_table(mfd_performance, decimals=4)

    # 8d. Entwicklung über die Zeit (wie Schritt 7): H-L-Rendite, Long- und Short-Seite gegen Equal Weight pro Monat
    mfd_series = {
        "H-L return EW (%)": util.portfolio_returns(preds, "MFD", high_is_good=False)[1][["H-L"]]
                             .rename(columns={"H-L": "MFD"}) * 100,
        "Long minus Equal Weight (%)": mfd_strategy["Long"].sub(equal_weight, axis=0) * 100,
        "Equal Weight minus shorted stocks (%)": mfd_strategy["Short"].add(equal_weight, axis=0) * 100,
    }
    mfd_trends = pd.concat({label: pd.DataFrame({name: util.trend_test(monthly[name], split_date)
                                                 for name in monthly}).T
                            for label, monthly in mfd_series.items()}, names=["series", "signal"])
    print("\n  Evolution through time (trend tests as in step 7):")
    util.print_table(mfd_trends)

    # Speichern: output/<ESTIMATION>/investor_disagreement
    mfd_results = {
        "disagreement_distribution.csv": mfd_distribution.to_frame("MFD (%)"),
        "fama_macbeth_regressions.csv": mfd_regressions,
        "sorted_portfolios.csv": mfd_portfolios,
        "strategy_returns.csv": mfd_strategy,
        "performance.csv": mfd_performance,
        "time_variation_monthly.csv": pd.concat(mfd_series, axis=1),
        "time_variation_trends.csv": mfd_trends,
    }
    for filename, table in mfd_results.items():
        table.to_csv(os.path.join(MFD_DIR, filename))
    util.plot_performance(mfd_strategy.droplevel(1, axis=1), f"Strategies sorted by disagreement vs. equal weight "
                          f"({period})", os.path.join(MFD_DIR, "performance.png"))
    for window in [ROLLING_MONTHS, 120]:
        util.plot_time_variation(mfd_series, window, f"Disagreement effect through time, {window}-month rolling "
                                 f"mean ({period})", os.path.join(MFD_DIR, f"time_variation_{window}m.png"))
    print(f"\n  Results saved to '{MFD_DIR}'")

# Grafiken der Strategien (Schritt 6 und 8), eine pro Strategie: die Modelle ohne Investors (deren Prognose
# steht in den Tabellen), dazu die Strategie nach der Disagreement (MFD) und Equal Weight
for label in STRATEGIES:
    lines = strategy[label].drop(columns="Investors", errors="ignore")
    if "Investors" in MODELS_TO_RUN:
        lines = lines.assign(MFD=mfd_strategy[(label, "MFD")])
    util.plot_performance(lines.assign(**{"Equal Weight": equal_weight}),
                          f"{label}: models and disagreement (MFD) vs. equal weight ({period})",
                          os.path.join(PRED_DIR, f"performance_{label.lower()}.png"))
