"""
Masterprojekt - Prognose von Aktienrenditen mit baumbasierten Modellen.

Replikation von Lewellen (2015), "The Cross-Section of Expected Stock Returns",
mit Machine-Learning-Modellen anstelle der linearen Regression. Das Skript läuft
von oben nach unten durch:
    1. Daten: pro Monat die 500 größten Aktien mit Werten für alle Charakteristika
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
Ergebnisse (CSV-Dateien und Grafiken) landen im Ordner output/.

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
UNIVERSE_SIZE = 500          # pro Monat nur die 500 größten Aktien (max. 1000)
ESTIMATION = "fama_macbeth"  # "fama_macbeth": ein Modell pro Monat, Prognose = Mittel der Monatsmodelle (Lewellen)
                             # "stacked": ein Modell auf allen gestapelten Monaten des Fensters (Gu, Kelly & Xiu)
WINDOW_TYPE = "rolling"      # "rolling": die letzten TRAIN_WINDOW Monate (Paper: Hauptergebnis)
                             # "expanding": alle Monate seit 1964 (Paper: "cumulative")
TRAIN_WINDOW = 120           # Fensterlänge in Monaten (Paper: 10 Jahre); bei "expanding" die Mindestlänge
REFIT_EVERY = 12             # nur "stacked": Modell alle 12 Monate neu trainieren (wie Gu, Kelly & Xiu)
TEST_START = "1974-01-01"    # erster Prognosemonat (Paper: 1974)
TEST_END = "2013-12-31"      # letzter Prognosemonat (Paper: 2013; None: bis zum Ende der Daten, 2024)
MODELS_TO_RUN = ["OLS", "DecisionTree", "RandomForest", "XGBoost", "Investors"]
                             # "Investors": 100 simulierte Investoren, liefert die Disagreement für Schritt 8
LONG_PCT = 0.20              # Strategie: Kauf der besten 20 % der Aktien nach prognostizierter Rendite
ROLLING_MONTHS = 12          # Schritt 7: gleitender Mittelwert (zusätzlich immer 120 Monate wie in Fig. 2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. Daten
# =============================================================================
util.print_section("1. Data")

# Vorbereitete Daten (eine Zeile pro Monat und Aktie, Charakteristika bereits in Ränge umgerechnet).
# Fehlt die Datei, wird sie einmalig aus den 6-GB-Rohdaten des Tutors erzeugt.
prepared_path = os.path.join(BASE_DIR, "data", "jkp_top1000_25y_5y_rank.csv.gz")
if not os.path.exists(prepared_path):
    print("  Building prepared data from the raw JKP data (about 1 minute) ...")
    util.build_prepared_data(os.path.join(BASE_DIR, "Thema 3 - Stock Return Prediction", "DATA", "USA.parquet"),
                             prepared_path)
df = pd.read_csv(prepared_path, parse_dates=[DATE])
features = util.LEWELLEN_MODELS[FEATURE_SET]

# Stichprobe wie im Paper: ab 1964, nur Aktien mit Rendite und allen Charakteristika,
# davon pro Monat die 500 größten nach Marktkapitalisierung
df = df.sort_values([DATE, ID])
df = df[df[DATE] >= "1964-01-01"].dropna(subset=[RET, *features])
size_rank = df.groupby(DATE)[SIZE].rank(ascending=False, method="first")  # 1 = größte Aktie
df = df[size_rank <= UNIVERSE_SIZE].reset_index(drop=True)
print(f"  {len(df):,} rows | {df[ID].nunique():,} stocks | {df[DATE].nunique()} months "
      f"({df[DATE].min():%Y-%m} - {df[DATE].max():%Y-%m}) | {len(features)} characteristics ({FEATURE_SET})")


# =============================================================================
# 2. Schätzung: Fama-MacBeth (Mittel der Monatsmodelle) oder gestapelt (ein Modell auf dem ganzen Fenster)
# =============================================================================
window_label = f"{TRAIN_WINDOW}-month rolling window" if WINDOW_TYPE == "rolling" else "expanding window"
if ESTIMATION == "fama_macbeth":
    method_label = f"Fama-MacBeth: one model per month, forecasts averaged over a {window_label}"
else:
    method_label = f"stacked: one model on a {window_label}, refit every {REFIT_EVERY} months"
util.print_section(f"2. Estimation ({method_label})")

# Monate durchnummerieren (0 = Januar 1964) und Nummer des ersten und letzten Prognosemonats bestimmen.
# Vor der ersten Prognose müssen TRAIN_WINDOW Monate liegen.
dates = np.sort(df[DATE].unique())
month_nr = np.searchsorted(dates, df[DATE].to_numpy())
first = max(TRAIN_WINDOW, np.searchsorted(dates, np.datetime64(TEST_START)))
last = len(dates) - 1 if TEST_END is None else np.searchsorted(dates, np.datetime64(TEST_END), side="right") - 1

X = df[features].to_numpy(dtype=float)  # Charakteristika am Ende von Monat t
y = df[RET].to_numpy(dtype=float)       # Rendite im Monat t + 1

# Ergebnistabelle: eine Zeile pro Prognosemonat und Aktie, eine Prognosespalte pro Modell
is_test_row = (month_nr >= first) & (month_nr <= last)
preds = df.loc[is_test_row, [DATE, ID, SIZE, RET]].reset_index(drop=True)
estimate = models.fama_macbeth_forecast if ESTIMATION == "fama_macbeth" else models.stacked_forecast
options = {"refit_every": REFIT_EVERY} if ESTIMATION == "stacked" else {}
importances = {}
for name in MODELS_TO_RUN:
    forecast, model_importances = estimate(models.MODELS[name], X, y, month_nr, first, last, TRAIN_WINDOW,
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
# 6. Handelsstrategie: Long Top 20 % gegen Equal Weight
# =============================================================================
util.print_section(f"6. Trading strategy: long top {LONG_PCT:.0%}")

strategy = pd.DataFrame({name: util.strategy_returns(preds, name, LONG_PCT) for name in MODELS_TO_RUN})
strategy["Equal Weight"] = preds.groupby(DATE)[RET].mean()  # Benchmark: alle 500 Aktien gleichgewichtet
performance = pd.DataFrame({name: util.performance_metrics(strategy[name]) for name in strategy}).T
util.print_table(performance, decimals=4)


# =============================================================================
# 7. Entwicklung über die Zeit (Lewellen, Abschnitt 3.6 und Fig. 2)
# =============================================================================
util.print_section("7. Evolution through time (Lewellen Section 3.6 / Fig. 2)")

# Drei Kennzahlen für jeden Monat einzeln:
# - Steigung der Regression realisierte Rendite auf Prognose (ihr Mittelwert ist die Steigung aus Tabelle 3)
# - Rendite High-Dezil minus Low-Dezil, gleichgewichtet (vgl. Tabelle 6)
# - Rendite der Strategie aus Schritt 6 minus Equal Weight
series = {
    "Predictive slope": pd.DataFrame(
        {name: util.fama_macbeth(preds, RET, [name])[1][name] for name in MODELS_TO_RUN}),
    "H-L return EW (%)": pd.DataFrame(
        {name: util.portfolio_returns(preds, name)[1]["H-L"] for name in MODELS_TO_RUN}) * 100,
    "Strategy minus Equal Weight (%)": strategy[MODELS_TO_RUN].sub(strategy["Equal Weight"], axis=0) * 100,
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
# 8. Machine Forecast Disagreement (Bali, Kelly, Mörke & Rahman, 2026)
# =============================================================================
mfd_tables = {}
if "MFD" in preds:
    util.print_section("8. Machine forecast disagreement (MFD)")

    # Verteilung der MFD im Querschnitt, über die Monate gemittelt (in % pro Monat)
    mfd_distribution = preds.groupby(DATE)["MFD"].describe(percentiles=[0.1, 0.5, 0.9]).mean() * 100
    print(f"  MFD in % per month: mean {mfd_distribution['mean']:.3f} | p10 {mfd_distribution['10%']:.3f} | "
          f"median {mfd_distribution['50%']:.3f} | p90 {mfd_distribution['90%']:.3f}")

    # Dezilportfolios nach MFD: Low = Aktien, über die sich die Investoren am meisten einig sind.
    # Das Paper findet: hohe Disagreement -> niedrige Rendite, also H-L < 0 (Paper: Low minus High > 0).
    mfd_tables["mfd_sorted_portfolios"] = util.sorted_portfolios(preds, "MFD")
    print("\n  Deciles sorted by MFD (Pred = mean MFD in %):")
    util.print_table(mfd_tables["mfd_sorted_portfolios"])

    # Fama-MacBeth-Regression der Renditen auf die MFD, allein und zusammen mit der mittleren Prognose
    mfd_tables["mfd_regressions"] = pd.concat({
        "MFD only": util.fama_macbeth(preds, RET, ["MFD"])[0],
        "MFD + forecast": util.fama_macbeth(preds, RET, ["MFD", "Investors"])[0],
    }, names=["regression", "variable"])
    print("\n  Fama-MacBeth regressions of returns on MFD (Newey-West t-stats):")
    util.print_table(mfd_tables["mfd_regressions"], decimals=3)
    print("  A negative MFD slope means: the more the investors disagree, the lower the next-month return")


# =============================================================================
# Ergebnisse speichern
# =============================================================================
results = {
    "predictions.csv.gz": preds.set_index([DATE, ID]),  # komprimiert, eine Zeile pro Monat und Aktie
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
    **{f"{name}.csv": table for name, table in mfd_tables.items()},
}
for filename, table in results.items():
    table.to_csv(os.path.join(OUTPUT_DIR, filename))

period = f"{preds[DATE].min():%Y} - {preds[DATE].max():%Y}, top {UNIVERSE_SIZE} stocks, {ESTIMATION}, {WINDOW_TYPE}"
util.plot_performance(strategy, f"Long top {LONG_PCT:.0%} by predicted return vs. equal weight ({period})",
                      os.path.join(OUTPUT_DIR, "performance.png"))
for window in [ROLLING_MONTHS, 120]:
    util.plot_time_variation(series, window, f"Predictive ability through time, {window}-month rolling mean "
                             f"({period})", os.path.join(OUTPUT_DIR, f"time_variation_{window}m.png"))
print(f"\n  Results saved to '{OUTPUT_DIR}'")
