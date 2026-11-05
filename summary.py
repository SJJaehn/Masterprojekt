"""
Zusammenfassung aller Läufe in `output/`: eine CSV und Vergleichsgrafiken.

Liest die Ergebnisse, die `main.py` in `output/<Lauf>/` gespeichert hat - es
wird nichts neu geschätzt. Vorher muss `main.py` für die gewünschten
Einstellungen gelaufen sein; fehlende Läufe werden übersprungen.

Ausgabe in `output/summary/`:
- summary.csv                        eine Zeile pro Lauf und Signal (Modell oder MFD)
- hl_single_vs_stacked.png           H-L-Rendite single gegen stacked, mit 95-%-Intervall
- strategy_return_sharpe.png         Rendite p. a. und Sharpe Ratio von Long, Short, Long-Short
- wealth_long-short_<Universum>.png  Wertentwicklung Long-Short, single gegen stacked
- rolling_hl_<Universum>.png         gleitender 10-Jahres-Mittelwert der H-L-Rendite
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")

# Läufe: Ordner in output/ -> (Universum, Schätzverfahren, Prognosezeitraum)
RUNS = {
    "single":            ("500 largest", "single",  "1974-2013"),
    "stacked":           ("500 largest", "stacked", "1974-2013"),
    "single_large":      ("large stocks", "single",  "1974-2013"),
    "stacked_large":     ("large stocks", "stacked", "1974-2013"),
    "single_until_2024": ("500 largest", "single",  "1974-2024"),
}
SIGNALS = ["OLS", "DecisionTree", "RandomForest", "XGBoost", "Investors", "MFD"]
STRATEGIES = ["Long", "Short", "Long-Short"]
UNIVERSES = ["500 largest", "large stocks"]
ESTIMATION_COLORS = {"single": util.COLORS[0], "stacked": util.COLORS[1]}  # Farbe = Schätzverfahren
ROLLING_MONTHS = 120  # wie Lewellens Fig. 2

RUNS = {name: info for name, info in RUNS.items() if os.path.isdir(os.path.join(OUTPUT_DIR, name))}
os.makedirs(SUMMARY_DIR, exist_ok=True)
print(f"Runs found: {', '.join(RUNS)}")


def read(run, folder, file, header=0, index_col=0):
    """Eine CSV aus output/<run>/<folder>/ lesen."""
    path = os.path.join(OUTPUT_DIR, run, folder, file)
    return pd.read_csv(path, header=header, index_col=index_col)


def style(ax):
    ax.grid(axis="y", alpha=0.3)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


# =============================================================================
# 1. Kennzahlen aller Läufe einsammeln
# =============================================================================
# Für jedes Signal: Prognosegüte (Tabelle 3), H-L-Portfolio (Tabelle 6) und die
# drei Strategien (Schritt 6 bzw. 8). Die MFD hat keine Prognose-Steigung und
# keinen IC; dort steht stattdessen die Steigung der FM-Regression auf die MFD.
rows = []
monthly_hl = {}   # (Lauf, Signal) -> Monatsreihe der H-L-Rendite in %
strategy = {}     # Lauf -> Monatsrenditen aller Strategien (Spalten: Strategie, Signal)

for run, (universe, estimation, period) in RUNS.items():
    table3 = read(run, "predicted_return", "forecast_evaluation.csv")
    metrics = read(run, "predicted_return", "prediction_metrics.csv")
    deciles = {
        "predicted_return": read(run, "predicted_return", "sorted_portfolios.csv", header=[0, 1], index_col=[0, 1]),
        "investor_disagreement": read(run, "investor_disagreement", "sorted_portfolios.csv", header=[0, 1]),
    }
    performance = pd.concat([read(run, "predicted_return", "performance.csv", index_col=[0, 1]),
                             read(run, "investor_disagreement", "performance.csv", index_col=[0, 1]).drop("Benchmark", level=0)])
    mfd_regression = read(run, "investor_disagreement", "fama_macbeth_regressions.csv", index_col=[0, 1])
    equal_weight = performance.loc[("Benchmark", "Equal Weight")]

    for signal in SIGNALS:
        row = {"run": run, "universe": universe, "estimation": estimation, "period": period, "signal": signal}

        # Prognosegüte
        if signal == "MFD":
            row["Slope"] = mfd_regression.loc[("MFD only", "MFD"), "Coef"]
            row["Slope t-stat"] = mfd_regression.loc[("MFD only", "MFD"), "t-stat"]
        else:
            row["Slope"] = table3.loc[signal, "Slope"]
            row["Slope t-stat"] = table3.loc[signal, "t-stat"]
            row["IC mean"] = metrics.loc[signal, "IC mean"]
            row["IC t-stat"] = metrics.loc[signal, "IC t-stat"]

        # H-L-Portfolio aus den Dezilen (MFD: High = geringste Disagreement)
        hl = deciles["investor_disagreement"].loc["H-L"] if signal == "MFD" else deciles["predicted_return"].loc[(signal, "H-L")]
        for weighting in ["EW", "VW"]:
            row[f"H-L {weighting} (%)"] = hl[(weighting, "Avg (%)")]
            row[f"H-L {weighting} t-stat"] = hl[(weighting, "t-stat")]
            row[f"H-L {weighting} Sharpe"] = hl[(weighting, "Sharpe")]

        # Strategien
        for name in STRATEGIES:
            for metric in ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"]:
                row[f"{name} {metric}"] = performance.loc[(name, signal), metric]
        row["Equal Weight Ann. Return"] = equal_weight["Ann. Return"]
        row["Equal Weight Sharpe Ratio"] = equal_weight["Sharpe Ratio"]
        rows.append(row)

    # Monatsreihen für die Grafiken
    tv = read(run, "predicted_return", "time_variation_monthly.csv", header=[0, 1])
    tv_mfd = read(run, "investor_disagreement", "time_variation_monthly.csv", header=[0, 1])
    for signal in SIGNALS:
        source = tv_mfd if signal == "MFD" else tv
        monthly_hl[(run, signal)] = source[("H-L return EW (%)", signal)].rename(index=pd.to_datetime)
    strategy[run] = pd.concat([read(run, "predicted_return", "strategy_returns.csv", header=[0, 1]),
                               read(run, "investor_disagreement", "strategy_returns.csv", header=[0, 1]).drop(
                                   columns="Benchmark", level=0)], axis=1).rename(index=pd.to_datetime)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(SUMMARY_DIR, "summary.csv"), index=False)

util.print_section("Summary: H-L return (EW, % per month) and Long-Short Sharpe ratio")
for column, label in [("H-L EW (%)", "H-L EW (%)"), ("H-L EW t-stat", "t-stat"), ("Long-Short Sharpe Ratio", "Long-Short Sharpe")]:
    print(f"\n  {label}")
    util.print_table(summary.pivot(index="signal", columns="run", values=column).loc[SIGNALS, list(RUNS)])


# =============================================================================
# 2. H-L-Rendite: single gegen stacked
# =============================================================================
# Balken = mittlere H-L-Rendite, Strich = 95-%-Intervall (± 1,96 Standardfehler,
# Standardfehler = Mittel / t-Statistik). Ein Intervall, das die Null nicht
# schneidet, entspricht |t| > 1,96.
main_period = summary[summary["period"] == "1974-2013"]
universes = [u for u in UNIVERSES if u in set(main_period["universe"])]

fig, axes = plt.subplots(1, len(universes), figsize=(6.5 * len(universes), 4.8), sharey=True, squeeze=False)
x = np.arange(len(SIGNALS))
width = 0.38
for ax, universe in zip(axes[0], universes):
    for k, estimation in enumerate(["single", "stacked"]):
        data = main_period[(main_period["universe"] == universe) & (main_period["estimation"] == estimation)]
        if data.empty:
            continue
        data = data.set_index("signal").loc[SIGNALS]
        error = 1.96 * data["H-L EW (%)"] / data["H-L EW t-stat"]
        ax.bar(x + (k - 0.5) * width, data["H-L EW (%)"], width * 0.92, yerr=error, capsize=3,
               color=ESTIMATION_COLORS[estimation], error_kw={"elinewidth": 1, "ecolor": "#52514e"}, label=estimation)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x, SIGNALS, rotation=20)
    ax.set_title(universe, loc="left", fontsize=11)
    style(ax)
axes[0][0].set_ylabel("H-L return EW (% per month)")
axes[0][0].legend(frameon=False, loc="upper right")
fig.suptitle("High-minus-low decile return, 1974-2013 (bars: mean, lines: 95% interval)", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(SUMMARY_DIR, "hl_single_vs_stacked.png"), dpi=150)
plt.close(fig)


# =============================================================================
# 3. Realisierte Rendite und Sharpe Ratio der Strategien
# =============================================================================
# Eine Zeile pro Strategie, links die Rendite p. a., rechts die Sharpe Ratio.
# Die gestrichelte Linie ist Equal Weight des jeweiligen Universums.
groups = [(u, e) for u in universes for e in ["single", "stacked"]
          if not main_period[(main_period["universe"] == u) & (main_period["estimation"] == e)].empty]
width = 0.8 / len(groups)
shades = {"500 largest": 1.0, "large stocks": 0.5}  # large stocks heller

fig, axes = plt.subplots(len(STRATEGIES), 2, figsize=(14, 3.6 * len(STRATEGIES)), sharex=True)
for i, name in enumerate(STRATEGIES):
    for j, (metric, label) in enumerate([("Ann. Return", "Return p. a. (%)"), ("Sharpe Ratio", "Sharpe ratio")]):
        ax = axes[i, j]
        scale = 100 if metric == "Ann. Return" else 1
        for k, (universe, estimation) in enumerate(groups):
            data = main_period[(main_period["universe"] == universe) & (main_period["estimation"] == estimation)]
            data = data.set_index("signal").loc[SIGNALS]
            offset = (k - (len(groups) - 1) / 2) * width
            ax.bar(x + offset, data[f"{name} {metric}"] * scale, width * 0.92, color=ESTIMATION_COLORS[estimation],
                   alpha=shades[universe], label=f"{estimation}, {universe}")
            if name == "Long":  # nur bei Long ist Equal Weight der direkte Vergleich
                ew_metric = "Ann. Return" if metric == "Ann. Return" else "Sharpe Ratio"
                ax.axhline(data[f"Equal Weight {ew_metric}"].iloc[0] * scale, color=util.BENCHMARK_COLOR,
                           lw=1, ls="--" if universe == "500 largest" else ":")
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(f"{name}: {label}", loc="left", fontsize=11)
        ax.set_xticks(x, SIGNALS, rotation=20)
        style(ax)
axes[0, 0].legend(frameon=False, ncol=2, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.12))
fig.suptitle("Strategies, 1974-2013: long top 10%, short bottom 10%, both "
             "(dashed / dotted: Equal Weight, 500 largest / large stocks)", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(SUMMARY_DIR, "strategy_return_sharpe.png"), dpi=150)
plt.close(fig)


# =============================================================================
# 4. Zeitreihen je Universum: Wertentwicklung und gleitender H-L-Mittelwert
# =============================================================================
# Ein Teildiagramm pro Signal, darin single und stacked.
for universe in universes:
    runs = {RUNS[run][1]: run for run in RUNS if RUNS[run][0] == universe and RUNS[run][2] == "1974-2013"}
    suffix = universe.split()[0] if universe != "500 largest" else "top500"

    for kind in ["wealth", "rolling"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, sharey=kind == "rolling")
        for ax, signal in zip(axes.flat, SIGNALS):
            for estimation, run in runs.items():
                if kind == "wealth":
                    line = (1 + strategy[run][("Long-Short", signal)]).cumprod()
                else:
                    line = monthly_hl[(run, signal)].rolling(ROLLING_MONTHS).mean()
                ax.plot(line.index, line, lw=1.6, color=ESTIMATION_COLORS[estimation], label=estimation)
            if kind == "wealth":
                benchmark = (1 + strategy[next(iter(runs.values()))][("Benchmark", "Equal Weight")]).cumprod()
                ax.plot(benchmark.index, benchmark, lw=1.2, color=util.BENCHMARK_COLOR, label="Equal Weight")
                ax.set_yscale("log")
            else:
                ax.axhline(0, color="grey", lw=1)
            ax.set_title(signal, loc="left", fontsize=11)
            style(ax)
        axes[0, 0].legend(frameon=False, fontsize=9)
        if kind == "wealth":
            title, file = f"Long-Short, value of 1 invested (log scale), {universe}", f"wealth_long-short_{suffix}.png"
        else:
            title, file = (f"H-L return EW, {ROLLING_MONTHS}-month rolling mean (% per month), {universe}",
                           f"rolling_hl_{suffix}.png")
        fig.suptitle(title, x=0.01, ha="left", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(SUMMARY_DIR, file), dpi=150)
        plt.close(fig)

print(f"\n  Results saved to '{SUMMARY_DIR}'")
