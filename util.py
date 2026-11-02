"""
Werkzeugkasten für main.py: Datenaufbereitung, Statistik (Fama-MacBeth,
Newey-West), Portfolios, Kennzahlen und Grafiken.

Jede Funktion wird in main.py für jedes Modell aufgerufen; deshalb stehen die
Rechnungen hier und nicht direkt im Skript. Alle Renditen sind Monatsrenditen
als Dezimalzahl (0.01 = 1 %).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Spalten des Datensatzes
DATE, ID, RET, SIZE = "eom", "id", "ret_exc_lead1m", "me"

# Charakteristika aus dem Tutor-Skript b00PrepareDataset.m (Replikation von Lewellen, 2015, Modell 3).
# Rechts jeweils die Variable bei Lewellen, die die JKP-Spalte annähert.
JKP_CHARACTERISTICS = [
    "market_equity",   # Größe (log. Marktkapitalisierung)
    "be_me",           # Buchwert / Marktwert (B/M)
    "ret_12_1",        # Momentum: Rendite von Monat t-12 bis t-2
    "chcsho_12m",      # Aktienemissionen: Veränderung der Aktienanzahl über 12 Monate
    "oaccruals_at",    # Accruals: operative Accruals / Bilanzsumme
    "ni_at",           # Profitabilität: Return on Assets
    "at_gr1",          # Asset Growth: Wachstum der Bilanzsumme
    "beta_60m",        # Beta aus 60 Monatsrenditen
    "div12m_me",       # Dividendenrendite der letzten 12 Monate
    "ret_60_12",       # Langfristige Rendite: Monat t-60 bis t-12
    "rvol_252d",       # Volatilität der Tagesrenditen (1 Jahr)
    "turnover_126d",   # Turnover (Handelsvolumen / Aktienanzahl, 6 Monate)
    "debt_me",         # Verschuldung: Schulden / Marktwert
    "sale_me",         # Umsatz / Marktwert
]

# Prädiktoren der drei Modelle bei Lewellen (2015), Tabelle 2
LEWELLEN_MODELS = {
    "Model 1": JKP_CHARACTERISTICS[:3],   # Größe, B/M, Momentum
    "Model 2": JKP_CHARACTERISTICS[:7],   # + Emissionen, Accruals, Profitabilität, Asset Growth
    "Model 3": JKP_CHARACTERISTICS,       # + Beta, Dividendenrendite, Langfristrendite, Volatilität,
                                          #   Turnover, Verschuldung, Umsatz/Preis
}


# =============================================================================
# Daten
# =============================================================================

def build_prepared_data(raw_path, prepared_path, top_k=1000, years_before=25, years_after=5):
    """Erzeugt aus den JKP-Rohdaten (USA.parquet, 6 GB) die kleine CSV in data/.

    Wird nur gebraucht, wenn die CSV fehlt. Analog zu b00PrepareDataset.m und b01LoadDataset.m:
    1. Nur die benötigten Spalten lesen (plus `me` als untransformierte Marktkapitalisierung)
    2. Zeilen ohne Rendite oder Marktkapitalisierung sowie nano/micro Stocks entfernen
    3. Charakteristika pro Monat in Ränge umrechnen, skaliert auf [-0.5, 0.5] (wie fCrossSectTransChars.m)
    4. Nur Aktien behalten, die irgendwann zu den `top_k` größten gehören, und von
       diesen die Daten der `years_before` Jahre davor und der `years_after` Jahre
       danach - das macht die Datei klein genug für Git
    """
    # 1. Spalten lesen
    df = pd.read_parquet(raw_path, columns=["id", "eom", "size_grp", "ret_exc_lead1m", "me",
                                            *JKP_CHARACTERISTICS])

    # 2. Zeilen filtern, Datum umwandeln
    df = df[df["ret_exc_lead1m"].notna() & df["me"].notna() & ~df["size_grp"].isin(["nano", "micro"])]
    df = df.drop(columns="size_grp")
    df["eom"] = pd.to_datetime(df["eom"].astype(str), format="%Y%m%d")

    # 3. Rangtransformation pro Monat (fehlende Werte bleiben fehlend)
    ranks = df.groupby("eom")[JKP_CHARACTERISTICS].rank(method="average")  # gleiche Werte: mittlerer Rang
    df[JKP_CHARACTERISTICS] = (ranks - 1) / (ranks.groupby(df["eom"]).transform("max") - 1) - 0.5

    # 4. Aktien behalten, die in einem Monat t zu den `top_k` größten gehören:
    #    alle ihre Zeilen von t - months_before bis t + months_after Monaten
    months_before, months_after = 12 * years_before, 12 * years_after
    month = (df["eom"].dt.year * 12 + df["eom"].dt.month).to_numpy()  # fortlaufender Monatsindex
    is_top = (df.groupby("eom")["me"].rank(ascending=False, method="first") <= top_k).to_numpy()

    rows = pd.DataFrame({"pos": np.arange(len(df)), "id": df["id"].to_numpy(), "month": month})
    rows = rows.sort_values("month")
    tops = pd.DataFrame({"id": df["id"].to_numpy()[is_top], "top_month": month[is_top]})
    tops = tops.sort_values("top_month")

    # für jede Zeile: nächster Top-Monat derselben Aktie ab bzw. letzter Top-Monat bis zu diesem Monat
    next_top = pd.merge_asof(rows, tops, left_on="month", right_on="top_month", by="id", direction="forward")
    prev_top = pd.merge_asof(rows, tops, left_on="month", right_on="top_month", by="id", direction="backward")
    keep = ((next_top["top_month"] - next_top["month"] <= months_before)
            | (prev_top["month"] - prev_top["top_month"] <= months_after)).to_numpy()
    df = df.iloc[np.sort(rows["pos"].to_numpy()[keep])]

    # Speichern, gerundet auf 6 signifikante Stellen (bleibt unter dem 100-MB-Limit von GitHub)
    os.makedirs(os.path.dirname(prepared_path), exist_ok=True)
    df.sort_values(["eom", "id"]).to_csv(prepared_path, index=False, float_format="%.6g")


# =============================================================================
# Statistik
# =============================================================================

def newey_west_test(series, lags=4):
    """Mittelwert einer Zeitreihe mit Newey-West-Standardfehler und t-Statistik.

    Die Regression auf eine Konstante schätzt genau den Mittelwert; die
    HAC-Kovarianz berücksichtigt Autokorrelation bis `lags` Monate (Paper: 4).
    Rückgabe: (Mittelwert, Standardfehler, t-Statistik).
    """
    y = pd.Series(series).dropna().to_numpy()
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0], fit.bse[0], fit.tvalues[0]


def fama_macbeth(df, y_col, x_cols):
    """Fama-MacBeth-Regression wie im Paper.

    1. Jeden Monat eine Querschnittsregression y = a + b'x (OLS)
    2. Koeffizienten über alle Monate mitteln, t-Statistik mit Newey-West-Standardfehler

    Rückgabe: (Tabelle mit Coef, S.E. und t-stat pro Variable, monatliche
    Koeffizienten mit R2 und Anzahl der Aktien N).
    """
    # 1. Querschnittsregression pro Monat
    monthly = []
    for date, g in df.groupby(DATE):
        fit = sm.OLS(g[y_col], sm.add_constant(g[x_cols])).fit()
        monthly.append({DATE: date, **fit.params.rename({"const": "Intercept"}),
                        "R2": fit.rsquared, "N": int(fit.nobs)})
    monthly = pd.DataFrame(monthly).set_index(DATE)

    # 2. Mittelwert und Newey-West-t-Statistik der monatlichen Koeffizienten
    coefs = monthly.drop(columns=["R2", "N"])
    table = pd.DataFrame([newey_west_test(coefs[c]) for c in coefs],
                         index=coefs.columns, columns=["Coef", "S.E.", "t-stat"])
    return table, monthly


def forecast_evaluation(preds, model):
    """Verteilung und Prognosegüte der erwarteten Renditen eines Modells (Lewellen, Tabelle 3).

    - Avg, Std, p10, p90: Mittelwert, Standardabweichung, 10 %- und 90 %-Quantil
      der Prognosen im Querschnitt (in % pro Monat), gemittelt über alle Monate
    - Slope, S.E., t-stat, R2: Fama-MacBeth-Regression der realisierten Renditen
      auf die Prognosen. Eine unverzerrte Prognose hat eine Steigung von 1;
      > 0 heißt, die Prognose trennt Aktien mit hoher und niedriger Rendite.
    """
    dist = preds.groupby(DATE)[model].describe(percentiles=[0.1, 0.9]).mean() * 100
    table, monthly = fama_macbeth(preds, RET, [model])
    slope = table.loc[model]
    return {
        "Avg (%)": dist["mean"], "Std (%)": dist["std"], "p10 (%)": dist["10%"], "p90 (%)": dist["90%"],
        "Slope": slope["Coef"], "S.E.": slope["S.E."], "t-stat": slope["t-stat"], "R2": monthly["R2"].mean(),
    }


def prediction_metrics(preds, model):
    """Weitere Kennzahlen aus der Machine-Learning-Literatur.

    - RMSE / MAE : durchschnittlicher Prognosefehler
    - R2 OOS     : 1 - SSE / sum(y^2), Vergleich mit einer Prognose von null (Gu, Kelly & Xiu, 2020)
    - IC         : Information Coefficient, Rangkorrelation zwischen Prognose und
                   realisierter Rendite im Querschnitt, pro Monat, dann gemittelt
    - Hit rate   : Anteil der Prognosen mit dem richtigen Vorzeichen
    """
    y_true, y_pred = preds[RET], preds[model]
    ic = preds.groupby(DATE).apply(lambda g: g[RET].corr(g[model], method="spearman"), include_groups=False)
    return {
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "R2 OOS": 1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2),
        "IC mean": ic.mean(),
        "IC IR": ic.mean() / ic.std(),
        "IC t-stat": stats.ttest_1samp(ic, 0).statistic,
        "Hit rate": np.mean(np.sign(y_pred) == np.sign(y_true)),
    }


def trend_test(series, split_date):
    """Gibt es einen Rückgang? Zwei Tests für eine monatliche Reihe, beide mit Newey-West (4 Lags).

    1. Hälftenvergleich: Regression auf eine Dummy-Variable, die ab `split_date` 1 ist;
       der Koeffizient ist die Differenz der Mittelwerte (zweite minus erste Hälfte)
    2. Linearer Trend: Regression auf die Zeit in Jahrzehnten; der Koeffizient ist
       die Veränderung des Mittelwerts pro 10 Jahre
    Ein negativer Wert mit |t| > 2 spricht für einen signifikanten Rückgang.
    """
    y = series.dropna()
    second_half = (y.index >= pd.Timestamp(split_date)).astype(float)
    decades = (y.index - y.index[0]).days / 3652.5
    hac = {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}

    halves = sm.OLS(y.to_numpy(), sm.add_constant(second_half)).fit(**hac)
    trend = sm.OLS(y.to_numpy(), sm.add_constant(np.asarray(decades))).fit(**hac)
    return {
        "Mean 1st half": y[second_half == 0].mean(),
        "Mean 2nd half": y[second_half == 1].mean(),
        "Difference": halves.params[1],
        "t-stat diff": halves.tvalues[1],
        "Trend per decade": trend.params[1],
        "t-stat trend": trend.tvalues[1],
    }


# =============================================================================
# Portfolios und Handelsstrategie
# =============================================================================

def portfolio_returns(preds, model, n_groups=10, value_weighted=False):
    """Monatliche Prognose und Rendite der nach der Prognose sortierten Portfolios.

    Jeden Monat werden die Aktien nach ihrer Prognose in `n_groups` gleich große
    Portfolios eingeteilt (1 = niedrigste Prognose); gleichgewichtet oder nach
    Marktkapitalisierung gewichtet. Rückgabe: (Prognosen, Renditen), je eine
    Tabelle Monate x Portfolios plus Spalte H-L (High minus Low).
    """
    # Portfolio 1 ... n_groups pro Monat (rank: gleiche Prognosen werden auf die Portfolios verteilt)
    df = preds.copy()
    df["portfolio"] = df.groupby(DATE)[model].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False) + 1)
    grouped = df.groupby([DATE, "portfolio"])

    # mittlere Prognose und mittlere Rendite je Monat und Portfolio
    if value_weighted:
        means = grouped.apply(lambda g: pd.Series(
            {c: np.average(g[c], weights=g[SIZE]) for c in (model, RET)}), include_groups=False)
    else:
        means = grouped[[model, RET]].mean()
    pred, ret = means[model].unstack(), means[RET].unstack()  # Monate x Portfolios
    pred["H-L"] = pred[n_groups] - pred[1]
    ret["H-L"] = ret[n_groups] - ret[1]
    return pred, ret


def sorted_portfolios(preds, model, n_groups=10):
    """Nach der Prognose sortierte Portfolios (Lewellen, Tabelle 6).

    Pro Portfolio, gleich- (EW) und wertgewichtet (VW):
    Pred (mittlere Prognose), Avg (mittlere realisierte Rendite), Std (in % pro
    Monat), annualisierte Sharpe Ratio und Newey-West-t-Statistik.
    """
    tables = {}
    for weighting, value_weighted in [("EW", False), ("VW", True)]:
        pred, ret = portfolio_returns(preds, model, n_groups, value_weighted)
        tables[weighting] = pd.DataFrame({
            "Pred (%)": pred.mean() * 100,
            "Avg (%)": ret.mean() * 100,
            "Std (%)": ret.std() * 100,
            "Sharpe": ret.mean() / ret.std() * np.sqrt(12),
            "t-stat": [newey_west_test(ret[c])[2] for c in ret],
        })
    table = pd.concat(tables, axis=1).rename(index={1: "Low (L)", n_groups: "High (H)"})
    table.index = table.index.astype(str)
    return table


def strategy_returns(preds, model, long_pct):
    """Rendite pro Monat: gleichgewichteter Kauf der `long_pct` Aktien mit der höchsten Prognose."""
    returns = {}
    for date, g in preds.groupby(DATE):
        # realisierte Renditen, sortiert von der höchsten zur niedrigsten Prognose
        realised = g.sort_values(model, ascending=False, kind="stable")[RET]
        n_long = max(1, int(round(len(realised) * long_pct)))  # mindestens 1 Aktie
        returns[date] = realised.iloc[:n_long].mean()
    return pd.Series(returns, name=model)


def performance_metrics(returns):
    """Annualisierte Performance einer Reihe von Monatsrenditen (die bereits Überrenditen sind)."""
    returns = returns.dropna()
    wealth = (1 + returns).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)  # das Startkapital zählt als Höchststand
    return {
        "Ann. Return": wealth.iloc[-1] ** (12 / len(returns)) - 1,   # geometrisch (CAGR)
        "Ann. Volatility": returns.std() * np.sqrt(12),
        "Sharpe Ratio": returns.mean() / returns.std() * np.sqrt(12),
        "Max Drawdown": (wealth / peak - 1).min(),
    }


# =============================================================================
# Ausgabe
# =============================================================================

def print_section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_table(table, decimals=2):
    print(table.to_string(float_format=lambda x: f"{x:.{decimals}f}"))


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # ein Farbton pro Modell
BENCHMARK_COLOR = "#52514e"


def plot_performance(returns, title, save_path):
    """Wertentwicklung von 1 investierten Dollar je Strategie; die letzte Spalte ist die Benchmark."""
    wealth = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(wealth.columns):
        is_benchmark = i == len(wealth.columns) - 1
        ax.plot(wealth.index, wealth[name], lw=2, label=f"{name}  ({wealth[name].iloc[-1]:.1f})",
                color=BENCHMARK_COLOR if is_benchmark else COLORS[i % len(COLORS)])
    ax.set_yscale("log")  # gleiche Rendite = gleiche Steigung, egal wie groß das Vermögen schon ist
    ax.set_ylabel("Value of 1 invested (log scale)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_time_variation(series, window, title, save_path):
    """Gleitender Mittelwert über `window` Monate, ein Teildiagramm pro Kennzahl in `series`."""
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 3.2 * len(series)), sharex=True)
    for ax, (label, monthly) in zip(axes, series.items()):
        rolling = monthly.rolling(window).mean()
        for i, name in enumerate(rolling.columns):
            ax.plot(rolling.index, rolling[name], lw=1.6, color=COLORS[i % len(COLORS)], label=name)
        ax.axhline(0, color="grey", lw=1)
        if label == "Predictive slope":
            ax.axhline(1, color="grey", lw=1, ls="--")  # Idealwert: unverzerrte Prognose
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, ncol=len(rolling.columns), loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
