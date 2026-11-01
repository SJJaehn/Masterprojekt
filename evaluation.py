"""
Schritte 3 bis 6: Auswertung der Prognosen.

- Fama-MacBeth-Regression mit Newey-West-t-Statistik (Lewellen, Tabelle 2)
- Güte der Prognosen: Verteilung und Steigung (Lewellen, Tabelle 3) sowie
  weitere Kennzahlen aus der Machine-Learning-Literatur (R2 OOS, IC, ...)
- Nach der Prognose sortierte Dezilportfolios (Lewellen, Tabelle 6)
- Handelsstrategie und ihre Performance-Kennzahlen

Alle Renditen sind Monatsrenditen als Dezimalzahl (0.01 = 1 %).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# =============================================================================
# Bausteine: Newey-West-Test und Fama-MacBeth-Regression
# =============================================================================

def newey_west_test(series, lags=4):
    """Mittelwert einer Zeitreihe mit Newey-West-Standardfehler und t-Statistik (statsmodels).

    Die Regression auf eine Konstante schätzt genau den Mittelwert; die
    HAC-Kovarianz berücksichtigt Autokorrelation bis `lags` Perioden (Paper: 4).
    Rückgabe: (Mittelwert, Standardfehler, t-Statistik).
    """
    y = pd.Series(series).dropna().to_numpy()
    if len(y) < 2:  # zu wenige Beobachtungen für einen Test
        return np.nan, np.nan, np.nan
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0], fit.bse[0], fit.tvalues[0]


def fama_macbeth(df, y_col, x_cols, date_col, nw_lags=4):
    """Fama-MacBeth-Regression wie im Paper (statsmodels).

    1. Jeden Monat eine Querschnittsregression y = a + b'x (OLS)
    2. Koeffizienten über alle Monate mitteln, t-Statistik mit Newey-West-Standardfehler

    Rückgabe: (Tabelle mit Coef, S.E. und t-stat pro Variable, monatliche
    Koeffizienten mit R2 und Anzahl der Aktien N).
    """
    # 1. Querschnittsregression pro Monat
    monthly = []
    for date, g in df.groupby(date_col):
        if g[x_cols].nunique().min() < 2:  # ohne Streuung in x ist die Regression nicht definiert
            continue
        fit = sm.OLS(g[y_col], sm.add_constant(g[x_cols])).fit()
        monthly.append({date_col: date, **fit.params.rename({"const": "Intercept"}),
                        "R2": fit.rsquared, "N": int(fit.nobs)})
    monthly = pd.DataFrame(monthly, columns=[date_col, "Intercept", *x_cols, "R2", "N"]).set_index(date_col)

    # 2. Mittelwert und Newey-West-t-Statistik der monatlichen Koeffizienten
    coefs = monthly.drop(columns=["R2", "N"])
    table = pd.DataFrame([newey_west_test(coefs[c], nw_lags) for c in coefs],
                         index=coefs.columns, columns=["Coef", "S.E.", "t-stat"])
    return table, monthly


# =============================================================================
# Güte der Prognosen
# =============================================================================

def forecast_evaluation(preds, pred_col, target_col, date_col, nw_lags=4):
    """Verteilung und Prognosegüte der erwarteten Renditen (Lewellen, Tabelle 3).

    - Avg, Std, p10, p90: Mittelwert, Standardabweichung, 10 %- und 90 %-Quantil
      der Prognosen im Querschnitt (in % pro Monat), gemittelt über alle Monate
    - Slope, S.E., t-stat, R2: Out-of-Sample-Fama-MacBeth-Regression der
      realisierten Renditen auf die Prognosen (siehe `fama_macbeth`). Eine
      unverzerrte Prognose hat eine Steigung von 1; > 0 heißt, die Prognose
      trennt Aktien mit hoher und niedriger Rendite (> 0.5: kleinerer MSE als
      eine Prognose, die allen Aktien dieselbe Rendite gibt).
    """
    dist = preds.groupby(date_col)[pred_col].describe(percentiles=[0.1, 0.9]).mean() * 100
    table, monthly = fama_macbeth(preds, target_col, [pred_col], date_col, nw_lags)
    slope = table.loc[pred_col]
    return {
        "Avg (%)": dist["mean"], "Std (%)": dist["std"], "p10 (%)": dist["10%"], "p90 (%)": dist["90%"],
        "Slope": slope["Coef"], "S.E.": slope["S.E."], "t-stat": slope["t-stat"], "R2": monthly["R2"].mean(),
    }


def prediction_metrics(preds, pred_col, target_col, date_col):
    """Weitere Out-of-Sample-Kennzahlen für die Renditeprognosen eines Modells.

    - RMSE / MAE      : klassische Prognosefehler (sklearn)
    - R2 OOS          : 1 - SSE / sum(y^2), Benchmark ist eine Prognose von null
                        (Gu, Kelly & Xiu, 2020) - positive Werte sind bereits gut
    - IC mean         : mittlere Spearman-Rangkorrelation zwischen Prognose und
                        realisierter Rendite im Querschnitt (pro Monat)
    - IC IR / t-stat  : Stabilität / Signifikanz des IC über die Zeit (t-Test, scipy)
    - Hit rate        : Anteil der Prognosen mit dem richtigen Vorzeichen
    """
    y_true, y_pred = preds[target_col], preds[pred_col]

    # Information Coefficient: für jeden Monat separat im Querschnitt berechnet
    ic = preds.groupby(date_col).apply(
        lambda g: g[target_col].corr(g[pred_col], method="spearman"), include_groups=False).dropna()

    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2 OOS": 1 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2),
        "IC mean": ic.mean(),
        "IC IR": ic.mean() / ic.std(),
        "IC t-stat": stats.ttest_1samp(ic, 0).statistic,
        "Hit rate": np.mean(np.sign(y_pred) == np.sign(y_true)),
    }


def forecast_correlations(preds, pred_cols, date_col):
    """Mittlere Korrelation der Prognosen zweier Modelle im Querschnitt (vgl. Lewellen, Tabelle 4)."""
    return preds.groupby(date_col)[pred_cols].corr().groupby(level=1).mean().loc[pred_cols, pred_cols]


# =============================================================================
# Dezilportfolios (Lewellen, Tabelle 6)
# =============================================================================

def sorted_portfolios(preds, pred_col, target_col, date_col, size_col=None, n_groups=10,
                      periods_per_year=12, nw_lags=4):
    """Nach der Prognose sortierte Portfolios (Lewellen, Tabelle 6).

    Jeden Monat werden die Aktien nach ihrer prognostizierten Rendite in
    `n_groups` gleich große Portfolios eingeteilt (Low = niedrigste Prognose).
    Pro Portfolio, gleich- (EW) und wertgewichtet (VW, mit `size_col`):
    - Pred:   mittlere prognostizierte Rendite (in % pro Monat)
    - Avg:    mittlere realisierte Rendite (in % pro Monat)
    - Std:    Standardabweichung der monatlichen Renditen (in %)
    - Sharpe: annualisierte Sharpe Ratio
    - t-stat: Newey-West-t-Statistik für eine mittlere Rendite > 0
    H-L ist das Long-Short-Portfolio High minus Low.
    """
    weightings = {"EW": None, "VW": size_col} if size_col else {"EW": None}
    tables = {}
    for weighting, weight_col in weightings.items():
        pred, ret = portfolio_returns(preds, pred_col, target_col, date_col, weight_col, n_groups)
        tables[weighting] = pd.DataFrame({
            "Pred (%)": pred.mean() * 100,
            "Avg (%)": ret.mean() * 100,
            "Std (%)": ret.std() * 100,
            "Sharpe": ret.mean() / ret.std() * np.sqrt(periods_per_year),
            "t-stat": [newey_west_test(ret[c], nw_lags)[2] for c in ret],
        })
    table = pd.concat(tables, axis=1).rename(index={1: "Low (L)", n_groups: "High (H)"})
    table.index = table.index.astype(str)
    return table


def portfolio_returns(preds, pred_col, target_col, date_col, weight_col=None, n_groups=10):
    """Monatliche Prognose und Rendite der nach der Prognose sortierten Portfolios.

    Jeden Monat werden die Aktien nach ihrer Prognose in `n_groups` gleich große
    Portfolios eingeteilt (1 = niedrigste Prognose); gleichgewichtet oder, mit
    `weight_col`, nach Marktkapitalisierung gewichtet. Rückgabe: (Prognosen,
    Renditen), je eine Tabelle Monate x Portfolios plus Spalte H-L (High minus Low).
    """
    # Portfolio 1 ... n_groups pro Monat (rank: gleiche Prognosen werden auf die Portfolios verteilt)
    df = preds.copy()
    df["portfolio"] = df.groupby(date_col)[pred_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False) + 1)
    grouped = df.groupby([date_col, "portfolio"])

    # mittlere Prognose und mittlere Rendite je Monat und Portfolio (gleich- oder wertgewichtet)
    if weight_col:
        means = grouped.apply(lambda g: pd.Series(
            {c: np.average(g[c], weights=g[weight_col]) for c in (pred_col, target_col)}),
            include_groups=False)
    else:
        means = grouped[[pred_col, target_col]].mean()
    pred, ret = means[pred_col].unstack(), means[target_col].unstack()  # Monate x Portfolios
    pred["H-L"] = pred[n_groups] - pred[1]
    ret["H-L"] = ret[n_groups] - ret[1]
    return pred, ret


# =============================================================================
# Entwicklung über die Zeit (Lewellen, Abschnitt 3.6 und Fig. 2)
# =============================================================================

def monthly_slopes(preds, pred_col, target_col, date_col):
    """Steigung der Querschnittsregression realisierte Rendite = a + b * Prognose, für jeden Monat einzeln.

    Der Mittelwert dieser Reihe ist die Steigung aus Tabelle 3; die Reihe selbst
    zeigt, wie gut die Prognosen in jedem einzelnen Monat waren.
    """
    _, monthly = fama_macbeth(preds, target_col, [pred_col], date_col)
    return monthly[pred_col]


def trend_test(series, split_date, nw_lags=4):
    """Gibt es einen Abwärtstrend? Zwei einfache Tests für eine monatliche Reihe.

    1. Hälftenvergleich: Mittelwert vor und ab `split_date` und die Differenz
       (Regression auf eine Dummy-Variable für die zweite Hälfte)
    2. Linearer Trend: Regression auf die Zeit in Jahrzehnten; die Steigung ist
       die Veränderung des Mittelwerts pro 10 Jahre
    Beide mit Newey-West-Standardfehlern (`nw_lags` Lags, wie im Paper).
    Ein negativer Wert mit |t| > 2 spricht für einen signifikanten Rückgang.
    """
    y = series.dropna()
    second_half = (y.index >= pd.Timestamp(split_date)).astype(float)
    decades = (y.index - y.index[0]).days / 3652.5
    hac = {"cov_type": "HAC", "cov_kwds": {"maxlags": nw_lags}}

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
# Handelsstrategie
# =============================================================================

def strategy_returns(preds, pred_col, target_col, date_col, long_pct=0.2, short_pct=0.0):
    """Rendite pro Monat eines gleichgewichteten Long(-Short)-Portfolios.

    In jedem Monat werden die Aktien nach ihrer prognostizierten Rendite sortiert:
    - long  die obersten `long_pct` (höchste prognostizierte Rendite)
    - short die untersten `short_pct` (niedrigste prognostizierte Rendite), 0 = kein Shorting
    Beide Seiten sind gleichgewichtet; mit Shorting ist die Rendite Long - Short.
    """
    if long_pct < 0 or short_pct < 0 or long_pct + short_pct > 1:
        raise ValueError("long_pct and short_pct must be >= 0 and sum to at most 1.")

    returns = {}
    for date, g in preds.groupby(date_col):
        # realisierte Renditen, sortiert von der höchsten zur niedrigsten Prognose
        realised = g.sort_values(pred_col, ascending=False, kind="stable")[target_col]
        n = len(realised)

        ret = 0.0
        if long_pct > 0:
            n_long = max(1, int(round(n * long_pct)))  # mindestens 1 Aktie
            ret += realised.iloc[:n_long].mean()
        if short_pct > 0:
            n_short = max(1, int(round(n * short_pct)))
            ret -= realised.iloc[-n_short:].mean()
        returns[date] = ret
    return pd.Series(returns, name=pred_col)


def equal_weight_returns(preds, target_col, date_col):
    """Benchmark: gleichgewichtetes Portfolio aus allen Aktien in jedem Monat."""
    return preds.groupby(date_col)[target_col].mean().rename("Equal Weight")


def strategy_label(long_pct, short_pct):
    label = f"Long top {long_pct:.0%}" if long_pct > 0 else ""
    if short_pct > 0:
        label += (" / " if label else "") + f"Short bottom {short_pct:.0%}"
    return label


def performance_metrics(returns, periods_per_year=12, rf=None):
    """Annualisierte Performance einer Reihe von Portfoliorenditen pro Periode.

    Die annualisierte Rendite ist geometrisch (CAGR); die Sharpe Ratio verwendet
    die arithmetisch gemittelte Überrendite über `rf` (Serie pro Periode aus
    `dataset.load_risk_free`). Bei rf=None werden die Rohrenditen verwendet, z.B. für
    selbstfinanzierende Long-Short-Portfolios oder wenn kein risikofreier Zins
    vorliegt.
    """
    returns = returns.dropna()
    wealth = (1 + returns).cumprod()
    excess = returns if rf is None else returns - rf.reindex(returns.index)
    peak = np.maximum(wealth.cummax(), 1.0)  # das Startkapital zählt als Höchststand

    return {
        "Ann. Return": wealth.iloc[-1] ** (periods_per_year / len(returns)) - 1,
        "Ann. Volatility": returns.std() * np.sqrt(periods_per_year),
        "Sharpe Ratio": excess.mean() / excess.std() * np.sqrt(periods_per_year),
        "Max Drawdown": (wealth / peak - 1).min(),
    }
