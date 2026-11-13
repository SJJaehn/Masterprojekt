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

def rank_characteristics(df):
    """Rangtransformation pro Monat, skaliert auf [-0.5, 0.5] (wie fCrossSectTransChars.m).

    Fehlende Werte bleiben fehlend, gleiche Werte erhalten den mittleren Rang.
    """
    ranks = df.groupby("eom")[JKP_CHARACTERISTICS].rank(method="average")
    return (ranks - 1) / (ranks.groupby(df["eom"]).transform("max") - 1) - 0.5


def rank_returns(df):
    """Rang der Rendite im Folgemonat innerhalb jedes Monats, skaliert auf [-0.5, 0.5] wie die Kennzahlen.

    Zielgröße der Bäume bei TARGET = "rank": Der Baum lernt dann nur die Reihenfolge der Aktien,
    nicht das Marktniveau des Monats und nicht die Höhe einzelner extremer Renditen.
    """
    ranks = df.groupby(DATE)[RET].rank(method="average")
    return ((ranks - 1) / (ranks.groupby(df[DATE]).transform("max") - 1) - 0.5).to_numpy()


def build_prepared_data(raw_path, prepared_path, universe="top1000", top_k=1000, years_before=25, years_after=5):
    """Erzeugt aus den JKP-Rohdaten (USA.parquet, 6 GB) die kleine CSV in data/.

    Wird nur gebraucht, wenn die CSV fehlt. Die Ränge beziehen sich auf alle Aktien der
    Rohdaten. Die Spalte `is_included` (1/0) markiert die Monate, in denen eine Stammaktie
    (CRSP-Share-Code 10/11) zum Universum gehört, je nach `universe`:

    "top1000": die `top_k` größten Aktien mit Rendite, ohne nano/micro Stocks
    "large":   Lewellens "large stocks", größer als der NYSE-Median

    Von diesen Aktien bleiben alle Zeilen der `years_before` Jahre vor und der `years_after`
    Jahre nach jedem solchen Monat erhalten, soweit die Rohdaten sie enthalten.
    """
    # Spalten lesen, Datum umwandeln
    df = pd.read_parquet(raw_path, columns=["id", "eom", "size_grp", "crsp_shrcd", "ret_exc_lead1m", "me",
                                            *JKP_CHARACTERISTICS])
    df["eom"] = pd.to_datetime(df["eom"].astype(str), format="%Y%m%d")
    df[JKP_CHARACTERISTICS] = rank_characteristics(df)
    is_common = df["crsp_shrcd"].isin([10, 11])

    if universe == "large":
        # JKP teilt die Aktien jeden Monat nach NYSE-Breakpoints der Marktkapitalisierung ein (size_grp):
        # nano < 1 %, micro 1-20 %, small 20-50 %, large 50-80 %, mega > 80 %.
        # Größer als der NYSE-Median sind also genau "large" und "mega".
        is_selected = (df["size_grp"].isin(["large", "mega"]) & is_common).to_numpy()
    else:
        eligible = is_common & df["ret_exc_lead1m"].notna() & ~df["size_grp"].isin(["nano", "micro"])
        size_rank = df["me"].where(eligible).groupby(df["eom"]).rank(ascending=False, method="first")
        is_selected = (size_rank <= top_k).to_numpy()
    df["is_included"] = is_selected.astype(int)
    df = df.drop(columns=["size_grp", "crsp_shrcd"])

    # Aktien behalten, die in einem Monat t ausgewählt sind:
    # alle ihre Zeilen von t - months_before bis t + months_after Monaten
    months_before, months_after = 12 * years_before, 12 * years_after
    month = (df["eom"].dt.year * 12 + df["eom"].dt.month).to_numpy()  # fortlaufender Monatsindex

    rows = pd.DataFrame({"pos": np.arange(len(df)), "id": df["id"].to_numpy(), "month": month})
    rows = rows.sort_values("month")
    tops = pd.DataFrame({"id": df["id"].to_numpy()[is_selected], "top_month": month[is_selected]})
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

    Monate, in denen eine Variable für alle Aktien gleich ist, werden übersprungen: Dort ist
    die Steigung nicht definiert (das kommt vor, wenn ein Baum der Tiefe 1 allen Aktien
    dieselbe Prognose gibt).

    Rückgabe: (Tabelle mit Coef, S.E. und t-stat pro Variable, monatliche
    Koeffizienten mit R2 und Anzahl der Aktien N).
    """
    # 1. Querschnittsregression pro Monat
    monthly = []
    for date, g in df.groupby(DATE):
        if (g[x_cols].nunique() <= 1).any():
            continue
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
                   (ohne Monate, in denen alle Aktien dieselbe Prognose haben)
    - Hit rate   : Anteil der Prognosen mit dem richtigen Vorzeichen
    """
    y_true, y_pred = preds[RET], preds[model]
    ic = preds.groupby(DATE).apply(lambda g: g[RET].corr(g[model], method="spearman"),
                                   include_groups=False).dropna()
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

def portfolio_returns(preds, model, n_groups=10, value_weighted=False, high_is_good=True):
    """Monatliche Prognose und Rendite der nach einer Spalte sortierten Portfolios.

    Jeden Monat werden die Aktien nach der Spalte `model` in `n_groups` gleich
    große Portfolios eingeteilt: 1 = niedrigster Wert, n_groups = höchster Wert.
    Mit `high_is_good=False` umgekehrt (für die Disagreement: n_groups = geringste
    Disagreement), damit High immer die erwartet beste Gruppe ist.
    Gleichgewichtet oder nach Marktkapitalisierung gewichtet. Rückgabe: (Werte
    der Spalte, Renditen), je eine Tabelle Monate x Portfolios plus Spalte H-L
    (High minus Low).
    """
    # Portfolio 1 ... n_groups pro Monat (rank: gleiche Werte werden auf die Portfolios verteilt)
    df = preds.copy()
    sign = 1 if high_is_good else -1
    df["portfolio"] = df.groupby(DATE)[model].transform(
        lambda s: pd.qcut((sign * s).rank(method="first"), n_groups, labels=False) + 1)
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


def sorted_portfolios(preds, model, n_groups=10, high_is_good=True, value_label="Pred (%)"):
    """Nach einer Spalte sortierte Portfolios (Lewellen, Tabelle 6).

    Pro Portfolio, gleich- (EW) und wertgewichtet (VW): mittlerer Wert der
    Sortierspalte (`value_label`, z. B. Pred = mittlere Prognose), Avg (mittlere
    realisierte Rendite), Std (in % pro Monat), annualisierte Sharpe Ratio und
    Newey-West-t-Statistik. `high_is_good` wie in `portfolio_returns`.
    """
    tables = {}
    for weighting, value_weighted in [("EW", False), ("VW", True)]:
        pred, ret = portfolio_returns(preds, model, n_groups, value_weighted, high_is_good)
        tables[weighting] = pd.DataFrame({
            value_label: pred.mean() * 100,
            "Avg (%)": ret.mean() * 100,
            "Std (%)": ret.std() * 100,
            "Sharpe": ret.mean() / ret.std() * np.sqrt(12),
            "t-stat": [newey_west_test(ret[c])[2] for c in ret],
        })
    table = pd.concat(tables, axis=1).rename(index={1: "Low (L)", n_groups: "High (H)"})
    table.index = table.index.astype(str)
    return table


def strategy_returns(preds, model, long_pct=0.0, short_pct=0.0, high_is_good=True):
    """Rendite pro Monat einer gleichgewichteten Long-, Short- oder Long-Short-Strategie.

    Jeden Monat werden die Aktien nach `model` sortiert, von der besten zur schlechtesten:
    - long  die besten `long_pct` Aktien kaufen
    - short die schlechtesten `short_pct` Aktien leerverkaufen (Rendite = minus deren Rendite)
    Sind beide gesetzt, ist die Rendite Long minus Short (selbstfinanzierend: der Erlös aus dem
    Leerverkauf bezahlt den Kauf). "Beste" heißt höchster Wert von `model`, mit
    `high_is_good=False` niedrigster Wert (z. B. die geringste Disagreement).
    """
    returns = {}
    for date, g in preds.groupby(DATE):
        # realisierte Renditen, sortiert von der besten zur schlechtesten Aktie nach `model`
        realised = g.sort_values(model, ascending=not high_is_good, kind="stable")[RET]
        ret = 0.0
        if long_pct > 0:
            n_long = max(1, int(round(len(realised) * long_pct)))  # mindestens 1 Aktie
            ret += realised.iloc[:n_long].mean()
        if short_pct > 0:
            n_short = max(1, int(round(len(realised) * short_pct)))
            ret -= realised.iloc[-n_short:].mean()
        returns[date] = ret
    return pd.Series(returns, name=model)


def backtest(preds, signals, strategies, high_is_good=True):
    """Monatsrenditen aller Strategien für alle Signale, plus Equal Weight als Benchmark.

    `strategies` ordnet jedem Strategienamen (Anteil long, Anteil short) zu, z. B.
    {"Long": (0.1, 0.0), "Short": (0.0, 0.1), "Long-Short": (0.1, 0.1)}.
    Rückgabe: Tabelle Monate x Spalten (Strategie, Signal).
    """
    returns = {(label, name): strategy_returns(preds, name, long_pct, short_pct, high_is_good)
               for label, (long_pct, short_pct) in strategies.items() for name in signals}
    returns[("Benchmark", "Equal Weight")] = preds.groupby(DATE)[RET].mean()  # alle Aktien gleichgewichtet
    return pd.DataFrame(returns)


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
# Transaktionskosten (Avramov, Cheng & Metzker, 2023, Abschnitt III.D)
# =============================================================================

def cost_per_stock(preds):
    """Einfache (one-way) Transaktionskosten je Aktie und Monat als Anteil des gehandelten Betrags.

    Wie bei Avramov, Cheng & Metzker nach Brandt, Santa-Clara & Valkanov (2009):
        c_it = z_it * T_t,   z_it = 0,006 - 0,0025 * ME_it
    - ME_it: Marktkapitalisierung, im Monat auf 0 (kleinste) bis 1 (größte Aktie) normiert, hier
      über den Rang. Große Aktien sind billiger zu handeln.
    - T_t: Kosten waren früher höher. T sinkt linear von 4 im Jahr 1974 auf 1 im Jahr 2002 und
      bleibt danach bei 1 (1974 viermal so teuer wie 2002; 1987 ergibt das 2,6 wie im Paper).
    Ergebnis: 2002 und später 0,60 % für die kleinste und 0,35 % für die größte Aktie, 1974 das
    Vierfache.
    """
    rank = preds.groupby(DATE)[SIZE].rank(method="average")
    me = (rank - 1) / (preds.groupby(DATE)[SIZE].transform("count") - 1)
    years = (preds[DATE] - pd.Timestamp("1974-01-01")).dt.days / 365.25
    trend = np.clip(4 - 3 * years / 28, 1, 4)
    return pd.Series((0.006 - 0.0025 * me) * trend, index=preds.index)


def leg_costs(weights, preds, cost):
    """Turnover und Kosten einer Seite (long oder short) für jeden Monat.

    `weights`: Monat -> Gewichte der Aktien (Summe 1). Vor dem Umschichten im Monat t haben sich
    die Gewichte vom Vormonat mit den Renditen verschoben (Gu, Kelly & Xiu; Avramov et al.):
        w~_i = w_i,t-1 * (1 + r_i) / sum_j w_j,t-1 * (1 + r_j)
    Gehandelt wird |w_it - w~_i|; Aktien, die neu dazukommen oder herausfallen, ganz.
    - Turnover (eine Seite, 0 bis 1) = 1/2 * Summe |w_it - w~_i|
    - Kosten = Summe c_i * |w_it - w~_i| (jeder Kauf und jeder Verkauf kostet c_i);
      Aktien, die im Monat t nicht mehr in der Stichprobe sind, mit ihren Kosten aus t-1.
    Im ersten Monat wird das Portfolio aus Bargeld aufgebaut.
    """
    ret = preds.set_index([DATE, ID])[RET]
    c = cost.set_axis(pd.MultiIndex.from_frame(preds[[DATE, ID]]))
    turnover, costs = {}, {}
    previous, previous_date = None, None
    for date, w in weights.items():
        if previous is None:
            drifted = pd.Series(dtype=float)
        else:
            grown = previous * (1 + ret.loc[previous_date].reindex(previous.index).fillna(0))
            drifted = grown / grown.sum()
        trade = w.sub(drifted, fill_value=0).abs()
        c_now = c.loc[date].reindex(trade.index)
        if previous is not None:  # herausgefallene Aktien: Kosten aus dem Vormonat
            c_now = c_now.fillna(c.loc[previous_date].reindex(trade.index))
        turnover[date] = trade.sum() / 2
        costs[date] = (trade * c_now.fillna(c_now.mean())).sum()
        previous, previous_date = w, date
    return pd.Series(turnover), pd.Series(costs)


def strategy_weights(preds, model, pct, side, high_is_good=True):
    """Gleichgewichtete Positionen einer Seite für jeden Monat, dieselbe Auswahl wie strategy_returns."""
    weights = {}
    for date, g in preds.groupby(DATE):
        ordered = g.sort_values(model, ascending=not high_is_good, kind="stable")[ID]
        n = max(1, int(round(len(ordered) * pct)))
        chosen = ordered.iloc[:n] if side == "long" else ordered.iloc[-n:]
        weights[date] = pd.Series(1 / n, index=chosen.to_numpy())
    return weights


def net_of_costs(preds, gross, signals, strategies, high_is_good=True):
    """Renditen nach Transaktionskosten, Turnover und Kosten für alle Strategien und Equal Weight.

    `gross` ist die Ausgabe von backtest(). Rückgabe: (Nettorenditen wie `gross`, Tabelle mit
    Turnover, Kosten und Break-even-Kosten je Strategie und Signal).
    """
    cost = cost_per_stock(preds)
    sides = {}  # (Signal, Seite, Anteil) -> (Turnover, Kosten) pro Monat
    for name in signals:
        for long_pct, short_pct in strategies.values():
            for side, pct in [("long", long_pct), ("short", short_pct)]:
                if pct > 0 and (name, side, pct) not in sides:
                    sides[(name, side, pct)] = leg_costs(strategy_weights(preds, name, pct, side, high_is_good),
                                                         preds, cost)
    everyone = {date: pd.Series(1 / len(g), index=g[ID].to_numpy()) for date, g in preds.groupby(DATE)}
    benchmark = leg_costs(everyone, preds, cost)

    net, rows = {}, []
    for (label, name), returns in gross.items():
        if label == "Benchmark":
            turnover, costs = benchmark
        else:
            long_pct, short_pct = strategies[label]
            parts = [sides[(name, s, pct)] for s, pct in [("long", long_pct), ("short", short_pct)] if pct > 0]
            turnover = sum(p[0] for p in parts)  # Long-Short: beide Seiten (0 bis 2)
            costs = sum(p[1] for p in parts)
        net[(label, name)] = returns - costs.reindex(returns.index)
        rows.append({"strategy": label, "signal": name,
                     "Turnover": turnover.mean(),
                     "Cost (% p.m.)": costs.mean() * 100,
                     "Gross return (% p.m.)": returns.mean() * 100,
                     "Net return (% p.m.)": net[(label, name)].mean() * 100,
                     # Kosten je Kauf bzw. Verkauf, bei denen die Bruttorendite ganz aufgezehrt wäre
                     "Break-even one-way cost (%)": returns.mean() / (2 * turnover.mean()) * 100,
                     # wie im Paper: Rendite / Turnover (entspricht den Kosten für Kauf und Verkauf zusammen)
                     "Break-even return / turnover (%)": returns.mean() / turnover.mean() * 100})
    return pd.DataFrame(net), pd.DataFrame(rows).set_index(["strategy", "signal"])


def evaluate_costs(preds, signals, strategies, pred_dir, mfd_dir, period):
    """Schritt 6b: Strategien nach Transaktionskosten, für die Modelle und (falls vorhanden) die MFD.

    Speichert in `pred_dir` bzw. `mfd_dir`: transaction_costs.csv (Turnover, Kosten, Break-even),
    strategy_returns_net.csv, performance_net.csv und performance_<Strategie>_net.png (die Modelle
    ohne Investors, dazu die MFD-Strategie und Equal Weight, alle nach Kosten).
    """
    gross = backtest(preds, signals, strategies)
    net, table = net_of_costs(preds, gross, signals, strategies)
    results = {pred_dir: (net, table)}
    if "MFD" in preds:
        mfd_gross = backtest(preds, ["MFD"], strategies, high_is_good=False)
        results[mfd_dir] = net_of_costs(preds, mfd_gross, ["MFD"], strategies, high_is_good=False)
    for folder, (net_returns, cost_table) in results.items():
        performance = pd.DataFrame({column: performance_metrics(net_returns[column])
                                    for column in net_returns}).T
        cost_table.to_csv(os.path.join(folder, "transaction_costs.csv"))
        net_returns.to_csv(os.path.join(folder, "strategy_returns_net.csv"))
        performance.to_csv(os.path.join(folder, "performance_net.csv"))
    for label in strategies:
        lines = net[label].drop(columns="Investors", errors="ignore")
        if mfd_dir in results:
            lines = lines.assign(MFD=results[mfd_dir][0][(label, "MFD")])
        plot_performance(lines.assign(**{"Equal Weight": net[("Benchmark", "Equal Weight")]}),
                         f"{label} after transaction costs: models and MFD vs. equal weight ({period})",
                         os.path.join(pred_dir, f"performance_{label.lower()}_net.png"))
    return results


# =============================================================================
# Ausgabe
# =============================================================================

def print_section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_table(table, decimals=2):
    print(table.to_string(float_format=lambda x: f"{x:.{decimals}f}"))


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]  # ein Farbton pro Modell
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
