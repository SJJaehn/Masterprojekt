"""
Hilfsfunktionen: Daten laden, Rolling Window, Evaluationskennzahlen (inkl. der
Tabellen aus Lewellen, 2015), Simulation der Handelsstrategie und Grafiken.

Jede Funktion ist von oben nach unten wie ein kleines Skript geschrieben: die
Schritte stehen direkt hintereinander (mit etwas Wiederholung), statt in vielen
kleinen Hilfsfunktionen versteckt zu sein.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# =============================================================================
# Daten
# =============================================================================

# Charakteristika aus dem Tutor-Skript b00PrepareDataset.m (Replikation von Lewellen, 2015, Modell 3)
JKP_CHARACTERISTICS = [
    "market_equity", "be_me", "ret_12_1", "chcsho_12m", "oaccruals_at", "ni_at", "at_gr1",
    "beta_60m", "div12m_me", "ret_60_12", "rvol_252d", "turnover_126d", "debt_me", "sale_me",
]

# Prädiktoren der drei Modelle bei Lewellen (2015), Tabelle 2:
# Modell 1: Größe, B/M, Momentum; Modell 2: + Emissionen, Accruals, Profitabilität, Asset Growth;
# Modell 3: + Beta, Dividendenrendite, Langfristrendite, Volatilität, Turnover, Verschuldung, Umsatz/Preis
LEWELLEN_MODELS = {
    "Model 1": JKP_CHARACTERISTICS[:3],
    "Model 2": JKP_CHARACTERISTICS[:7],
    "Model 3": JKP_CHARACTERISTICS,
}


def load_data(path, top_k=1000, years_before=25, years_after=5, transform="rank"):
    """Lädt die JKP-Daten (USA.parquet) analog zu b00PrepareDataset.m und b01LoadDataset.m.

    1. Nur die benötigten Spalten lesen (wie im Tutor-Skript, plus `me` als
       untransformierte Marktkapitalisierung für die Auswahl des Aktienuniversums)
    2. Zeilen ohne Rendite oder Marktkapitalisierung sowie nano/micro Stocks entfernen
    3. Charakteristika pro Monat im Querschnitt transformieren (Standard: Ränge
       auf [-0.5, 0.5], wie fCrossSectTransChars.m)
    4. Pro Monat die `top_k` größten Aktien nach Marktkapitalisierung bestimmen und
       von diesen Aktien die Daten der `years_before` Jahre davor (Training) und
       der `years_after` Jahre danach (Prognose) behalten

    Ergebnis ist ein Panel mit einer Zeile pro Monat und Aktie: eom (Monatsende),
    id, ret_exc_lead1m (Überrendite im Folgemonat), me und die Charakteristika
    zum Zeitpunkt eom. Die Zeilen eines Monats bilden einen Querschnitt
    (n Aktien x T Charakteristika).
    """
    # 1. Spalten lesen
    df = pd.read_parquet(path, columns=["id", "eom", "size_grp", "ret_exc_lead1m", "me",
                                        *JKP_CHARACTERISTICS])

    # 2. Zeilen filtern, Datum umwandeln
    df = df[df["ret_exc_lead1m"].notna() & df["me"].notna() & ~df["size_grp"].isin(["nano", "micro"])]
    df = df.drop(columns="size_grp")
    df["eom"] = pd.to_datetime(df["eom"].astype(str), format="%Y%m%d")

    # 3. Querschnittstransformation pro Monat (fehlende Werte bleiben fehlend)
    #    "rank": Ränge, skaliert auf [-0.5, 0.5] | "z": z-Standardisierung | "none": unverändert
    grouped = df.groupby("eom")[JKP_CHARACTERISTICS]
    if transform == "rank":
        ranks = grouped.rank(method="average")  # gleiche Werte erhalten den mittleren Rang (wie tiedrank)
        df[JKP_CHARACTERISTICS] = (ranks - 1) / (ranks.groupby(df["eom"]).transform("max") - 1) - 0.5
    elif transform == "z":
        df[JKP_CHARACTERISTICS] = ((df[JKP_CHARACTERISTICS] - grouped.transform("mean"))
                                   / grouped.transform("std"))
    elif transform != "none":
        raise ValueError(f"Unknown transform '{transform}'. Use 'rank', 'z' or 'none'.")

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

    return df.sort_values(["eom", "id"]).reset_index(drop=True)


def save_prepared_data(df, path):
    """Speichert die vorgefilterten Daten als gzip-komprimierte CSV.

    Gerundet auf 6 signifikante Stellen (Ränge bleiben exakt in ihrer
    Reihenfolge), damit die Datei unter dem 100-MB-Limit von GitHub bleibt.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")


def load_prepared_data(path, date_col):
    """Lädt die mit `save_prepared_data` gespeicherten Daten (dauert wenige Sekunden)."""
    return pd.read_csv(path, parse_dates=[date_col])


def load_risk_free(path, dates):
    """Risikofreier Zins pro Periode, ausgerichtet auf `dates`, oder None, wenn keine Datei vorhanden ist.

    Erwartet wird eine CSV mit zwei Spalten, date und rf, wobei rf der Zins pro
    Periode als Dezimalzahl ist (z.B. 0.002 = 0,2 % pro Monat - Fama/French-Daten
    sind in Prozent angegeben, also durch 100 teilen). Gleiches Timing wie die
    Zielgröße: rf zum Datum t ist die risikofreie Rendite über die NÄCHSTE
    Periode, und die Datumsangaben müssen mit denen der Aktiendaten übereinstimmen.
    """
    if not path or not os.path.exists(path):
        return None

    rf = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    rf = rf.reindex(pd.DatetimeIndex(dates))
    if rf.isna().all():
        raise ValueError(f"No dates in '{path}' match the stock data - check the date format/convention.")
    if rf.isna().any():
        print(f"  Warning: risk-free rate missing for {rf.isna().sum()} of {len(rf)} periods - set to 0")
        rf = rf.fillna(0.0)
    return rf


def make_dummy_data(n_stocks=100, n_periods=144, n_features=10, seed=42,
                    date_col="date", id_col="ticker", target_col="ret_next", size_col="me"):
    """Zufälliges monatliches Aktienpanel mit einem schwachen (teils nichtlinearen) Signal.

    Dient zum schnellen Testen der Pipeline (USE_DUMMY_DATA in main.py).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_periods, freq="ME")
    tickers = [f"STOCK_{i:03d}" for i in range(1, n_stocks + 1)]

    X = rng.standard_normal((n_periods, n_stocks, n_features))
    # Marktfaktor: in einer Periode für alle Aktien gleich
    market = rng.normal(0.006, 0.045, size=(n_periods, 1))
    # Signal: linear (feature_1, feature_2), Interaktion (feature_3 * feature_4), Schwelle (feature_5)
    signal = (0.008 * X[..., 0] - 0.006 * X[..., 1]
              + 0.006 * X[..., 2] * X[..., 3] + 0.010 * (X[..., 4] > 1))
    noise = rng.normal(0.0, 0.10, size=(n_periods, n_stocks))

    index = pd.MultiIndex.from_product([dates, tickers], names=[date_col, id_col])
    df = pd.DataFrame(X.reshape(-1, n_features), index=index,
                      columns=[f"feature_{i}" for i in range(1, n_features + 1)])
    df[target_col] = (market + signal + noise).reshape(-1)
    # Marktkapitalisierung: fester Größenunterschied je Aktie plus Schwankung
    df[size_col] = np.exp(rng.normal(7, 1.5, n_stocks)[None, :]
                          + rng.normal(0, 0.2, (n_periods, n_stocks))).reshape(-1)
    return df.reset_index()


def prepare_data(df, date_col, id_col, target_col, feature_cols):
    """Prüft die Spalten, wandelt das Datum um, entfernt Zeilen ohne Zielgröße und sortiert."""
    missing = [c for c in [date_col, id_col, target_col, *feature_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[target_col])
    return df.sort_values([date_col, id_col]).reset_index(drop=True)


def select_sample(df, date_col, size_col, feature_cols, start=None, drop_missing=True, universe_size=None):
    """Stichprobe wie bei Lewellen (2015).

    Nur Monate ab `start`, nur Aktien mit Werten für alle Charakteristika
    (`drop_missing`, wie im Paper) und davon pro Monat die `universe_size`
    größten Aktien nach `size_col` (None: alle).
    """
    if start is not None:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if drop_missing:
        df = df.dropna(subset=feature_cols)
    if universe_size:
        is_largest = df.groupby(date_col)[size_col].rank(ascending=False, method="first") <= universe_size
        df = df[is_largest]
    return df.reset_index(drop=True)


# =============================================================================
# Rolling Window
# =============================================================================

def test_period(dates, min_history, test_start=None, test_end=None):
    """Indizes (first, last) des ersten und letzten Prognosedatums in den sortierten `dates`.

    Vor dem ersten Prognosedatum liegen mindestens `min_history` Perioden.
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


def rolling_windows(dates, train_window, retrain_every=1, test_start=None, test_end=None, expanding=False):
    """Liefert (train_dates, test_dates) für ein Rolling Window über alle Datumswerte.

    Das Trainingsset enthält die `train_window` Perioden direkt vor der/den
    Testperiode(n) (bei `expanding` alle Perioden seit Beginn der Daten, aber
    mindestens `train_window`); das Testset enthält die nächsten `retrain_every`
    Perioden. Prognostiziert wird nur zwischen `test_start` und `test_end`.
    """
    dates = np.sort(pd.unique(dates))
    first, last = test_period(dates, train_window, test_start, test_end)
    for start in range(first, last + 1, retrain_every):
        train_from = 0 if expanding else start - train_window
        yield dates[train_from:start], dates[start:min(start + retrain_every, last + 1)]


def fixed_universe(train, test, date_col, id_col, size_col, size):
    """Beschränkt Trainings- und Testdaten eines Fensters auf die `size` größten Aktien zum ersten Testdatum t.

    Trainiert wird auf deren Historie, prognostiziert für genau diese Aktien in
    den Testperioden (nutzt nur die Marktkapitalisierung bis t).
    """
    is_first_date = test[date_col] == test[date_col].min()
    is_largest = test.groupby(date_col)[size_col].rank(ascending=False, method="first") <= size
    ids = test.loc[is_first_date & is_largest, id_col]
    return train[train[id_col].isin(ids)], test[test[id_col].isin(ids)]


def fmt_date(date):
    return pd.Timestamp(date).strftime("%Y-%m-%d")


# =============================================================================
# Prognosekennzahlen
# =============================================================================

def prediction_metrics(preds, pred_col, target_col, date_col):
    """Out-of-Sample-Kennzahlen für die Renditeprognosen eines Modells.

    - RMSE / MAE      : klassische Prognosefehler (sklearn)
    - R2 OOS          : 1 - SSE / sum(y^2), Benchmark ist eine Prognose von null
                        (Gu, Kelly & Xiu, 2020) - positive Werte sind bereits gut
    - IC mean         : mittlere Spearman-Rangkorrelation zwischen Prognose und
                        realisierter Rendite im Querschnitt (pro Periode)
    - IC IR / t-stat  : Stabilität / Signifikanz des IC über die Zeit (t-Test, scipy)
    - Hit rate        : Anteil der Prognosen mit dem richtigen Vorzeichen
    """
    y_true, y_pred = preds[target_col], preds[pred_col]

    # Information Coefficient: für jedes Datum separat im Querschnitt berechnet
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


# =============================================================================
# Auswertung wie Lewellen (2015)
# =============================================================================

EVALUATION_FORMATS = {"R2": "{:.3f}"}  # alle anderen Spalten mit 2 Nachkommastellen


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


def forecast_correlations(preds, pred_cols, date_col):
    """Mittlere Korrelation der Prognosen zweier Modelle im Querschnitt (vgl. Lewellen, Tabelle 4)."""
    return preds.groupby(date_col)[pred_cols].corr().groupby(level=1).mean().loc[pred_cols, pred_cols]


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
    # Portfolio 1 ... n_groups pro Monat (rank: gleiche Prognosen werden auf die Portfolios verteilt)
    df = preds.copy()
    df["portfolio"] = df.groupby(date_col)[pred_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False) + 1)
    grouped = df.groupby([date_col, "portfolio"])

    weightings = {"EW": None, "VW": size_col} if size_col else {"EW": None}
    tables = {}
    for weighting, weight_col in weightings.items():
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


# =============================================================================
# Handelsstrategie
# =============================================================================

def strategy_returns(preds, pred_col, target_col, date_col, long_pct=0.2, short_pct=0.0):
    """Rendite pro Periode eines gleichgewichteten Long(-Short)-Portfolios.

    In jeder Periode werden die Aktien nach ihrer prognostizierten Rendite sortiert:
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
    """Benchmark: gleichgewichtetes Portfolio aus allen Aktien in jeder Periode."""
    return preds.groupby(date_col)[target_col].mean().rename("Equal Weight")


def strategy_label(long_pct, short_pct):
    label = f"Long top {long_pct:.0%}" if long_pct > 0 else ""
    if short_pct > 0:
        label += (" / " if label else "") + f"Short bottom {short_pct:.0%}"
    return label


# =============================================================================
# Performance-Kennzahlen
# =============================================================================

PERFORMANCE_FORMATS = {
    "Ann. Return": "{:.2%}",
    "Ann. Volatility": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Max Drawdown": "{:.2%}",
}


def performance_metrics(returns, periods_per_year=12, rf=None):
    """Annualisierte Performance einer Reihe von Portfoliorenditen pro Periode.

    Die annualisierte Rendite ist geometrisch (CAGR); die Sharpe Ratio verwendet
    die arithmetisch gemittelte Überrendite über `rf` (Serie pro Periode aus
    `load_risk_free`). Bei rf=None werden die Rohrenditen verwendet, z.B. für
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


# =============================================================================
# Ausgabe
# =============================================================================

def format_table(table, formats=None, decimals=2):
    """Formatiert jede Spalte von `table` mit ihrem Format aus `formats` (sonst `decimals` Nachkommastellen)."""
    formats = formats or {}
    return table.apply(lambda col: col.map(formats.get(col.name, f"{{:.{decimals}f}}").format))


def print_section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_table(table, formats=None, decimals=2):
    """Gibt eine Tabelle formatiert aus (siehe `format_table`)."""
    print(format_table(table, formats, decimals).to_string())


# Diagrammfarben (die Farbreihenfolge ist auf Farbenblindheit geprüft).
# Die Benchmark ist bewusst in neutralem Dunkelgrau gehalten.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCHMARK_COLOR = "#52514e"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def plot_performance(returns, perf, benchmark="Equal Weight", title="", subtitle="",
                     save_path=None, show=True):
    """Wertentwicklung von 1 investierten Geldeinheit je Strategie plus Kennzahlentabelle.

    `returns` hat eine Spalte pro Strategie (inkl. Benchmark), `perf` ist die
    Performance-Tabelle aus `performance_metrics` (eine Zeile pro Strategie).
    """
    # 1. Vermögensverlauf: jede Linie startet bei 1, eine Periode vor der ersten Rendite
    wealth = (1 + returns).cumprod()
    start = wealth.index[0] - (wealth.index[1] - wealth.index[0])
    wealth = pd.concat([pd.DataFrame(1.0, index=[start], columns=wealth.columns), wealth])

    # 2. Figur: oben das Diagramm, unten die Kennzahlentabelle
    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25},
    )
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # 3. Farben: Strategien der Reihe nach, die Benchmark in Dunkelgrau und als letzte Linie
    strategies = [c for c in wealth.columns if c != benchmark]
    colors = {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, name in enumerate(strategies)}
    if benchmark in wealth.columns:
        strategies.append(benchmark)
        colors[benchmark] = BENCHMARK_COLOR

    # 4. Linien mit Punkt am Ende
    for name in strategies:
        ax.plot(wealth.index, wealth[name], color=colors[name], lw=2,
                solid_capstyle="round", solid_joinstyle="round", label=name, zorder=3)
        ax.plot(wealth.index[-1], wealth[name].iloc[-1], "o", ms=8, color=colors[name],
                mec=SURFACE, mew=2, zorder=4)

    # 5. Beschriftung direkt am Linienende; entfällt bei Überlappung (die Legende nennt trotzdem jede Linie)
    y_range = wealth.to_numpy().max() - wealth.to_numpy().min()
    placed = []
    for name, value in wealth.iloc[-1].sort_values(ascending=False).items():
        if all(abs(value - p) > 0.05 * y_range for p in placed):
            ax.annotate(f"{name}  {value:.2f}", (wealth.index[-1], value), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=TEXT_SECONDARY, annotation_clip=False)
            placed.append(value)

    # 6. Achsen, Gitter, Legende, Titel
    ax.axhline(1.0, color=AXIS, lw=1, zorder=1)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0, pad=6)
    ax.set_ylabel("Value of 1 invested", color=TEXT_SECONDARY, fontsize=10)
    ax.set_xlim(wealth.index[0], wealth.index[-1])

    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="semibold",
                     color=TEXT_PRIMARY, pad=24 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=TEXT_SECONDARY)

    # 7. Kennzahlentabelle unter dem Diagramm
    ax_table.axis("off")
    formatted = format_table(perf, PERFORMANCE_FORMATS).reset_index(names="Strategy")
    table = ax_table.table(cellText=formatted.to_numpy(), colLabels=formatted.columns,
                           cellLoc="right", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(SURFACE)
        cell.set_edgecolor(GRID)
        cell.visible_edges = "B"
        cell.get_text().set_color(TEXT_SECONDARY if row == 0 else TEXT_PRIMARY)
        if row == 0:
            cell.get_text().set_fontweight("semibold")
        if col == 0:
            cell.get_text().set_horizontalalignment("left")

    # 8. Speichern / anzeigen
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.04)
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    plt.close(fig)
