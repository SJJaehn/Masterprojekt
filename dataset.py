"""
Schritt 1: Daten laden und die Stichprobe bilden.

Die Daten kommen aus der JKP-Datenbank (Jensen, Kelly & Pedersen, "Global
Factor Data"). Sie liegen als Panel vor: eine Zeile pro Monat und Aktie mit
- eom              Monatsende (Datum der Charakteristika)
- id               JKP-ID der Aktie
- ret_exc_lead1m   Überrendite der Aktie im FOLGEMONAT (das wollen wir prognostizieren)
- me               Marktkapitalisierung (für das Universum der 500 größten Aktien)
- 14 Charakteristika zum Zeitpunkt eom (Größe, B/M, Momentum, ...)
"""

import os

import numpy as np
import pandas as pd


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
# Laden
# =============================================================================

def load_data(path, top_k=1000, years_before=25, years_after=5, transform="rank"):
    """Lädt die JKP-Rohdaten (USA.parquet, 6 GB) analog zu b00PrepareDataset.m und b01LoadDataset.m.

    Wird nur einmal gebraucht: Das Ergebnis wird als kleine CSV in data/ gespeichert
    und danach von dort geladen (siehe `load_prepared_data`).

    1. Nur die benötigten Spalten lesen (wie im Tutor-Skript, plus `me` als
       untransformierte Marktkapitalisierung für die Auswahl des Aktienuniversums)
    2. Zeilen ohne Rendite oder Marktkapitalisierung sowie nano/micro Stocks entfernen
    3. Charakteristika pro Monat im Querschnitt transformieren (Standard: Ränge
       auf [-0.5, 0.5], wie fCrossSectTransChars.m)
    4. Nur Aktien behalten, die irgendwann zu den `top_k` größten gehören, und
       von diesen die Daten der `years_before` Jahre davor (Training) und der
       `years_after` Jahre danach (Prognose) - das macht die Datei klein genug für Git
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

    Nur nötig, wenn die Zielgröße KEINE Überrendite ist (bei JKP ist sie es schon).
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


# =============================================================================
# Stichprobe
# =============================================================================

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

    1. Nur Monate ab `start`
    2. Nur Aktien mit Werten für alle Charakteristika (`drop_missing`, wie im Paper)
    3. Davon pro Monat die `universe_size` größten Aktien nach `size_col` (None: alle)
    """
    if start is not None:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if drop_missing:
        df = df.dropna(subset=feature_cols)
    if universe_size:
        size_rank = df.groupby(date_col)[size_col].rank(ascending=False, method="first")  # 1 = größte Aktie
        df = df[size_rank <= universe_size]
    return df.reset_index(drop=True)
