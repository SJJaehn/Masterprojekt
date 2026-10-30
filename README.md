# Masterprojekt: Aktienrenditen mit Entscheidungsbäumen prognostizieren

Replikation von **Lewellen (2015), "The Cross-Section of Expected Stock Returns"**
(`ssrn-2511246.pdf` in diesem Ordner) – mit einer Änderung: Wo Lewellen eine
**lineare Regression** schätzt, verwenden wir **Machine Learning**:

| Modell | Idee in einem Satz |
| --- | --- |
| **OLS** (Benchmark) | Die lineare Regression aus dem Paper. Läuft mit, damit wir einen direkten Vergleich haben. |
| **Decision Tree** | Teilt die Aktien mit einer Folge von Ja/Nein-Fragen zu ihren Kennzahlen in Gruppen; die Prognose ist die Durchschnittsrendite der Gruppe. |
| **Random Forest** | Viele leicht unterschiedliche Bäume, deren Prognosen gemittelt werden. |
| **XGBoost** | Viele kleine Bäume nacheinander; jeder neue Baum korrigiert die Fehler der bisherigen. |

Aus Rechenzeitgründen verwenden wir pro Monat nur die **500 größten US-Aktien**.

Diese README erklärt das Verfahren Schritt für Schritt und zeigt bei jedem
Schritt die Code-Zeilen, die ihn umsetzen. Sie ist für Leser mit
Finance-Hintergrund geschrieben; Programmierkenntnisse sind nicht nötig.

---

## Inhalt

- [Schnellstart](#schnellstart)
- [Aufbau des Projekts](#aufbau-des-projekts)
- [Die Idee des Papers in fünf Sätzen](#die-idee-des-papers-in-fünf-sätzen)
- [Schritt 1: Daten und Stichprobe](#schritt-1-daten-und-stichprobe)
- [Schritt 2: Erwartete Renditen schätzen](#schritt-2-erwartete-renditen-schätzen) ← der Kern
- [Schritt 3: Fama-MacBeth-Regressionen (Tabelle 2)](#schritt-3-fama-macbeth-regressionen-tabelle-2)
- [Schritt 4: Wie gut sind die Prognosen? (Tabelle 3)](#schritt-4-wie-gut-sind-die-prognosen-tabelle-3)
- [Schritt 5: Dezilportfolios (Tabelle 6)](#schritt-5-dezilportfolios-tabelle-6)
- [Schritt 6: Handelsstrategie](#schritt-6-handelsstrategie)
- [Ergebnisse](#ergebnisse)
- [Unterschiede zum Paper](#unterschiede-zum-paper)
- [Einstellungen ändern](#einstellungen-ändern)
- [Glossar](#glossar)

---

## Schnellstart

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

Das Skript lädt die vorbereiteten Daten aus `data/`, rechnet alle sechs
Schritte nacheinander und schreibt die Tabellen in die Konsole. Ergebnisse
(CSV-Dateien und eine Grafik) landen im Ordner `output/`. Ein kompletter Lauf
mit allen vier Modellen dauert auf einem Rechner mit 4 Kernen etwa
7 Minuten; der Random Forest braucht davon am längsten. Mit Grid Search
entsprechend ein Vielfaches.

Zum schnellen Ausprobieren: In `main.py` `USE_DUMMY_DATA = True` setzen
(Zufallsdaten, läuft in wenigen Minuten) oder `MODELS_TO_RUN` auf weniger
Modelle kürzen.

## Aufbau des Projekts

| Datei | Inhalt |
| --- | --- |
| `main.py` | **Hier anfangen.** Alle Einstellungen oben, darunter die sechs Schritte als Funktionen, ganz unten `main()`, die sie nacheinander aufruft. |
| `dataset.py` | Schritt 1: Daten laden, Charakteristika, Stichprobe |
| `models.py` | Schritt 2: die vier Modelle und die Fama-MacBeth-Schätzung |
| `evaluation.py` | Schritte 3–6: Regressionen, Tabellen 2/3/6, Handelsstrategie, Kennzahlen |
| `report.py` | Ausgabe: Tabellen formatieren, Grafik zeichnen (keine Rechnung) |
| `data/jkp_top1000_25y_5y_rank.csv.gz` | Vorbereitete Daten (aus den 6-GB-Rohdaten des Tutors erzeugt) |
| `Thema 3 - Stock Return Prediction/` | MATLAB-Skripte des Tutors, an denen sich die Datenaufbereitung orientiert |
| `ssrn-2511246.pdf` | Das Paper |

`main()` liest sich wie das Inhaltsverzeichnis des Verfahrens:

```python
def main():
    check_settings()
    df, feature_cols, rf = load_sample()                       # Schritt 1
    preds, importances, best_params = estimate(df, feature_cols)  # Schritt 2
    fm_table, fm_monthly = paper_regressions(df, feature_cols, preds)  # Schritt 3
    table3, metrics, correlations = evaluate_forecasts(preds)  # Schritt 4
    portfolios = portfolio_sorts(preds)                        # Schritt 5
    strategy, perf = trading_strategy(preds, rf)               # Schritt 6
```

> **Lesehilfe Python.** `df` ist eine Tabelle (wie ein Excel-Blatt), `df["me"]`
> ist eine Spalte daraus. `for x in liste:` wiederholt den eingerückten Block
> für jedes Element. Text nach `#` ist ein Kommentar und wird nicht ausgeführt.
> `model.fit(X, y)` heißt "Modell auf X und y schätzen", `model.predict(X)`
> heißt "Prognose für X berechnen" – diese beiden Befehle gelten für alle vier
> Modelle gleich.

---

## Die Idee des Papers in fünf Sätzen

1. In jedem Monat wird eine **Querschnittsregression** geschätzt: die Renditen
   aller Aktien im nächsten Monat auf ihre heutigen Kennzahlen (Größe, B/M,
   Momentum, …). Das ist die Fama-MacBeth-Methode.
2. Die Steigungen dieser Regressionen schwanken stark von Monat zu Monat.
   Lewellen **mittelt sie daher über die letzten 10 Jahre** (120 Monate).
3. Mit diesen gemittelten Steigungen und den heutigen Kennzahlen einer Aktie
   berechnet er ihre **erwartete Rendite für den nächsten Monat** – nur mit
   Informationen, die zu diesem Zeitpunkt bekannt waren (out-of-sample).
4. Dann prüft er, ob die Prognosen etwas taugen: Haben Aktien mit hoher
   Prognose tatsächlich höhere Renditen? (Tabelle 3: Regression der
   realisierten auf die prognostizierten Renditen; Tabelle 6: Dezilportfolios.)
5. Ergebnis im Paper: Ja – die Prognosen streuen stark und sagen die
   tatsächlichen Renditen gut voraus.

**Unsere Frage:** Werden die Prognosen besser, wenn wir statt der linearen
Regression Entscheidungsbäume verwenden, die auch nichtlineare Zusammenhänge
und Wechselwirkungen zwischen den Kennzahlen erkennen können?

---

## Schritt 1: Daten und Stichprobe

**Code:** `load_sample()` in `main.py`, Funktionen in `dataset.py`

### Die Daten

Die Daten kommen aus der **JKP-Datenbank** (Jensen, Kelly & Pedersen, "Global
Factor Data"). Sie bilden ein **Panel**: eine Zeile pro Monat und Aktie.

| eom | id | ret_exc_lead1m | me | market_equity | be_me | ret_12_1 | … |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1990-01-31 | 10107 | 0.0412 | 3 912 | 0.48 | -0.21 | 0.33 | … |
| 1990-01-31 | 10145 | -0.0170 | 1 507 | 0.39 | 0.12 | -0.05 | … |
| … | | | | | | | |

- `eom` – Monatsende, zu dem die Kennzahlen gemessen werden (Monat *t*)
- `ret_exc_lead1m` – **Überrendite der Aktie im Folgemonat** *t + 1* (das wollen wir prognostizieren)
- `me` – Marktkapitalisierung in Mio. USD (nur für die Auswahl der 500 größten und für Wertgewichte)
- 14 **Charakteristika** zum Zeitpunkt *t*

Die Zeilen eines Monats bilden einen **Querschnitt**: rund 500 Aktien × 14 Kennzahlen.
Wichtig für das Timing: Kennzahlen am Ende von Monat *t*, Rendite über Monat
*t + 1*. Dadurch wird nie eine Rendite mit Informationen "erklärt", die es zu
Beginn der Renditeperiode noch nicht gab.

### Die 14 Charakteristika (Lewellens Modell 3)

Festgelegt im Tutor-Skript `b00PrepareDataset.m`, in Python in `dataset.py`:

```python
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
```

Lewellens drei Modelle sind Teilmengen davon (Einstellung `FEATURE_SET` in `main.py`):

```python
LEWELLEN_MODELS = {
    "Model 1": JKP_CHARACTERISTICS[:3],   # Größe, B/M, Momentum
    "Model 2": JKP_CHARACTERISTICS[:7],   # + Emissionen, Accruals, Profitabilität, Asset Growth
    "Model 3": JKP_CHARACTERISTICS,       # alle 14
}
```

(`[:3]` heißt "die ersten drei Einträge der Liste".)

### Rangtransformation

Wie im Tutor-Skript `fCrossSectTransChars.m` wird jede Kennzahl **in jedem
Monat in Ränge umgerechnet** und auf das Intervall **[−0,5; +0,5]** skaliert.
Beispiel mit fünf Aktien und ihrem B/M in einem Monat:

| Aktie | B/M | Rang | skaliert: (Rang − 1) / (5 − 1) − 0,5 |
| --- | --- | --- | --- |
| A | 0,20 | 1 | −0,50 |
| B | 0,45 | 2 | −0,25 |
| C | 0,60 | 3 | 0,00 |
| D | 1,10 | 4 | +0,25 |
| E | 3,50 | 5 | +0,50 |

Warum? Ausreißer (Aktie E) verlieren ihren übergroßen Einfluss, und alle
Kennzahlen haben in jedem Monat dieselbe Skala. Die Steigung einer Regression
auf eine Rangvariable lässt sich direkt lesen: **Renditeunterschied zwischen der
Aktie mit dem niedrigsten und der mit dem höchsten Wert**.

```python
ranks = grouped.rank(method="average")  # gleiche Werte erhalten den mittleren Rang (wie tiedrank)
df[JKP_CHARACTERISTICS] = (ranks - 1) / (ranks.groupby(df["eom"]).transform("max") - 1) - 0.5
```

Diese Aufbereitung (`dataset.load_data`) läuft nur einmal auf den 6-GB-Rohdaten
des Tutors. Das Ergebnis ist die Datei in `data/`, die `main.py` danach direkt lädt.

### Stichprobe: 500 größte Aktien mit vollständigen Daten

```python
def select_sample(df, date_col, size_col, feature_cols, start=None, drop_missing=True, universe_size=None):
    if start is not None:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if drop_missing:
        df = df.dropna(subset=feature_cols)
    if universe_size:
        size_rank = df.groupby(date_col)[size_col].rank(ascending=False, method="first")  # 1 = größte Aktie
        df = df[size_rank <= universe_size]
    return df.reset_index(drop=True)
```

Drei Filter, Zeile für Zeile:

1. Nur Monate **ab 1964** (wie im Paper: ab dann gibt es Compustat-Daten).
2. `dropna`: nur Aktien, für die **alle 14 Kennzahlen** vorliegen (wie im Paper).
3. In jedem Monat (`groupby(date_col)`) die Aktien nach Marktkapitalisierung
   absteigend ranken und nur die **Ränge 1 bis 500** behalten.

Das Universum wird jeden Monat neu bestimmt; eine Aktie kann also hinein- und
herausfallen. Da die Marktkapitalisierung am Monatsende *t* bekannt ist,
entsteht dadurch kein Look-ahead-Bias.

---

## Schritt 2: Erwartete Renditen schätzen

**Code:** `estimate()` in `main.py`, `run_fama_macbeth()` und
`average_of_monthly_models()` in `models.py`

Das ist der Kern der Arbeit. Wir gehen in drei Teilen vor: erst Lewellens
Verfahren, dann die Übertragung auf Machine Learning, dann der Code.

### 2a) Lewellens Verfahren

In **jedem Monat j** schätzt Lewellen eine Querschnittsregression:

```
r(i, j+1) = a(j) + b1(j) · Größe(i, j) + b2(j) · B/M(i, j) + … + Fehler
```

Das ergibt pro Monat einen Satz Koeffizienten a(j), b(j). Für die Prognose am
Ende von Monat *t* mittelt er die Koeffizienten der **letzten 120 Monate**:

```
ā = Mittelwert von a(t−120) … a(t−1)
b̄ = Mittelwert von b(t−120) … b(t−1)

Prognose(i, t) = ā + b̄1 · Größe(i, t) + b̄2 · B/M(i, t) + …
```

### 2b) Übertragung auf Machine Learning

Ein Entscheidungsbaum hat keine Steigungen, die man mitteln könnte. Wir nutzen
deshalb eine Umformung, die für die Regression **exakt dasselbe** ergibt:
Statt erst die Koeffizienten zu mitteln und dann zu prognostizieren, lassen wir
**jedes der 120 Monatsmodelle eine eigene Prognose** abgeben und **mitteln die
Prognosen**:

```
Mittel der Prognosen = 1/120 · Σ [ a(j) + b(j) · x ]
                     = 1/120 · Σ a(j)  +  (1/120 · Σ b(j)) · x
                     = ā + b̄ · x          ← genau Lewellens Prognose
```

Weil das Mitteln bei einem linearen Modell egal ist, ist unsere OLS-Prognose
**identisch mit dem Paper**. Für Bäume funktioniert dieselbe Rechnung genauso –
man ersetzt nur die Regression durch einen Baum. Damit bleibt alles andere
(Daten, Timing, Fenster, Auswertung) wie im Paper, und der einzige Unterschied
zwischen den Modellen ist das Modell selbst. Das macht den Vergleich sauber.

Zeitlich sieht das für den ersten Prognosemonat so aus (Fenster von 120 Monaten):

```
Monatsmodelle j:  1964-01   1964-02   …   1973-11   1973-12      (120 Modelle,
                  Kennzahlen Ende j  →  Rendite im Monat j + 1     je ~500 Aktien)
                     │         │             │         │
                     └─────────┴──── Mittel ─┴─────────┘
                                       │
                                       ▼
Prognose:   Kennzahlen Ende Januar 1974  →  erwartete Rendite im Februar 1974
```

Einen Monat später fällt das Modell von 1964-01 heraus und das von 1974-01
kommt hinzu – das Fenster rollt.

Präziser, mit den Zeilen des Datensatzes: Eine Zeile mit Datum *t* enthält die
Kennzahlen am Ende von *t* und die Rendite über *t + 1*.

- Monatsmodell *j* lernt aus den Zeilen von Monat *j*:
  **Kennzahlen Ende j → Rendite im Monat j + 1**.
- Die Prognose für die Zeilen von Monat *t* (also für die Rendite über *t + 1*)
  ist das Mittel der Modelle *j = t − 120, …, t − 1*.
- Das jüngste dieser Modelle (*j = t − 1*) braucht die Rendite über Monat *t* –
  die ist am Ende von *t* bekannt. **Es wird also nie in die Zukunft geschaut.**

### 2c) Die vier Modelle

Jedes Monatsmodell bekommt die rund 500 Aktien eines Monats mit ihren 14
(rangtransformierten) Kennzahlen und deren Folgemonatsrenditen.

**OLS** – die lineare Regression von oben. Eine Steigung pro Kennzahl.

**Decision Tree.** Der Baum sucht die Ja/Nein-Frage zu einer Kennzahl, die die
Aktien in zwei Gruppen mit möglichst unterschiedlichen Durchschnittsrenditen
teilt (genauer: die den quadrierten Prognosefehler am stärksten senkt). Dann
macht er in jeder Gruppe dasselbe, bis eine Abbruchregel greift. Die Prognose für
eine Aktie ist die Durchschnittsrendite der Endgruppe ("Blatt"), in der sie landet.
Ein **erfundenes** Beispiel zur Anschauung:

```
                     Momentum-Rang > 0,2 ?
                    /                     \
                 nein                      ja
       Größe-Rang < −0,3 ?           Volatilitäts-Rang > 0,4 ?
          /         \                   /            \
      +1,1 %      +0,7 %            +1,3 %         +0,6 %
```

Anders als die Regression kann der Baum so **Schwellen** ("nur extreme
Momentum-Aktien") und **Wechselwirkungen** ("Momentum wirkt nur bei kleinen
Aktien") abbilden, ohne dass wir sie vorher festlegen.

Das Risiko: Ein unbegrenzter Baum teilt so lange, bis jede Gruppe nur noch eine
Aktie enthält – dann hat er das Rauschen eines Monats auswendig gelernt
(**Overfitting**). Monatsrenditen bestehen zum größten Teil aus Rauschen; das
Begrenzen ist daher entscheidend. Das tun die **Hyperparameter**:

```python
model_params = {
    "OLS": {},
    "DecisionTree": {"max_depth": 5, "min_samples_leaf": 50},
    "RandomForest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50,
                     "max_features": "sqrt", "n_jobs": -1},
    "XGBoost": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1},
}
```

| Hyperparameter | Bedeutung | Wert |
| --- | --- | --- |
| `max_depth` | höchstens so viele Fragen hintereinander | Baum 5, Forest 6, XGBoost 3 |
| `min_samples_leaf` | mindestens so viele Aktien pro Endgruppe – die Prognose ist also immer ein Durchschnitt über ≥ 50 Aktien | 50 |
| `n_estimators` | Anzahl der Bäume | 200 |
| `max_features` | Random Forest: bei jeder Frage nur aus einer zufälligen Auswahl der Kennzahlen wählen (`"sqrt"`: √14 ≈ 3) | `"sqrt"` |
| `learning_rate` | XGBoost: Gewicht, mit dem jeder neue Baum in die Prognose eingeht | 0,05 |
| `subsample`, `colsample_bytree` | XGBoost: jeder Baum sieht nur 80 % der Aktien bzw. Kennzahlen | 0,8 |
| `n_jobs = -1` | alle Prozessorkerne nutzen (ändert nichts am Ergebnis) | |

**Random Forest** (Bagging). Ein einzelner Baum ist instabil: Ein paar andere
Aktien in der Stichprobe, und er stellt ganz andere Fragen. Der Random Forest
trainiert daher 200 Bäume, jeden auf einer zufälligen Ziehung der Aktien (mit
Zurücklegen) und mit zufälliger Auswahl der Kennzahlen pro Frage, und **mittelt
ihre Prognosen**. Das ist Diversifikation: Die Fehler der einzelnen Bäume heben
sich teilweise auf, wie das unsystematische Risiko in einem Portfolio.

**XGBoost** (Boosting). Die Bäume werden **nacheinander** gebaut. Der erste
macht eine grobe Prognose; jeder weitere Baum wird auf die **verbleibenden
Fehler** (Residuen) der bisherigen Prognose trainiert und mit dem Gewicht
`learning_rate` (5 %) addiert. Die vielen kleinen Korrekturen (Tiefe nur 3)
bauen langsam ein feineres Bild auf.

Die Modelle werden in `models.py` angelegt; `random_state` legt den Startwert
des Zufallsgenerators fest, damit jeder Lauf dieselben Zahlen liefert:

```python
MODEL_CLASSES = {
    "OLS": LinearRegression,              # Benchmark: mit run_fama_macbeth genau das Modell aus dem Paper
    "DecisionTree": DecisionTreeRegressor,  # ein einzelner Entscheidungsbaum
    "RandomForest": RandomForestRegressor,  # Mittel vieler unabhängiger Bäume (Bagging)
    "XGBoost": XGBRegressor,              # Bäume nacheinander, jeder korrigiert die Fehler der vorigen (Boosting)
}
```

### 2d) Der Code der Schätzung

Die ganze Logik von 2a und 2b steckt in einer Funktion,
`average_of_monthly_models()` in `models.py`. Gekürzt auf das Wesentliche:

```python
forecast_sum = np.zeros(len(y))  # Summe der Prognosen je Aktie und Monat
n_models = np.zeros(len(y))      # Anzahl der Monatsmodelle, die diese Prognose enthält

first_model = 0 if expanding else max(first - train_window, 0)
for j in range(first_model, last):
    # 1. Modell auf dem Querschnitt von Monat j trainieren (ca. 500 Aktien x 14 Charakteristika)
    in_month_j = month_nr == j
    fitted = clone(model).fit(X[in_month_j], y[in_month_j])

    # 2. Damit alle Aktien der Monate j + 1 ... j + train_window prognostizieren
    last_month = last if expanding else min(j + train_window, last)
    in_forecast_months = (month_nr > j) & (month_nr >= first) & (month_nr <= last_month)
    forecast_sum[in_forecast_months] += fitted.predict(X[in_forecast_months])
    n_models[in_forecast_months] += 1

# 3. Mittelwert: Summe der Prognosen / Anzahl der Modelle
return forecast_sum / n_models, importances
```

Was hier passiert:

- `month_nr` nummeriert die Monate durch (0 = Januar 1964, 1 = Februar 1964, …);
  `first` und `last` sind die Nummern des ersten und letzten Prognosemonats
  (Januar 1974 und Dezember 2013).
- `X` sind die Kennzahlen aller Zeilen, `y` die Folgemonatsrenditen.
  `X[in_month_j]` wählt die Zeilen von Monat *j* aus.
- Die Schleife `for j in …` geht Monat für Monat durch. In jedem Monat wird
  **ein frisches Modell** (`clone(model)`) auf genau diesem einen Querschnitt
  geschätzt (`.fit`).
- Statt "für Monat *t* die Modelle *t − 120 … t − 1* einsammeln" dreht der Code
  die Blickrichtung um: **Jedes Modell *j* prognostiziert sofort alle Monate,
  in deren Fenster es liegt** (*j + 1 … j + 120*), und legt seine Prognose in
  `forecast_sum` ab. `n_models` zählt mit, wie viele Modelle zu jeder Prognose
  beigetragen haben. Das ergibt dasselbe, ist aber viel schneller, weil jedes
  Modell nur einmal aufgerufen wird.
- Am Ende ist `forecast_sum / n_models` der **Mittelwert der Prognosen der
  letzten 120 Monatsmodelle** – für jede Aktie in jedem Monat.

`run_fama_macbeth()` ruft diese Funktion für jedes der vier Modelle auf und
sammelt die Ergebnisse in einer Tabelle `predictions` mit einer Zeile pro
(Monat, Aktie): realisierte Rendite plus je eine Prognosespalte `OLS`,
`DecisionTree`, `RandomForest`, `XGBoost`. Alle folgenden Schritte arbeiten nur
noch mit dieser Tabelle.

### 2e) Feature Importance

Bäume berichten, wie stark jede Kennzahl den Prognosefehler beim Aufteilen
gesenkt hat (`feature_importances_`). Wir mitteln das über alle Monatsmodelle;
`main.py` druckt die zehn wichtigsten Kennzahlen. Das ist das Gegenstück zu den
t-Statistiken der Regression – es zeigt, worauf die Bäume ihre Prognosen
stützen, aber nicht in welche Richtung.

### 2f) Optional: Hyperparameter jeden Monat neu wählen (Grid Search)

Mit `GRID_SEARCH = True` werden die Hyperparameter nicht fest vorgegeben,
sondern aus einem Raster gewählt:

```python
param_grids = {
    "DecisionTree": {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50, 100]},
    "RandomForest": {"max_depth": [3, 6], "max_features": ["sqrt", 0.5]},
    "XGBoost": {"max_depth": [2, 3, 5], "learning_rate": [0.01, 0.05, 0.1]},
}
```

Für jede Kombination (beim Baum 3 × 3 = 9) wird das komplette Verfahren aus 2d
gerechnet. In jedem Prognosemonat *t* wird dann die Kombination verwendet,
deren Prognosen in den **12 Monaten vor t** den kleinsten mittleren quadratischen
Fehler hatten (`choose_params_each_month()` in `models.py`). Auch das nutzt nur
Daten, die in *t* bekannt sind. Der Lauf dauert entsprechend 4- bis 9-mal so lange.

### 2g) Optional: gepoolte Schätzung

Mit `ESTIMATION = "pooled"` wird statt 120 Monatsmodellen **ein einziges Modell**
auf allen 120 Querschnitten zusammen trainiert (120 × 500 = 60 000 Zeilen) und
jeden Monat neu geschätzt (`run_pooled()` in `models.py`). Das ist der übliche
Ansatz der Machine-Learning-Literatur (z. B. Gu, Kelly & Xiu, 2020): Der Baum hat
mehr Daten pro Schätzung, weicht aber stärker vom Verfahren des Papers ab.

---

## Schritt 3: Fama-MacBeth-Regressionen (Tabelle 2)

**Code:** `paper_regressions()` in `main.py`, `fama_macbeth()` in `evaluation.py`

Zur Einordnung reproduzieren wir Lewellens Tabelle 2: die über alle Monate
gemittelten Steigungen der Querschnittsregressionen, für unsere Stichprobe.
(Code-Auszüge in dieser README sind teils leicht gekürzt.)

```python
# 1. Querschnittsregression pro Monat
monthly = []
for date, g in df.groupby(date_col):
    fit = sm.OLS(g[y_col], sm.add_constant(g[x_cols])).fit()
    monthly.append({date_col: date, **fit.params.rename({"const": "Intercept"}),
                    "R2": fit.rsquared, "N": int(fit.nobs)})

# 2. Mittelwert und Newey-West-t-Statistik der monatlichen Koeffizienten
coefs = monthly.drop(columns=["R2", "N"])
table = pd.DataFrame([newey_west_test(coefs[c], nw_lags) for c in coefs],
                     index=coefs.columns, columns=["Coef", "S.E.", "t-stat"])
```

1. `df.groupby(date_col)` teilt die Daten in Monate; für jeden Monat `g` wird
   mit `statsmodels` eine OLS-Regression der Renditen auf die Kennzahlen plus
   Konstante geschätzt und die Koeffizienten werden gespeichert.
2. Für jede Kennzahl ergibt das eine Zeitreihe von ~600 Steigungen. Deren
   **Mittelwert** ist der Fama-MacBeth-Koeffizient; die **t-Statistik**
   kommt aus der Streuung dieser Zeitreihe.

Wie im Paper werden die Standardfehler nach **Newey-West mit 4 Lags** korrigiert,
weil die monatlichen Steigungen autokorreliert sein können. Technisch ist das
eine Regression der Zeitreihe auf eine Konstante (= ihr Mittelwert) mit
HAC-Standardfehlern:

```python
def newey_west_test(series, lags=4):
    y = pd.Series(series).dropna().to_numpy()
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0], fit.bse[0], fit.tvalues[0]
```

**Lesart:** Da die Kennzahlen Ränge in [−0,5; 0,5] sind, ist eine Steigung von
0,50 (in % pro Monat) beim Momentum so zu lesen: Die Aktie mit dem höchsten
Momentum hat – bei sonst gleichen Kennzahlen – eine um 0,5 Prozentpunkte pro
Monat höhere erwartete Rendite als die mit dem niedrigsten.

---

## Schritt 4: Wie gut sind die Prognosen? (Tabelle 3)

**Code:** `evaluate_forecasts()` in `main.py`, `forecast_evaluation()` und
`prediction_metrics()` in `evaluation.py`

### Tabelle 3 wie im Paper

```python
dist = preds.groupby(date_col)[pred_col].describe(percentiles=[0.1, 0.9]).mean() * 100
table, monthly = fama_macbeth(preds, target_col, [pred_col], date_col, nw_lags)
```

**Linker Teil – Verteilung der Prognosen** (`dist`): In jedem Monat Mittelwert,
Standardabweichung sowie 10 %- und 90 %-Quantil der Prognosen über alle Aktien,
dann über die Monate gemittelt. Eine große Streuung heißt: Das Modell traut sich,
Aktien deutlich zu unterscheiden.

**Rechter Teil – Prognosegüte**: Dieselbe Fama-MacBeth-Funktion aus Schritt 3,
aber jetzt mit nur einer erklärenden Variablen – der Prognose:

```
r(i, t+1) = α(t) + β(t) · Prognose(i, t) + Fehler
```

Die gemittelte Steigung β ist die zentrale Kennzahl des Papers:

| Steigung | Bedeutung |
| --- | --- |
| **β = 1** | Ideal: Ein Prozentpunkt mehr Prognose bedeutet im Schnitt einen Prozentpunkt mehr Rendite. |
| **0 < β < 1** | Die Prognose trennt gute und schlechte Aktien, übertreibt aber die Unterschiede. |
| **β = 0** | Die Prognose ist wertlos. |

Die t-Statistik (Newey-West) zeigt, ob β signifikant von null verschieden ist.
Lewellen findet für große Aktien Steigungen von 0,44–0,72.

### Weitere Kennzahlen aus der Machine-Learning-Literatur

| Kennzahl | Bedeutung |
| --- | --- |
| `RMSE`, `MAE` | durchschnittlicher Prognosefehler (Wurzel des mittleren quadratischen bzw. mittlerer absoluter Fehler) |
| `R2 OOS` | 1 − Σ(r − Prognose)² / Σ r² nach Gu, Kelly & Xiu (2020): Vergleich mit einer Prognose von null. Schon leicht positive Werte sind bei Monatsrenditen gut. |
| `IC mean` | **Information Coefficient**: Rangkorrelation zwischen Prognose und realisierter Rendite in jedem Monat, gemittelt. Misst, ob die *Reihenfolge* der Aktien stimmt – das ist für eine Long-Short-Strategie entscheidend. |
| `IC IR`, `IC t-stat` | Mittelwert / Standardabweichung des IC bzw. t-Test, ob der IC über die Zeit verlässlich positiv ist |
| `Hit rate` | Anteil der Prognosen mit dem richtigen Vorzeichen |

Außerdem zeigt `main.py` die mittlere Korrelation der Prognosen verschiedener
Modelle (vgl. Lewellens Tabelle 4): Sagen die Bäume etwas anderes als OLS?

---

## Schritt 5: Dezilportfolios (Tabelle 6)

**Code:** `portfolio_sorts()` in `main.py`, `sorted_portfolios()` in `evaluation.py`

Jeden Monat werden die Aktien nach ihrer Prognose in **10 gleich große
Portfolios** sortiert (Dezil 1 = niedrigste Prognose, Dezil 10 = höchste):

```python
df["portfolio"] = df.groupby(date_col)[pred_col].transform(
    lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False) + 1)
```

Für jedes Portfolio wird die Rendite im Folgemonat berechnet –
**gleichgewichtet (EW)** und **wertgewichtet (VW)** nach Marktkapitalisierung –
und zusätzlich das **High-minus-Low-Portfolio H−L** (Dezil 10 long, Dezil 1 short):

```python
pred, ret = means[pred_col].unstack(), means[target_col].unstack()  # Monate x Portfolios
pred["H-L"] = pred[n_groups] - pred[1]
ret["H-L"] = ret[n_groups] - ret[1]
```

Die Tabelle zeigt pro Portfolio die mittlere **Prognose** (`Pred`), die
mittlere **realisierte Rendite** (`Avg`), deren Standardabweichung, die
annualisierte Sharpe Ratio und die Newey-West-t-Statistik.

**Lesart:** Ein gutes Modell zeigt eine **monoton steigende** Rendite von Low
nach High und eine signifikant positive H−L-Rendite. `Pred` und `Avg`
nebeneinander zeigen, ob die Prognosen die tatsächlichen Unterschiede über-
oder unterschätzen (vgl. die Steigung aus Schritt 4).

---

## Schritt 6: Handelsstrategie

**Code:** `trading_strategy()` in `main.py`, `strategy_returns()` und
`performance_metrics()` in `evaluation.py`

Zum Schluss eine einfache, umsetzbare Strategie: **Jeden Monat die 20 % der
Aktien mit der höchsten Prognose gleichgewichtet kaufen** (`LONG_PCT = 0.20`),
optional die schlechtesten x % leerverkaufen (`SHORT_PCT`). Benchmark ist das
gleichgewichtete Portfolio aller 500 Aktien ("Equal Weight").

```python
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
```

Für jeden Monat: Aktien nach Prognose sortieren, die ersten 20 % (`iloc[:n_long]`)
nehmen, deren durchschnittliche realisierte Rendite ist die Portfoliorendite.

Daraus die Kennzahlen:

```python
wealth = (1 + returns).cumprod()
...
"Ann. Return": wealth.iloc[-1] ** (periods_per_year / len(returns)) - 1,
"Ann. Volatility": returns.std() * np.sqrt(periods_per_year),
"Sharpe Ratio": excess.mean() / excess.std() * np.sqrt(periods_per_year),
"Max Drawdown": (wealth / peak - 1).min(),
```

| Kennzahl | Berechnung |
| --- | --- |
| Ann. Return | geometrische Rendite p. a. (CAGR) aus dem Endvermögen |
| Ann. Volatility | Standardabweichung der Monatsrenditen × √12 |
| Sharpe Ratio | mittlere Überrendite / Standardabweichung × √12 (die JKP-Renditen sind bereits Überrenditen über den risikofreien Zins) |
| Max Drawdown | größter prozentualer Verlust vom bisherigen Höchststand |

Die Grafik `output/performance.png` zeigt die Wertentwicklung von 1 investierten
Dollar je Modell gegen Equal Weight.

Hinweis: Transaktionskosten sind nicht berücksichtigt; da jeden Monat neu
sortiert wird, kann der Umschlag hoch sein.

---

## Ergebnisse

Ein Lauf mit den Standardeinstellungen (Modell 3, 500 größte Aktien,
10-Jahres-Fenster, Prognosen 1974–2013, 480 Monate). Alle Zahlen stammen aus
`python main.py`; sie sind als Ausgangspunkt für die eigene Interpretation
gedacht.

**Tabelle 3 – Prognosegüte**

| Modell | Std der Prognosen (%) | Steigung | t-Stat | IC mean | IC t-Stat |
| --- | --- | --- | --- | --- | --- |
| OLS | 0,41 | 0,42 | 2,28 | 0,027 | 3,83 |
| Decision Tree | 0,36 | 0,31 | 1,72 | 0,016 | 2,62 |
| Random Forest | 0,15 | 0,59 | 1,20 | 0,014 | 2,09 |
| XGBoost | 0,45 | 0,37 | 2,63 | 0,022 | 3,71 |

**Tabelle 6 – Dezilportfolios, gleichgewichtet (realisierte Rendite in % pro Monat)**

| Modell | Low | High | H−L | t-Stat H−L |
| --- | --- | --- | --- | --- |
| OLS | 0,32 | 1,03 | 0,71 | 2,56 |
| Decision Tree | 0,51 | 0,90 | 0,39 | 1,49 |
| Random Forest | 0,52 | 0,89 | 0,37 | 1,22 |
| XGBoost | 0,41 | 1,00 | 0,58 | 2,27 |

**Handelsstrategie – Long Top 20 %**

| Strategie | Rendite p. a. | Volatilität p. a. | Sharpe Ratio | Max Drawdown |
| --- | --- | --- | --- | --- |
| OLS | 9,83 % | 18,45 % | 0,60 | −54,93 % |
| Decision Tree | 8,67 % | 18,95 % | 0,54 | −58,81 % |
| Random Forest | 7,63 % | 19,14 % | 0,48 | −72,18 % |
| XGBoost | 9,30 % | 19,33 % | 0,56 | −63,44 % |
| Equal Weight | 7,06 % | 16,61 % | 0,50 | −53,61 % |

Beobachtungen, die sich direkt aus den Zahlen ablesen lassen:

- Alle vier Modelle haben einen positiven, signifikanten IC: Sie sortieren
  die Aktien besser als der Zufall.
- In dieser Einstellung liefert **OLS** die höchste H−L-Rendite, den höchsten
  IC und die beste Strategie; **XGBoost** kommt am nächsten heran.
- Der **Random Forest** prognostiziert sehr vorsichtig (Streuung 0,15 % statt
  0,41 %): Das Mitteln über 200 Bäume *und* über 120 Monatsmodelle glättet
  die Prognosen stark. Die Steigung ist deshalb nahe an der Idealrichtung,
  aber unpräzise geschätzt (t = 1,20).
- Die wichtigsten Kennzahlen für die Bäume sind Volatilität, Beta und
  Momentum (Feature Importance in der Konsolenausgabe).
- Die OLS-Steigung von 0,42 liegt am unteren Rand von Lewellens Werten für
  große Aktien (0,44–0,72, Tabelle 3, Panel C).

Mögliche Ansatzpunkte für die Arbeit: Grid Search (`GRID_SEARCH = True`), die
gepoolte Schätzung (`ESTIMATION = "pooled"`), die mehr Daten pro Baum hat,
sowie Modell 1 und 2 im Vergleich.

---

## Unterschiede zum Paper

| | Lewellen (2015) | Hier |
| --- | --- | --- |
| Modell | lineare Regression | OLS (identisches Verfahren) **plus** Decision Tree, Random Forest, XGBoost |
| Daten | CRSP/Compustat | JKP Global Factor Data (Kennzahlen als Annäherung an Lewellens Definitionen, siehe Kommentare in `b00PrepareDataset.m`) |
| Aktien | alle / "all-but-tiny" / "large" (> NYSE-Median) | die 500 größten pro Monat – am ehesten mit Lewellens "large stocks" vergleichbar |
| Kennzahlen | Rohwerte (teils logarithmiert) | Ränge in [−0,5; 0,5] pro Monat (wie im Tutor-Skript) |
| Rendite | Rendite | Überrendite über den risikofreien Zins (verschiebt nur das Niveau, nicht die Querschnittsunterschiede) |
| Modell 3 | 15 Variablen | 14 Variablen (eine Emissionsvariable statt zwei) |
| Zeitraum | Schätzung ab 1964, Prognosen 1974–2013 | gleich (einstellbar bis 2024) |
| Standardfehler | Newey-West, 4 Lags | gleich |

---

## Einstellungen ändern

Alle wichtigen Einstellungen stehen oben in `main.py`:

| Einstellung | Standard | Bedeutung |
| --- | --- | --- |
| `FEATURE_SET` | `"Model 3"` | Lewellens Modell 1, 2 oder 3 |
| `UNIVERSE_SIZE` | `500` | Anzahl der größten Aktien pro Monat (max. 1000) |
| `ESTIMATION` | `"fama_macbeth"` | `"fama_macbeth"` (wie Paper) oder `"pooled"` (2g) |
| `WINDOW_TYPE` | `"rolling"` | `"rolling"` (letzte 10 Jahre) oder `"expanding"` (alle Jahre seit 1964, im Paper "cumulative") |
| `TRAIN_WINDOW` | `120` | Fensterlänge in Monaten |
| `TEST_START`, `TEST_END` | 1974–2013 | Prognosezeitraum |
| `MODELS_TO_RUN` | alle vier | Liste der Modelle |
| `GRID_SEARCH` | `False` | Hyperparameter monatlich wählen (2f) |
| `LONG_PCT`, `SHORT_PCT` | `0.20`, `0.0` | Strategie aus Schritt 6 |
| `USE_DUMMY_DATA` | `False` | Zufallsdaten zum Testen |

Die Hyperparameter der Modelle stehen in `estimate()` in `main.py`.

### Ausgabedateien in `output/`

| Datei | Inhalt |
| --- | --- |
| `predictions.csv` | alle Prognosen: eine Zeile pro Monat und Aktie |
| `fama_macbeth_regressions.csv` | Tabelle 2 |
| `fama_macbeth_monthly_coefficients.csv` | die monatlichen Regressionskoeffizienten |
| `forecast_evaluation.csv` | Tabelle 3 |
| `prediction_metrics.csv`, `forecast_correlations.csv` | weitere Kennzahlen |
| `sorted_portfolios.csv` | Tabelle 6 |
| `feature_importance.csv` | Feature Importances der Bäume |
| `strategy_returns.csv`, `performance.csv`, `performance.png` | Handelsstrategie |

---

## Glossar

| Begriff | Erklärung |
| --- | --- |
| **Querschnitt** | alle Aktien zu einem Zeitpunkt (ein Monat) |
| **Fama-MacBeth** | jeden Monat eine Querschnittsregression, dann die Koeffizienten über die Zeit mitteln |
| **Out-of-sample** | die Prognose nutzt nur Daten, die zum Prognosezeitpunkt bekannt waren |
| **Look-ahead-Bias** | Fehler, bei dem versehentlich zukünftige Informationen in die Prognose einfließen |
| **Overfitting** | das Modell lernt zufällige Muster der Trainingsdaten statt echter Zusammenhänge und prognostiziert neue Daten schlecht |
| **Hyperparameter** | Einstellungen eines Modells, die nicht geschätzt, sondern vorgegeben werden (z. B. Tiefe eines Baums) |
| **Bagging** | viele Modelle auf zufälligen Teilstichproben trainieren und mitteln (Random Forest) |
| **Boosting** | Modelle nacheinander trainieren, jedes auf den Fehlern der vorigen (XGBoost) |
| **Newey-West** | Standardfehler, die Autokorrelation in einer Zeitreihe berücksichtigen |
| **Information Coefficient (IC)** | Rangkorrelation zwischen Prognose und realisierter Rendite im Querschnitt |

### Literatur

- Lewellen, J. (2015): The Cross-Section of Expected Stock Returns. *Critical Finance Review* 4, 1–44.
- Gu, S., Kelly, B., Xiu, D. (2020): Empirical Asset Pricing via Machine Learning. *Review of Financial Studies* 33, 2223–2273.
- Jensen, T. I., Kelly, B., Pedersen, L. H. (2023): Is There a Replication Crisis in Finance? *Journal of Finance* 78, 2465–2518.
