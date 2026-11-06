# Paper

Die Paper, auf denen das Projekt aufbaut. Die Verlagsfassungen liegen aus
Urheberrechtsgründen nicht im Repository; Lewellen liegt als frei verfügbare
SSRN-Fassung im Hauptordner (`ssrn-2511246.pdf`).

| Paper | Quelle | Rolle im Projekt |
| --- | --- | --- |
| Lewellen, J. (2015): The Cross Section of Expected Stock Returns. *Critical Finance Review* 4(1), 1–44. | [SSRN 2511246](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2511246) | Replikation: Fama-MacBeth-Prognose (`"single"`), Tabellen 2, 3, 6, Fig. 2 |
| Gu, S., Kelly, B., Xiu, D. (2020): Empirical Asset Pricing via Machine Learning. *Review of Financial Studies* 33(5), 2223–2274. | [doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009), Internet Appendix als "Supplementary data" auf der Verlagsseite | gestapelte Schätzung (`"stacked"`), Bäume, R² OOS |
| Bali, T. G., Kelly, B. T., Mörke, M., Rahman, J. (2026): Machine Forecast Disagreement. *Review of Financial Studies*, Advance Article. | [doi:10.1093/rfs/hhag042](https://doi.org/10.1093/rfs/hhag042) | Investors und MFD (Schritt 8) |

## Stellen, auf die sich das Projekt stützt

**Gu, Kelly & Xiu (2020)**
- Abschnitt 1, Gl. (1)–(2): ein Modell g(·) für alle Aktien und Monate, "depends neither on i nor t"; die Prognose nutzt nur die Kennzahlen z_i,t, keine Historie vor t.
- Gl. (4): Verlust als Summe über alle Aktien i und Monate t ("pooled OLS estimator") – die Daten sind ein gestapeltes Panel mit einer Zeile je Aktie und Monat.
- Abschnitt 2.1: 94 Kennzahlen, je Monat in Ränge auf [−1, 1], fehlende Werte durch den Median des Monats ersetzt; 8 Makrovariablen nach Welch & Goyal (2008); Eingaben z_i,t = x_t ⊗ c_i,t plus 74 Branchen-Dummies, zusammen 920.
- Abschnitt 2.1: Training 1957–1974 (wächst jedes Jahr um ein Jahr), Validierung 12 Jahre (rollt mit), Test 1987–2016; neu trainiert einmal im Jahr.
- Tabelle 7: Dezilportfolios, wertgewichtet.
- Internet Appendix, Tabelle A.5: Hyperparameter-Raster (Random Forest Tiefe 1–6, 300 Bäume; GBRT Tiefe 1–2, bis 1000 Bäume, Lernrate 0,01 oder 0,1, Huber-Verlust); Tabellen A.9 und A.10: gleichgewichtete Dezilportfolios, A.10 ohne Microcaps.

**Bali, Kelly, Mörke & Rahman (2026)**
- Abschnitt 1: MFD = Standardabweichung der Prognosen von K Investoren (Gl. 4).
- Abschnitt 2 und Anhang B, Tabelle B1: 100 Investoren mit je 50 % der 153 Kennzahlen; je Investor ein Random Forest (2 000 Bäume, Tiefe 3, je 5 % der Beobachtungen und Kennzahlen); 10 Jahre rollierend, neu trainiert alle 12 Monate; Renditen im Training je 1 % winsorisiert.
