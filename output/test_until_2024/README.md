# Lauf mit Prognosen bis 2024

Dieselben Einstellungen wie der Hauptlauf in `output/`, nur mit Prognosen von
Januar 1974 bis November 2024 statt bis Dezember 2013. Die Prognosen für
1974–2013 sind identisch mit dem Hauptlauf.

Erzeugt mit diesen zwei Änderungen oben in `main.py`:

```python
TEST_END = None
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "test_until_2024")
```

Auswertung des Zeitraums 2014–2024: siehe Abschnitt "Erweiterung: Prognosen
bis 2024" in der README im Hauptordner.
