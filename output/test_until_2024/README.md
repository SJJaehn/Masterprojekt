# Lauf mit Prognosen bis 2024

Dieselben Einstellungen wie der Hauptlauf in `output/`, nur mit
`TEST_END = None` in `main.py`: Prognosen von Januar 1974 bis November 2024
statt bis Dezember 2013. Die Prognosen für 1974–2013 sind identisch mit dem
Hauptlauf.

Erzeugt mit:

```python
import main
main.TEST_END = None
main.OUTPUT_DIR = "output/test_until_2024"
main.main()
```

Auswertung des Zeitraums 2014–2024: siehe Abschnitt "Erweiterung: Prognosen
bis 2024" in der README im Hauptordner.
