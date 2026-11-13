"""
Schritt 9 (Transaktionskosten) für Läufe, die schon in output/results/ liegen.

Liest die gespeicherten Prognosen (predictions.csv.gz) jedes Laufs und rechnet die Strategien
aus Schritt 6 und 8 nach Transaktionskosten nach - genau wie Schritt 9 in main.py, ohne neu zu
schätzen. Neue Läufe mit main.py brauchen dieses Skript nicht.

    python transaction_costs.py                        alle Läufe
    python transaction_costs.py fixed_leaf25/return    nur Läufe unter output/results/fixed_leaf25/return
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import util
from util import DATE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
SIGNALS = ["OLS", "DecisionTree", "RandomForest", "XGBoost", "Investors"]
STRATEGIES = {"Long": (0.10, 0.0), "Short": (0.0, 0.10), "Long-Short": (0.10, 0.10)}  # wie in main.py

pattern = os.path.join(RESULTS_DIR, sys.argv[1] if len(sys.argv) > 1 else "", "**", "predictions.csv.gz")
for path in sorted(glob.glob(pattern, recursive=True)):
    run_dir = os.path.dirname(path)
    setup, target, years, universe, estimation = os.path.relpath(run_dir, RESULTS_DIR).split(os.sep)
    preds = pd.read_csv(path, parse_dates=[DATE])
    universe_label = "top 500 stocks" if universe == "top500" else f"{universe} stocks"
    period = (f"{years.replace('-', ' - ')}, {universe_label}, {estimation}, rolling"
              + (", trees on return ranks" if target == "rank" else ""))
    results = util.evaluate_costs(preds, [s for s in SIGNALS if s in preds], STRATEGIES,
                                  os.path.join(run_dir, "predicted_return"),
                                  os.path.join(run_dir, "investor_disagreement"), period)
    table = results[os.path.join(run_dir, "predicted_return")][1]
    print(f"{os.path.relpath(run_dir, RESULTS_DIR)}: Long-Short net % p.m. "
          + ", ".join(f"{s} {table.loc[('Long-Short', s), 'Net return (% p.m.)']:.2f}"
                      for s in SIGNALS if ("Long-Short", s) in table.index))
