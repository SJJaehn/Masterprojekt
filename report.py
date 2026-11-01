"""
Ausgabe: Tabellen in der Konsole und die Grafik der Handelsstrategie.
Hier wird nichts gerechnet, nur dargestellt.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Zahlenformate einzelner Spalten (alle anderen Spalten mit 2 Nachkommastellen)
EVALUATION_FORMATS = {"R2": "{:.3f}"}
PERFORMANCE_FORMATS = {
    "Ann. Return": "{:.2%}",
    "Ann. Volatility": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Max Drawdown": "{:.2%}",
}


def fmt_date(date):
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def format_table(table, formats=None, decimals=2):
    """Formatiert jede Spalte von `table` mit ihrem Format aus `formats` (sonst `decimals` Nachkommastellen)."""
    formats = formats or {}
    return table.apply(lambda col: col.map(formats.get(col.name, f"{{:.{decimals}f}}").format))


def print_section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_table(table, formats=None, decimals=2):
    """Gibt eine Tabelle formatiert aus (siehe `format_table`)."""
    print(format_table(table, formats, decimals).to_string())


# =============================================================================
# Grafik
# =============================================================================

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
    Performance-Tabelle aus `evaluation.performance_metrics` (eine Zeile pro Strategie).
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
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
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
            cell.get_text().set_fontweight("bold")
        if col == 0:
            cell.get_text().set_horizontalalignment("left")

    # 8. Speichern / anzeigen
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.04)
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    plt.close(fig)


def plot_time_variation(panels, window, title="", subtitle="", save_path=None, show=True):
    """Gleitender Mittelwert monatlicher Kennzahlen über die Zeit, ein Teildiagramm pro Kennzahl.

    `panels` ordnet jeder Überschrift eine Tabelle zu (Monate x Modelle, Werte pro
    Monat); gezeichnet wird der Mittelwert über die letzten `window` Monate.
    Für die Prognose-Steigung zeigt eine gestrichelte Linie den Idealwert 1.
    """
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (panel_title, monthly) in zip(axes, panels.items()):
        rolling = monthly.rolling(window).mean().dropna(how="all")
        ax.set_facecolor(SURFACE)
        for i, name in enumerate(rolling.columns):
            ax.plot(rolling.index, rolling[name], color=SERIES_COLORS[i % len(SERIES_COLORS)], lw=1.6,
                    label=name, zorder=3)
        ax.axhline(0.0, color=AXIS, lw=1, zorder=1)
        if "slope" in panel_title.lower():
            ax.axhline(1.0, color=TEXT_MUTED, lw=1, ls="--", zorder=1)
        ax.set_title(panel_title, loc="left", fontsize=10, color=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0, pad=6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.072, 0.94), ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    if title:
        fig.text(0.08, 0.99, title, va="top", fontsize=13, fontweight="bold", color=TEXT_PRIMARY)
    if subtitle:
        fig.text(0.08, 0.962, subtitle, va="top", fontsize=10, color=TEXT_SECONDARY)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.05, hspace=0.35)
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    plt.close(fig)
