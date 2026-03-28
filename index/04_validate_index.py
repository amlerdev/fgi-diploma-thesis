"""
validate_index.py
=================
Validace dvou variant indexu proti CNN Fear & Greed (2011–2026).

Metriky: Pearson r, R², MAE, RMSE, Bias, P90 absolutní chyby
Výstup:  validation_chart.png (do code/index/)

Vstup:  code/data/fg_index_final.csv

Author: Petr Amler (AML0005)
"""

import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

INDEX_DIR = Path(__file__).resolve().parent
INPUT     = INDEX_DIR / 'fg_index_final.csv'

VARIANTS = [
    ('FG_Equal', '#2196F3', 'Equal (1/7)'),
    ('FG_OLS',   '#FF9800', 'OLS'),
]

COMP_COLS = ['Mom_Norm', 'Strength_Norm', 'Breadth_Norm',
             'PutCall_Norm', 'VIX_Norm', 'SafeHaven_Norm', 'JunkBond_Norm']

print("=" * 65)
print("VALIDACE INDEXU vs CNN FEAR & GREED")
print("=" * 65)

# ── Načtení a příprava ────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, parse_dates=['Date'], index_col='Date')

def get_overlap(col):
    mask = df['CNN_FearGreed'].notna() & df[col].notna()
    return df[mask][['CNN_FearGreed', col]].copy()

# ── Statistiky ────────────────────────────────────────────────────────────────
def compute_stats(actual, predicted):
    err     = predicted - actual
    abs_err = err.abs()
    r, _    = pearsonr(actual, predicted)
    return {
        'r':    r,
        'r2':   r ** 2,
        'mae':  abs_err.mean(),
        'rmse': np.sqrt((err ** 2).mean()),
        'bias': err.mean(),
        'p90':  abs_err.quantile(0.90),
    }

results = {}
for col, color, label in VARIANTS:
    ov = get_overlap(col)
    results[col] = compute_stats(ov['CNN_FearGreed'], ov[col])
    results[col]['n']     = len(ov)
    results[col]['label'] = label
    results[col]['color'] = color

# ── Tisk tabulky ─────────────────────────────────────────────────────────────
print(f"\n{'Metrika':<18} {'Equal (1/7)':>14} {'OLS':>14}")
print(f"{'-'*48}")
for key, label in [('r','Pearson r'), ('r2','R²'), ('mae','MAE'),
                   ('rmse','RMSE'), ('bias','Bias'), ('p90','P90 |chyba|'), ('n','n (dní)')]:
    row = f"   {label:<16}"
    for col, _, _ in VARIANTS:
        val = results[col][key]
        if key == 'n':
            row += f" {int(val):>14,}"
        elif key == 'bias':
            row += f" {val:>+14.3f}"
        elif key in ('r', 'r2'):
            row += f" {val:>14.4f}"
        else:
            row += f" {val:>14.3f}"
    print(row)

# ── Bias podle CNN pásem ──────────────────────────────────────────────────────
bins        = [0, 25, 45, 55, 75, 100]
zone_labels = ['Extreme Fear (0–25)', 'Fear (25–45)', 'Neutral (45–55)',
               'Greed (55–75)', 'Extreme Greed (75–100)']

print(f"\nBias podle CNN pásem:")
print(f"   {'Pásmo':<24} {'Equal':>10} {'OLS':>10}")
print(f"   {'-'*46}")
for zone in zone_labels:
    row = f"   {zone:<24}"
    for col, _, _ in VARIANTS:
        ov = get_overlap(col).copy()
        ov['zone'] = pd.cut(ov['CNN_FearGreed'], bins=bins, labels=zone_labels, include_lowest=True)
        sub  = ov[ov['zone'] == zone]
        bias = (sub[col] - sub['CNN_FearGreed']).mean() if len(sub) > 0 else float('nan')
        row += f" {bias:>+10.2f}"
    print(row)

# ── Příprava bias per pásmo pro grafy ────────────────────────────────────────
zone_labels_short = ['Extreme Fear\n(0–25)', 'Fear\n(25–45)', 'Neutral\n(45–55)',
                     'Greed\n(55–75)', 'Extreme Greed\n(75–100)']
zone_colors       = ['#F44336', '#FF9800', '#9E9E9E', '#8BC34A', '#4CAF50']

def zone_biases(col):
    ov = get_overlap(col).copy()
    ov['zone'] = pd.cut(ov['CNN_FearGreed'], bins=bins,
                        labels=zone_labels_short, include_lowest=True)
    biases = []
    for z in zone_labels_short:
        sub = ov[ov['zone'] == z]
        biases.append((sub[col] - sub['CNN_FearGreed']).mean() if len(sub) > 0 else 0.0)
    return biases

# ── Vizualizace ───────────────────────────────────────────────────────────────
ov_base = get_overlap('FG_Equal')

fig = plt.figure(figsize=(14, 18))
fig.suptitle(
    "Validace rekonstrukce Fear & Greed Indexu (2011–2026)\n"
    f"Equal r={results['FG_Equal']['r']:.3f}  |  "
    f"OLS r={results['FG_OLS']['r']:.3f}",
    fontsize=13, fontweight='bold', y=0.99
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# Řada 1: Časová řada (plná šířka)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(ov_base.index, ov_base['CNN_FearGreed'],
         color='black', linewidth=1.2, alpha=0.9, label='CNN Fear & Greed (skutečný)', zorder=5)
for col, color, label in VARIANTS:
    ov = get_overlap(col)
    ax1.plot(ov.index, ov[col], color=color, linewidth=0.9, alpha=0.75,
             label=f"{label} (r={results[col]['r']:.3f})")
ax1.axhline(25, color='#F44336', linestyle='--', linewidth=0.7, alpha=0.4)
ax1.axhline(75, color='#4CAF50', linestyle='--', linewidth=0.7, alpha=0.4)
ax1.set_ylabel('Fear & Greed (0–100)')
ax1.set_title('Časová řada: CNN vs. rekonstrukce')
ax1.legend(fontsize=9, loc='upper left')
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Řada 2: Scatter plot
for i, (col, color, label) in enumerate(VARIANTS):
    ax = fig.add_subplot(gs[1, i])
    ov = get_overlap(col)
    ax.scatter(ov['CNN_FearGreed'], ov[col], alpha=0.12, s=3, color=color, rasterized=True)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.0, label='Ideální shoda')
    z      = np.polyfit(ov['CNN_FearGreed'], ov[col], 1)
    x_line = np.linspace(0, 100, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color=color, linewidth=1.5,
            label=f"r={results[col]['r']:.3f}")
    ax.set_xlabel('CNN Fear & Greed')
    ax.set_ylabel(col)
    ax.set_title(f'Scatter: {label}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

# Řada 3: Bias podle CNN pásem
for i, (col, color, label) in enumerate(VARIANTS):
    ax     = fig.add_subplot(gs[2, i])
    biases = zone_biases(col)
    bars   = ax.bar(zone_labels_short, biases, color=zone_colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, biases):
        ypos = bar.get_height() + 0.15 if val >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{val:+.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_ylabel('Průměrný bias (body)')
    ax.set_title(f'Bias podle CNN pásma: {label}')
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='y', alpha=0.3)

# Řada 4: Souhrnná tabulka metrik
for i, (col, color, label) in enumerate(VARIANTS):
    ax = fig.add_subplot(gs[3, i])
    ax.axis('off')
    s  = results[col]
    ov = get_overlap(col)
    table_data = [
        ['Metrika',       'Hodnota'],
        ['Pearson r',     f"{s['r']:.4f}"],
        ['R²',            f"{s['r2']:.4f}"],
        ['MAE (body)',    f"{s['mae']:.2f}"],
        ['RMSE (body)',   f"{s['rmse']:.2f}"],
        ['Bias',          f"{s['bias']:+.2f}"],
        ['P90 |chyba|',   f"{s['p90']:.1f}"],
        ['Počet dní',     f"{s['n']:,}"],
        ['Období',        f"{ov.index[0].date()} → {ov.index[-1].date()}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    tbl[0, 0].set_facecolor('#37474F'); tbl[0, 0].set_text_props(color='white', fontweight='bold')
    tbl[0, 1].set_facecolor('#37474F'); tbl[0, 1].set_text_props(color='white', fontweight='bold')
    ax.set_title(f'Souhrnné statistiky: {label}', fontsize=10, fontweight='bold', pad=40)

out_chart = INDEX_DIR / 'validation_chart.png'
plt.savefig(out_chart, dpi=150, bbox_inches='tight')
print(f"\nGraf uložen: {out_chart}")
print("\nHotovo.")
