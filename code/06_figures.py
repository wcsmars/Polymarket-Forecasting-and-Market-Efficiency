"""Generate calibration-study figures from results/ and the panel."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = f"{ROOT}/figures/calibration"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
BLUE, RED, GRAY = "#2563eb", "#dc2626", "#6b7280"

results = json.load(open(f"{ROOT}/results/analysis.json"))
panel = pd.read_csv(f"{ROOT}/data/processed/panel.csv", dtype={"id": str, "event_id": str})
panel["t_res"] = pd.to_datetime(panel["t_res"], utc=True, format="mixed")


# ---- Figure 1: calibration curves at four horizons ----
fig, axes = plt.subplots(2, 2, figsize=(8, 7.2), sharex=True, sharey=True)
for ax, h in zip(axes.flat, [1, 7, 30, 90]):
    bins = pd.read_csv(f"{ROOT}/results/calibration_bins_h{h}.csv")
    bins = bins[bins["n"] >= 20]
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", zorder=1)
    ax.errorbar(bins["p_mean"], bins["y_freq"],
                yerr=[bins["y_freq"] - bins["ci_lo"], bins["ci_hi"] - bins["y_freq"]],
                fmt="o", ms=3.5, color=BLUE, ecolor=BLUE, elinewidth=0.9,
                capsize=2, zorder=3)
    sc = results["horizon"][str(h)]["scores"]
    reg = results["horizon"][str(h)]["regressions"]["logodds"]
    ax.set_title(f"h = {h} day{'s' if h > 1 else ''} before resolution", fontsize=9.5)
    ax.text(0.03, 0.92, f"N = {sc['n']:,}\nBrier = {sc['brier']:.4f}\n"
            f"b = {reg['b']:.3f} ({reg['se_b']:.3f})",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(fc="white", ec="#d1d5db", lw=0.6, boxstyle="round,pad=0.3"))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
for ax in axes[1]:
    ax.set_xlabel("Market price (implied probability)")
for ax in axes[:, 0]:
    ax.set_ylabel("Empirical YES frequency")
fig.suptitle("Calibration of Polymarket prices by forecast horizon", y=0.995, fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_calibration_by_horizon.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: local miscalibration (y - p) at h=7, zoomed ----
bins7 = pd.read_csv(f"{ROOT}/results/calibration_bins_h7.csv")
bins7 = bins7[bins7["n"] >= 20]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.axhline(0, color=GRAY, lw=1, ls="--")
d = bins7["y_freq"] - bins7["p_mean"]
ax.errorbar(bins7["p_mean"], d,
            yerr=[d - (bins7["ci_lo"] - bins7["p_mean"]),
                  (bins7["ci_hi"] - bins7["p_mean"]) - d],
            fmt="o", ms=4, color=RED, ecolor=RED, elinewidth=1, capsize=2)
ax.set_xlabel("Market price (implied probability), h = 7 days")
ax.set_ylabel("Empirical frequency − price")
ax.set_title("Local miscalibration with 95% cluster-bootstrap bands (h = 7)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_miscalibration_h7.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: log-odds slope b by horizon ----
hs = sorted(int(k) for k in results["horizon"])
bs = [results["horizon"][str(h)]["regressions"]["logodds"]["b"] for h in hs]
ses = [results["horizon"][str(h)]["regressions"]["logodds"]["se_b"] for h in hs]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.axhline(1, color=GRAY, lw=1, ls="--", label="perfect calibration (b = 1)")
ax.errorbar(hs, bs, yerr=1.96 * np.array(ses), fmt="o-", color=BLUE,
            ms=4, lw=1.2, capsize=3)
ax.set_xscale("log")
ax.set_xticks(hs); ax.set_xticklabels(hs)
ax.set_xlabel("Horizon (days before resolution, log scale)")
ax.set_ylabel("Log-odds calibration slope b")
ax.set_title("Calibration slope by horizon (b > 1 ⇒ favorite-longshot bias)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_slope_by_horizon.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: decile returns of buying YES, h=7 ----
dec = results["backtest"]["h7"]["gross"]["decile_buy_yes_gross"]
dec = [d for d in dec if d]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = [d["mean_p"] for d in dec]
y = [100 * d["mean_ret"] for d in dec]
err = [196 * d["se"] for d in dec]
ax.axhline(0, color=GRAY, lw=1, ls="--")
ax.bar(range(len(dec)), y, color=[BLUE if v > 0 else RED for v in y], alpha=0.85)
ax.errorbar(range(len(dec)), y, yerr=err, fmt="none", ecolor="black",
            elinewidth=0.9, capsize=3)
ax.set_xticks(range(len(dec)))
ax.set_xticklabels([f"{d['mean_p']:.2f}" for d in dec], fontsize=7.5)
ax.set_xlabel("Mean price within equal-width price bin (h = 7 days)")
ax.set_ylabel("Mean return of buying YES (%)")
ax.set_title("Gross return to a $1 YES bet by price bin, held to resolution")
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_decile_returns.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 5: sample composition ----
df7 = panel[panel["h"] == 7]
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
cat_counts = df7["cat"].value_counts()
axes[0].barh(cat_counts.index[::-1], cat_counts.values[::-1], color=BLUE, alpha=0.85)
axes[0].set_title("Markets by category (h = 7 sample)")
axes[0].set_xlabel("Markets")
monthly = panel[panel["h"] == 1].set_index("t_res").groupby(pd.Grouper(freq="ME"))["id"].count()
axes[1].plot(monthly.index, monthly.values, color=BLUE, lw=1.4)
# Few, short date labels: full YYYY-MM ticks overlap at this panel width.
locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
axes[1].xaxis.set_major_locator(locator)
axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
axes[1].set_title("Resolved markets per month (h = 1 sample)")
axes[1].set_ylabel("Markets")
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_sample_composition.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 6: calibration slope by volume tercile and category, h=7 ----
mods = results["moderators_h7"]
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
vt = mods["volume_tercile"]
keys = [k for k in ["low", "mid", "high"] if k in vt]
axes[0].axhline(1, color=GRAY, lw=1, ls="--")
axes[0].errorbar(range(len(keys)), [vt[k]["b"] for k in keys],
                 yerr=[1.96 * vt[k]["se_b"] for k in keys],
                 fmt="o", color=BLUE, ms=5, capsize=3)
axes[0].set_xticks(range(len(keys)))
axes[0].set_xticklabels([f"{k}\n(med ${vt[k]['median_volume']:,.0f})" for k in keys], fontsize=8)
axes[0].set_ylabel("Log-odds slope b")
axes[0].set_title("By lifetime-volume tercile (h = 7)")
ct = mods["category"]
ck = sorted(ct, key=lambda k: -ct[k]["n"])
axes[1].axhline(1, color=GRAY, lw=1, ls="--")
axes[1].errorbar(range(len(ck)), [ct[k]["b"] for k in ck],
                 yerr=[1.96 * ct[k]["se_b"] for k in ck],
                 fmt="o", color=BLUE, ms=5, capsize=3)
axes[1].set_xticks(range(len(ck)))
axes[1].set_xticklabels(ck, rotation=30, ha="right", fontsize=8)
axes[1].set_title("By category (h = 7)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig6_moderators.png", bbox_inches="tight")
plt.close(fig)

print("figures written to", FIG)
