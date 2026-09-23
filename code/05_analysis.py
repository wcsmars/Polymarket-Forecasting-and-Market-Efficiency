"""Empirical analysis: calibration, favorite-longshot bias, moderators, backtest.

All inference clusters on event_id (markets within an event have mechanically
dependent outcomes). Calibration-curve CIs use a cluster bootstrap.
Writes JSON/CSV results into results/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 7, 14, 30, 60, 90]
NBINS = 20
BOOT = 1000
RNG = np.random.default_rng(42)


def load_panel():
    panel = pd.read_csv(f"{ROOT}/data/processed/panel.csv", dtype={"id": str, "event_id": str})
    panel["event_id"] = panel["event_id"].fillna("mkt_" + panel["id"])
    panel["t_res"] = pd.to_datetime(panel["t_res"], utc=True, format="mixed")
    return panel


def logit(p):
    return np.log(p / (1 - p))


# ---------- calibration curve with cluster bootstrap ----------

def calib_bins(df, nbins=NBINS, boot=BOOT):
    edges = np.linspace(0, 1, nbins + 1)
    b = np.clip(np.digitize(df["p"], edges) - 1, 0, nbins - 1)
    df = df.assign(bin=b)
    # per-event sufficient statistics: counts and y-sums per bin
    ev = df.groupby(["event_id", "bin"]).agg(n=("y", "size"), s=("y", "sum")).reset_index()
    ev_ids = ev["event_id"].unique()
    E = len(ev_ids)
    idx = {e: i for i, e in enumerate(ev_ids)}
    N_mat = np.zeros((E, nbins))
    S_mat = np.zeros((E, nbins))
    N_mat[ev["event_id"].map(idx), ev["bin"]] = ev["n"]
    S_mat[ev["event_id"].map(idx), ev["bin"]] = ev["s"]

    n_k = N_mat.sum(0)
    s_k = S_mat.sum(0)
    with np.errstate(invalid="ignore"):
        freq = s_k / n_k
    p_mean = df.groupby("bin")["p"].mean().reindex(range(nbins)).values

    # cluster bootstrap: resample events with replacement
    boots = np.full((boot, nbins), np.nan)
    for r in range(boot):
        w = RNG.multinomial(E, np.full(E, 1 / E)).astype(float)
        nn = w @ N_mat
        ss = w @ S_mat
        with np.errstate(invalid="ignore"):
            boots[r] = np.where(nn > 0, ss / nn, np.nan)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return pd.DataFrame({
        "bin": range(nbins), "n": n_k.astype(int), "p_mean": p_mean,
        "y_freq": freq, "ci_lo": lo, "ci_hi": hi,
    })


# ---------- scores ----------

def scores(df, nbins=NBINS):
    p, y = df["p"].values, df["y"].values
    bs = float(np.mean((p - y) ** 2))
    ybar = float(y.mean())
    unc = ybar * (1 - ybar)
    edges = np.linspace(0, 1, nbins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(
        n=("y", "size"), pbar=("p", "mean"), ybar=("y", "mean"))
    n = len(df)
    rel = float((g["n"] * (g["pbar"] - g["ybar"]) ** 2).sum() / n)
    res = float((g["n"] * (g["ybar"] - ybar) ** 2).sum() / n)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logscore = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return {
        "n": n, "n_events": int(df["event_id"].nunique()), "base_rate": ybar,
        "brier": bs, "brier_ref": unc, "bss": 1 - bs / unc if unc > 0 else None,
        "reliability": rel, "resolution": res, "uncertainty": unc,
        "log_score": logscore,
    }


# ---------- CORP (isotonic / PAV) calibration, Dimitriadis-Gneiting-Jordan 2021 ----------

def pav(y_sorted):
    """Pool-adjacent-violators: isotonic (non-decreasing) fit of y."""
    n = len(y_sorted)
    level_sum = list(y_sorted.astype(float))
    level_n = [1] * n
    stack_sum, stack_n = [], []
    for i in range(n):
        s, c = level_sum[i], level_n[i]
        while stack_sum and stack_sum[-1] / stack_n[-1] >= s / c:
            s += stack_sum.pop()
            c += stack_n.pop()
        stack_sum.append(s)
        stack_n.append(c)
    out = np.empty(n)
    pos = 0
    for s, c in zip(stack_sum, stack_n):
        out[pos:pos + c] = s / c
        pos += c
    return out


def corp_decomposition(df):
    """BS = MCB - DSC + UNC via isotonic recalibration (binning-free)."""
    p, y = df["p"].values, df["y"].values.astype(float)
    order = np.argsort(p, kind="mergesort")
    f = pav(y[order])
    bs_p = float(np.mean((p[order] - y[order]) ** 2))
    bs_f = float(np.mean((f - y[order]) ** 2))
    ybar = float(y.mean())
    unc = ybar * (1 - ybar)
    return {"brier": bs_p, "mcb": bs_p - bs_f, "dsc": unc - bs_f, "unc": unc}


# ---------- calibration regressions ----------

def wald(params, cov, C, c):
    d = C @ params - c
    W = float(d @ np.linalg.solve(C @ cov @ C.T, d))
    from scipy.stats import chi2
    return W, float(chi2.sf(W, len(c)))


def mz_regressions(df):
    groups = df["event_id"]
    X = sm.add_constant(df["p"].values)
    ols = sm.OLS(df["y"].values, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    W_lin, p_lin = wald(ols.params, ols.cov_params(), np.eye(2), np.array([0.0, 1.0]))

    Xl = sm.add_constant(logit(df["p"].values))
    glm = sm.GLM(df["y"].values, Xl, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    W_log, p_log = wald(glm.params, glm.cov_params(), np.eye(2), np.array([0.0, 1.0]))

    return {
        "linear": {"alpha": ols.params[0], "beta": ols.params[1],
                   "se_alpha": ols.bse[0], "se_beta": ols.bse[1],
                   "wald_chi2": W_lin, "wald_p": p_lin},
        "logodds": {"a": glm.params[0], "b": glm.params[1],
                    "se_a": glm.bse[0], "se_b": glm.bse[1],
                    "wald_chi2": W_log, "wald_p": p_log,
                    "z_b_vs_1": (glm.params[1] - 1) / glm.bse[1]},
        "n": len(df), "n_events": int(df["event_id"].nunique()),
    }


# ---------- bucket means with clustered SE ----------

def cluster_mean_se(df, col):
    """Mean of col and SE clustered by event (cluster-sum formula)."""
    g = df.groupby("event_id")[col].agg(["sum", "size"])
    n = g["size"].sum()
    mean = g["sum"].sum() / n
    resid_sums = g["sum"] - mean * g["size"]
    se = np.sqrt((resid_sums ** 2).sum()) / n
    return float(mean), float(se), int(n), len(g)


def flb_buckets(df):
    out = {}
    for name, lo, hi in [("longshot_0_10", 0.0, 0.10), ("mid_10_90", 0.10, 0.90),
                         ("favorite_90_100", 0.90, 1.0)]:
        sub = df[(df["p"] >= lo) & (df["p"] < hi)].copy()
        if len(sub) == 0:
            continue
        sub["d"] = sub["y"] - sub["p"]
        mean_d, se_d, n, n_ev = cluster_mean_se(sub, "d")
        out[name] = {"n": n, "n_events": n_ev, "mean_p": float(sub["p"].mean()),
                     "mean_y": float(sub["y"].mean()), "y_minus_p": mean_d,
                     "se": se_d, "t": mean_d / se_d if se_d > 0 else None}
    return out


# ---------- strategy backtest ----------

def backtest(df, cost):
    out = {}
    fav = df[(df["p"] >= 0.90) & (df["p"] <= 0.99)].copy()
    fav["ret"] = fav["y"] / (fav["p"] + cost) - 1
    lng = df[(df["p"] >= 0.01) & (df["p"] <= 0.10)].copy()
    lng["ret"] = (1 - lng["y"]) / ((1 - lng["p"]) + cost) - 1
    for name, sub in [("buy_favorites_yes", fav), ("fade_longshots_buy_no", lng)]:
        if len(sub) == 0:
            continue
        m, se, n, n_ev = cluster_mean_se(sub, "ret")
        monthly = sub.set_index("t_res").groupby(pd.Grouper(freq="ME"))["ret"].mean().dropna()
        out[name] = {
            "n_trades": n, "n_events": n_ev, "mean_ret": m, "se_clust": se,
            "t": m / se if se > 0 else None, "win_rate": float((sub["ret"] > 0).mean()),
            "monthly_mean": float(monthly.mean()), "monthly_std": float(monthly.std()),
            "n_months": int(len(monthly)),
        }
    # decile return table (buy YES at p, gross)
    dec = df.copy()
    dec["decile"] = np.clip((dec["p"] * 10).astype(int), 0, 9)
    dec["ret"] = dec["y"] / dec["p"] - 1
    tab = []
    for d in range(10):
        sub = dec[dec["decile"] == d]
        if len(sub) < 30:
            tab.append(None)
            continue
        m, se, n, n_ev = cluster_mean_se(sub, "ret")
        tab.append({"decile": d, "n": n, "mean_p": float(sub["p"].mean()),
                    "mean_ret": m, "se": se})
    out["decile_buy_yes_gross"] = tab
    return out


def main():
    panel = load_panel()
    results = {"horizon": {}}

    for h in HORIZONS:
        df = panel[panel["h"] == h]
        if len(df) < 200:
            continue
        bins = calib_bins(df)
        bins.to_csv(f"{ROOT}/results/calibration_bins_h{h}.csv", index=False)
        results["horizon"][h] = {
            "scores": scores(df),
            "corp": corp_decomposition(df),
            "regressions": mz_regressions(df),
            "flb": flb_buckets(df),
        }
        print(f"h={h}: n={len(df)}, brier={results['horizon'][h]['scores']['brier']:.4f}, "
              f"logodds_b={results['horizon'][h]['regressions']['logodds']['b']:.3f}", flush=True)

    # ---- moderators at h=7 ----
    df7 = panel[panel["h"] == 7].copy()
    mods = {}
    df7["logv"] = np.log10(df7["volumeNum"].clip(lower=1))
    df7["vol_grp"] = pd.qcut(df7["logv"], 3, labels=["low", "mid", "high"])
    for grp_col, name in [("vol_grp", "volume_tercile"), ("cat", "category"), ("era", "era")]:
        mods[name] = {}
        for g, sub in df7.groupby(grp_col, observed=True):
            if len(sub) < 500 or sub["y"].nunique() < 2:
                continue
            reg = mz_regressions(sub)
            sc = scores(sub)
            mods[name][str(g)] = {
                "n": len(sub), "b": reg["logodds"]["b"], "se_b": reg["logodds"]["se_b"],
                "alpha": reg["logodds"]["a"], "brier": sc["brier"], "bss": sc["bss"],
                "reliability": sc["reliability"], "base_rate": sc["base_rate"],
                "median_volume": float(sub["volumeNum"].median()),
            }
    results["moderators_h7"] = mods

    # volume x extremeness interaction: does FLB shrink with volume?
    mods_flb = {}
    for g, sub in df7.groupby("vol_grp", observed=True):
        mods_flb[str(g)] = flb_buckets(sub)
    results["flb_by_volume_h7"] = mods_flb

    # ---- backtest at h=7 and h=30 ----
    results["backtest"] = {}
    for h in [7, 30]:
        df = panel[panel["h"] == h]
        results["backtest"][f"h{h}"] = {
            "gross": backtest(df, 0.0),
            "net_1c": backtest(df, 0.01),
        }

    # ---- robustness at h=7 ----
    rob = {}
    dedup = df7.sort_values("volumeNum", ascending=False).drop_duplicates("event_id")
    rob["one_per_event"] = {**mz_regressions(dedup)["logodds"], "n": len(dedup),
                            "brier": scores(dedup)["brier"]}
    nocd = df7[df7["cat"] != "crypto_daily"]
    rob["excl_crypto_daily"] = {**mz_regressions(nocd)["logodds"], "n": len(nocd)}
    nonr = df7[~(df7["event_neg_risk"].astype(bool) | df7["negRisk"].astype("boolean").fillna(False).astype(bool))]
    rob["excl_negrisk"] = {**mz_regressions(nonr)["logodds"], "n": len(nonr)}
    bigv = df7[df7["volumeNum"] >= 10000]
    rob["volume_ge_10k"] = {**mz_regressions(bigv)["logodds"], "n": len(bigv)}

    # Schedule-anchored prices: available quotes h days before scheduled end.
    # These retrospectively collected anchors can fall after the closure-time
    # proxy; see README limitations before interpreting them as live forecasts.
    mkt = pd.read_csv(f"{ROOT}/data/processed/market_level.csv",
                      dtype={"id": str, "event_id": str})
    mkt["event_id"] = mkt["event_id"].fillna("mkt_" + mkt["id"])
    mkt["t_res"] = pd.to_datetime(mkt["t_res"], utc=True, format="mixed")
    sched_res = {}
    for h in HORIZONS:
        sub = mkt[mkt[f"p_sched_{h}"].notna()].copy()
        sub["p"] = sub[f"p_sched_{h}"]
        if len(sub) < 500:
            continue
        sched_res[f"h{h}"] = {
            "logodds": mz_regressions(sub)["logodds"], "n": len(sub),
            "scores": scores(sub), "flb": flb_buckets(sub),
        }
    results["schedule_anchored"] = sched_res
    # Retrospective schedule-anchored backtest at h=30.
    sub30 = mkt[mkt["p_sched_30"].notna()].copy()
    sub30["p"] = sub30["p_sched_30"]
    results["backtest_sched_h30"] = {
        "gross": backtest(sub30, 0.0), "net_1c": backtest(sub30, 0.01)}

    midl = mkt[mkt["p_mid"].notna()].copy()
    midl["p"] = midl["p_mid"]
    rob["midlife_price"] = {**mz_regressions(midl)["logodds"], "n": len(midl)}
    rob["sched_anchored_h7"] = {**sched_res["h7"]["logodds"], "n": sched_res["h7"]["n"]} \
        if "h7" in sched_res else None
    results["robustness"] = rob

    # ---- balanced panel: same markets observed at h = 1, 7, 30 ----
    wide = panel[panel["h"].isin([1, 7, 30])].pivot_table(
        index="id", columns="h", values="p", aggfunc="first")
    ids_bal = wide.dropna().index
    bal = {}
    for h in [1, 7, 30]:
        sub = panel[(panel["h"] == h) & (panel["id"].isin(ids_bal))]
        bal[f"h{h}"] = {"scores": scores(sub),
                        "logodds": mz_regressions(sub)["logodds"],
                        "flb": flb_buckets(sub)}
    results["balanced_panel_h1_7_30"] = {"n_markets": int(len(ids_bal)), **bal}

    with open(f"{ROOT}/results/analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("DONE -> results/analysis.json", flush=True)


if __name__ == "__main__":
    main()
