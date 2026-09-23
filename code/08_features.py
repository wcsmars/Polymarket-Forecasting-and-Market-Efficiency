"""Model study: build the ML dataset (one row per market x horizon x anchoring).

Feature timing rules:
  - Snapshot tau = (scheduled end - h days) for anchor='sched' (primary) or
    (closure-time proxy - h days) for anchor='res' (robustness).
  - Standing price at tau: latest observation <= tau, staleness <= 1.5 days.
  - Path features use ONLY observations with t <= tau.
  - No lifetime volume / current liquidity (post-snapshot info).
The closure-time proxy t_res uses closedTime, falling back to endDate.
Scheduled snapshots are not filtered to precede this proxy. Metadata is
collected retrospectively; see the README timing limitations.
Output: data/processed/ml_dataset.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 7, 14, 30, 60, 90]
STALE_TOL = 1.5 * 86400
DAY = 86400


def path_features(ts, ps, tau):
    """Features from observations strictly at/before tau."""
    idx = np.searchsorted(ts, tau, side="right")
    if idx < 2:
        return None
    t, p = ts[:idx], ps[:idx]
    if tau - t[-1] > STALE_TOL:
        return None
    dp = np.diff(p)
    last7 = p[t >= tau - 7 * DAY]
    dp7 = np.diff(last7) if len(last7) > 1 else np.array([0.0])
    p_now = p[-1]

    def pk(days):
        """price 'days' before tau (standing), nan if unavailable"""
        j = np.searchsorted(t, tau - days * DAY, side="right") - 1
        return p[j] if j >= 0 else np.nan

    p7, p30 = pk(7), pk(30)
    return {
        "p": float(np.clip(p_now, 0.001, 0.999)),
        "n_obs_pre": idx,
        "days_live": (tau - t[0]) / DAY,
        "p_start": p[0],
        "p_mean_pre": float(p.mean()),
        "p_max_pre": float(p.max()),
        "p_min_pre": float(p.min()),
        "p_range_pre": float(p.max() - p.min()),
        "chg_7": float(p_now - p7) if np.isfinite(p7) else 0.0,
        "chg_30": float(p_now - p30) if np.isfinite(p30) else 0.0,
        "vol_all": float(dp.std()) if len(dp) > 1 else 0.0,
        "vol_7": float(dp7.std()) if len(dp7) > 1 else 0.0,
        "absmove_7": float(np.abs(dp7).mean()) if len(dp7) else 0.0,
        "active_frac": float((np.abs(dp) > 1e-4).mean()) if len(dp) else 0.0,
        "dist_to_half": abs(p_now - 0.5),
    }


def main():
    sample = pd.read_csv(f"{ROOT}/data/processed/sample_markets.csv",
                         dtype={"id": str, "yes_token": str, "event_id": str})
    for c in ["t_res", "t_created", "t_end"]:
        sample[c] = pd.to_datetime(sample[c], utc=True, format="mixed")
    sample["event_id"] = sample["event_id"].fillna("mkt_" + sample["id"])
    ev_size = sample.groupby("event_id")["id"].transform("count")
    sample["event_n_markets"] = ev_size

    hist = {}
    with open(f"{ROOT}/data/raw/price_histories.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["n"] > 1:
                a = np.array(rec["history"], dtype=float)
                hist[rec["id"]] = (a[:, 0], a[:, 1])

    rows = []
    for r in sample.itertuples(index=False):
        h_arr = hist.get(r.id)
        if h_arr is None:
            continue
        ts, ps = h_arr
        t_res = r.t_res.timestamp()
        t_end = r.t_end.timestamp() if pd.notna(r.t_end) else np.nan
        t_created = r.t_created.timestamp() if pd.notna(r.t_created) else ts[0]
        q = (r.question or "") if isinstance(r.question, str) else ""
        ql = q.lower()
        static = {
            "id": r.id, "event_id": r.event_id, "y": int(r.y), "cat": r.cat,
            "neg_risk": bool(r.event_neg_risk) or bool(r.negRisk is True),
            "event_n_markets": int(r.event_n_markets),
            "sched_life_days": (t_end - t_created) / DAY if np.isfinite(t_end) else np.nan,
            "q_len_words": len(q.split()),
            "q_has_by": int(" by " in ql or ql.startswith("by ")),
            "q_starts_will": int(ql.startswith("will")),
            "q_has_digit": int(any(ch.isdigit() for ch in q)),
            "question": q,
            "t_res": r.t_res,
        }
        for anchor, t0 in [("sched", t_end), ("res", t_res)]:
            if not np.isfinite(t0):
                continue
            for h in HORIZONS:
                tau = t0 - h * DAY
                feats = path_features(ts, ps, tau)
                if feats is None:
                    continue
                rows.append({
                    **static, **feats, "anchor": anchor, "h": h,
                    "snap_ts": tau,
                    "frac_life": feats["days_live"] / max(static["sched_life_days"], 0.1)
                    if np.isfinite(static["sched_life_days"]) else np.nan,
                })

    df = pd.DataFrame(rows)
    df["snap_month"] = pd.to_datetime(df["snap_ts"], unit="s", utc=True).dt.to_period("M").astype(str)
    df["logit_p"] = np.log(df["p"] / (1 - df["p"]))
    df.to_parquet(f"{ROOT}/data/processed/ml_dataset.parquet", index=False)
    print(f"rows: {len(df)}")
    print(df.groupby(["anchor", "h"])["id"].count().unstack(0).to_string())
    print("\nsnap_month range:", df["snap_month"].min(), "->", df["snap_month"].max())


if __name__ == "__main__":
    main()
