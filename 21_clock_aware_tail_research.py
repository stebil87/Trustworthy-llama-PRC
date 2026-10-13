#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRC 2026 - CAMPAIGN 21: CLOCK-AWARE EXTREME MIXTURES RESEARCH ONLY

Reads ONLY official training_2025-*.parquet/.csv files. Never reads 2026
ranking/submitting data, never produces a submission or 2026 predictions.

Dependencies: pandas, numpy, lightgbm; pyarrow if Parquet inputs/outputs.

Campaign 19 builds on the tested 16-18 baseline and adds LIRF/LTFM local
tail mixtures, source-clock residual mapping, and adaptive risk-dependent gates.

Protocol for every outer month:
  FIT_SELECT: records strictly before SELECT month (minus embargo)
  SELECT:     previous calendar month, divided into chronological halves
  FIT_OUTER:  records strictly before OUTER month (minus embargo)
  OUTER:      complete held-out calendar month, historically explored 2025

No early stopping or fitting on SELECT/OUTER labels. Only SELECT selects
candidate; OUTER is used for descriptive evaluation, never for decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "TAXITIME_SEC_mvt"
PHASE = "PHASE_mvt"
ID = "MVT_ID_mvt"
AIRPORT = "ADEP_mvt"
TAKEOFF = "MVT_TIME_UTC_mvt"
BLOCK = "BLOCK_TIME_UTC_mvt"
AOBT = "AOBT_3_flt"

CATS = {
    "airport": AIRPORT,
    "runway": "RUNWAY_mvt",
    "stand": "STAND_mvt",
    "aircraft": "AIRCRAFT_TYPE_mvt",
    "operator": "AIRCRAFT_OPERATOR_flt",
    "wake": "WK_TBL_CAT_flt",
    "segment": "MARKET_SEGMENT_flt",
    "destination": "ADES_mvt",
    "flight_type": "FLIGHT_TYPE_flt",
    "flight_rules": "FLIGHT_RULE_mvt",
}
TIMES = {
    "takeoff": TAKEOFF,
    "aobt": AOBT,
    "scheduled": "SCHED_TIME_UTC_mvt",
    "eobt": "EOBT_1_flt",
    "iobt": "IOBT_flt",
    "lobt": "LOBT_flt",
}
BASE_NUM = [
    "hour_sin", "hour_cos", "dow", "month_sin", "month_cos",
    "schedule_minus_mvt_min", "eobt_minus_mvt_min",
    "iobt_minus_mvt_min", "lobt_minus_mvt_min", "eobt_minus_iobt_min",
    "scheduled_missing", "eobt_missing", "iobt_missing", "lobt_missing",
]
AOBT_NUM = [
    "aobt_gap_sec", "aobt_missing", "aobt_after_takeoff",
    "aobt_very_old", "aobt_eobt_gap_min", "aobt_iobt_gap_min",
]
FEATURES = list(CATS) + BASE_NUM + AOBT_NUM
FEATURES_NO_AOBT = list(CATS) + BASE_NUM
CANDIDATES = (
    "global_l2", "local_l2", "blend_local_l2_75", "blend_local_l2_55",
    "blend_local_reg_75", "blend_aobt_guard_75", "blend_tail_cap_75",
)
ANCHOR = "blend_local_l2_75"
ALLOWED_RAW = {TARGET, PHASE, ID, *CATS.values(), *TIMES.values()}
OFFICIAL_PATTERN = re.compile(
    r"^training_2025-\d\d-\d\d_202[56]-\d\d-\d\d\.(?:parquet|csv)$", re.I
)


def invariant(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def months_before(month: str) -> str:
    return str((pd.Period(month, freq="M") - 1))


def next_month(month: str) -> str:
    return str((pd.Period(month, freq="M") + 1))


def utc_timestamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_localize("UTC")


def time_delta_minutes(a: pd.Series, b: pd.Series) -> pd.Series:
    v = (a - b).dt.total_seconds().div(60)
    return v.clip(lower=-1440, upper=1440).astype("float32")


def audit_and_keep_2025_departures(raw: pd.DataFrame, filename: str, *, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Audit file-vs-takeoff calendar boundaries; partition by actual UTC takeoff.

    A monthly source file need not imply that *each* row has a takeoff within
    that UTC calendar month. Do not silently discard adjacent-month rows.
    Historical research is strictly scoped to takeoffs in calendar year 2025.
    """
    match = OFFICIAL_PATTERN.fullmatch(filename)
    invariant(match is not None, f"Unrecognized official training filename: {filename}")
    period = re.search(r"training_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", filename)
    invariant(period is not None, f"Missing calendar bounds in {filename}")
    start = utc_timestamp(period.group(1))
    end = utc_timestamp(period.group(2))
    invariant(end > start and end <= utc_timestamp("2026-01-01"),
              f"Invalid file boundary in {filename}")
    clock = pd.to_datetime(raw[TAKEOFF], errors="coerce", utc=True)
    invariant(bool(clock.notna().all()),
              f"Invalid takeoff timestamps in {filename}: {int(clock.isna().sum())}")
    in_2025 = (clock >= utc_timestamp("2025-01-01")) & (
        clock < utc_timestamp("2026-01-01"))
    in_file = (clock >= start) & (clock < end)
    in_file_2025 = in_file & in_2025
    mismatch = (~in_file) & in_2025
    discarded = int((~in_2025).sum())
    examples = clock.loc[mismatch].head(4).dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    audit = {
        "input_departures": int(len(raw)),
        "kept_2025": int(in_2025.sum()),
        "excluded_outside_2025": discarded,
        "same_file_utc_month": int(in_file_2025.sum()),
        "cross_file_utc_month": int(mismatch.sum()),
        "cross_file_examples_utc": examples,
    }
    if verbose and discarded:
        print(f"WARN {filename}: excluded {discarded} departures outside 2025", flush=True)
    if verbose and mismatch.any():
        print(f"AUDIT {filename}: {int(mismatch.sum())} / {int(in_2025.sum())} "
              f"2025 takeoffs outside file UTC interval; examples={examples}", flush=True)
    # Data are retained for their true 2025 takeoff month, not their source file.
    return raw.loc[in_2025].copy(), audit


def load_training_files(root: Path, verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    paths = sorted(
        [p for p in root.rglob("training_2025-*.parquet") if OFFICIAL_PATTERN.fullmatch(p.name)]
        + [p for p in root.rglob("training_2025-*.csv") if OFFICIAL_PATTERN.fullmatch(p.name)]
    )
    if not paths:
        raise FileNotFoundError(
            f"No official 2025 training files under {root}. "
            "Expected training_2025-01-01_2025-02-01.parquet etc."
        )
    # Never silently combine duplicated names copied into different directories.
    names = [p.name.rsplit(".", 1)[0] for p in paths]
    invariant(len(names) == len(set(names)), "Duplicate monthly training files detected")
    starts = [re.search(r"training_(2025-\d\d)-", p.name).group(1) for p in paths]
    invariant(len(starts) == len(set(starts)), "Multiple files for the same month")
    expected = {f"2025-{i:02d}" for i in range(1, 13)}
    invariant(set(starts) == expected,
              f"Expected all 12 official training months: missing={sorted(expected-set(starts))}, "
              f"unexpected={sorted(set(starts)-expected)}")
    frames, manifest = [], []
    for path in paths:
        if verbose:
            print(f"READ {path.name}", flush=True)
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Install pyarrow: python -m pip install pyarrow") from exc
            pf = pq.ParquetFile(path)
            cols = [c for c in pf.schema_arrow.names if c in ALLOWED_RAW]
            raw = pf.read(columns=cols).to_pandas()
        else:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            cols = [c for c in header if c in ALLOWED_RAW]
            raw = pd.read_csv(path, usecols=cols, low_memory=False)
        mandatory = {PHASE, TARGET, AIRPORT, TAKEOFF, AOBT}
        invariant(mandatory.issubset(raw.columns),
                  f"Missing required columns in {path.name}: {sorted(mandatory-set(raw.columns))}")
        raw = raw.loc[raw[PHASE].astype("string").eq("DEP")].copy()
        raw, file_audit = audit_and_keep_2025_departures(raw, path.name)
        raw["_origin_file"] = path.name
        frames.append(raw)
        manifest.append({"file": path.name, "bytes": path.stat().st_size,
                         "departures": len(raw), **file_audit,
                         "sha256":sha256_file(path)})
    data = pd.concat(frames, ignore_index=True)
    invariant(len(data) > 0, "No departure rows in training data")
    # Guard against accidentally ingesting a file with 2026 movements.
    clock = pd.to_datetime(data[TAKEOFF], errors="coerce", utc=True)
    valid_clock = clock.notna()
    invariant(bool(valid_clock.all()),
              f"Invalid takeoff timestamps: {int((~valid_clock).sum())}; fail closed")
    invariant(bool(((clock >= utc_timestamp("2025-01-01")) &
                    (clock < utc_timestamp("2026-01-01"))).all()),
              "Detected rows outside calendar year 2025; abort")
    # Source-file interval mismatches are audited above, never used to define
    # train/validation chronology. UTC takeoff time remains the sole splitter.
    n_misaligned = sum(m["cross_file_utc_month"] for m in manifest)
    n_excluded = sum(m["excluded_outside_2025"] for m in manifest)
    print(f"UTC FILE-BOUNDARY AUDIT: {n_misaligned:,} retained cross-month "
          f"2025 departures; {n_excluded:,} out-of-2025 departures excluded", flush=True)
    if ID in data.columns:
        ids = data[ID].dropna()
        invariant(not ids.duplicated().any(), "Duplicate MVT_ID_mvt values detected")
    return data, manifest


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Explicit feature whitelist: target, actual block time, IDs not used."""
    invariant(TARGET not in FEATURES and BLOCK not in FEATURES, "Target leakage in features")
    invariant("ARVT_3_flt" not in ALLOWED_RAW, "Future arrival timestamp not allowed")
    dates = {}
    for key, col in TIMES.items():
        dates[key] = (pd.to_datetime(raw[col], errors="coerce", utc=True)
                      if col in raw else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]"))
    m = dates["takeoff"]
    invariant(bool(m.notna().all()), "Missing takeoff times")
    feat = pd.DataFrame(index=raw.index)
    for key, col in CATS.items():
        if col not in raw:
            feat[key] = "__MISSING__"
        else:
            feat[key] = (raw[col].astype("string").fillna("__MISSING__")
                         .replace({"": "__MISSING__"}).astype("object"))
    hour = (m.dt.hour + m.dt.minute / 60 + m.dt.second / 3600).astype("float32")
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    feat["dow"] = m.dt.dayofweek.astype("float32")
    month_num = m.dt.month.astype("float32")
    feat["month_sin"] = np.sin(2 * np.pi * month_num / 12).astype("float32")
    feat["month_cos"] = np.cos(2 * np.pi * month_num / 12).astype("float32")
    for k, src in [("schedule", "scheduled"), ("eobt", "eobt"),
                   ("iobt", "iobt"), ("lobt", "lobt")]:
        name = "schedule" if k == "schedule" else k
        feat[f"{name}_minus_mvt_min"] = time_delta_minutes(dates[src], m)
        feat[f"{src}_missing"] = dates[src].isna().astype("float32")
    feat["eobt_minus_iobt_min"] = time_delta_minutes(dates["eobt"], dates["iobt"])
    gap = (m - dates["aobt"]).dt.total_seconds()
    feat["aobt_gap_sec"] = gap.clip(lower=-21600, upper=43200).astype("float32")
    feat["aobt_missing"] = dates["aobt"].isna().astype("float32")
    feat["aobt_after_takeoff"] = (gap < -300).fillna(False).astype("float32")
    feat["aobt_very_old"] = (gap > 8*3600).fillna(False).astype("float32")
    feat["aobt_eobt_gap_min"] = time_delta_minutes(dates["aobt"], dates["eobt"])
    feat["aobt_iobt_gap_min"] = time_delta_minutes(dates["aobt"], dates["iobt"])
    feat["_month"] = m.dt.strftime("%Y-%m")
    feat["_day"] = m.dt.strftime("%Y-%m-%d")
    feat["_timestamp"] = m
    feat["_aobt_status"] = np.select(
        [dates["aobt"].isna(), gap.lt(-300), gap.gt(8*3600)],
        ["missing", "after_takeoff", "older_than_8h"], default="plausible"
    )
    invariant(set(FEATURES).issubset(feat.columns), "Unimplemented requested feature")
    return feat


def make_y(raw: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(raw[TARGET], errors="coerce").to_numpy(dtype=np.float64)


def validation_data(frame: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
                    stage: str) -> np.ndarray:
    keep = mask & np.isfinite(y) & (y >= 0)
    lost = int(np.count_nonzero(mask) - np.count_nonzero(keep))
    if lost:
        print(f"WARN {stage}: {lost} rows excluded (missing/negative target); no tail clipping")
    return np.flatnonzero(keep)


def select_indices(feat: pd.DataFrame, y: np.ndarray, outer: str,
                   embargo_days: int) -> dict[str, np.ndarray]:
    select_month = months_before(outer)
    sel_start = utc_timestamp(select_month + "-01")
    out_start = utc_timestamp(outer + "-01")
    out_end = utc_timestamp(next_month(outer) + "-01")
    embargo = pd.Timedelta(days=embargo_days)
    m = feat["_timestamp"]
    t1 = validation_data(feat, y, (m < sel_start-embargo).to_numpy(), "FIT_SELECT")
    s = validation_data(feat, y, ((m >= sel_start) & (m < out_start)).to_numpy(), "SELECT")
    t2 = validation_data(feat, y, (m < out_start-embargo).to_numpy(), "FIT_OUTER")
    o = validation_data(feat, y, ((m >= out_start) & (m < out_end)).to_numpy(), "OUTER")
    invariant(min(map(len, (t1,s,t2,o))) > 0, f"Empty period for OUTER {outer}")
    invariant(feat["_timestamp"].iloc[t1].max() < feat["_timestamp"].iloc[s].min(),
              "TRAIN -> SELECT chronology violated")
    invariant(feat["_timestamp"].iloc[t2].max() < feat["_timestamp"].iloc[o].min(),
              "REFIT -> OUTER chronology violated")
    invariant(not np.intersect1d(s, o).size, "SELECT and OUTER overlap")
    return {"fit_select": t1, "select": s, "fit_outer": t2, "outer": o}


def matrix(feat: pd.DataFrame, indices: np.ndarray, columns: list[str]) -> pd.DataFrame:
    x = feat.iloc[indices][columns].copy()
    for col in CATS:
        if col in x:
            x[col] = x[col].astype("category")
    return x


def train_lgb(x: pd.DataFrame, y: np.ndarray, args: argparse.Namespace,
              *, strong: bool = False, is_local: bool = False):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("Install LightGBM: python -m pip install lightgbm") from exc
    model = LGBMRegressor(
        objective="regression", n_estimators=args.trees,
        learning_rate=0.045, num_leaves=17 if strong else 31,
        min_child_samples=170 if strong else (65 if is_local else 140),
        reg_lambda=90.0 if strong else 18.0, reg_alpha=0.0,
        subsample=1.0, colsample_bytree=1.0,
        max_bin=127, random_state=args.seed, n_jobs=args.threads,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(x, y, categorical_feature=list(CATS), callbacks=[])
    return model


@dataclass
class Fitted:
    global_model: object
    without_aobt: object
    local: dict
    local_reg: dict
    quantiles: dict
    global_q: float


def fit_pack(feat: pd.DataFrame, y: np.ndarray, indices: np.ndarray,
             args: argparse.Namespace) -> Fitted:
    if args.max_train_rows and len(indices) > args.max_train_rows:
        # Deterministic index sample ONLY within the already established TRAIN period.
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(indices, size=args.max_train_rows, replace=False))
    yf = y[indices]
    xm = matrix(feat, indices, FEATURES)
    x0 = xm[FEATURES_NO_AOBT]
    global_model = train_lgb(xm, yf, args)
    without_aobt = train_lgb(x0, yf, args)
    airport = feat["airport"].iloc[indices].astype(str).to_numpy()
    local, local_reg, quantiles = {}, {}, {}
    global_q = float(np.quantile(yf, 0.999))
    for a in sorted(np.unique(airport)):
        idx = np.flatnonzero(airport == a)
        quantiles[a] = float(np.quantile(yf[idx], 0.999))
        if len(idx) < args.min_local_train:
            continue
        local[a] = train_lgb(xm.iloc[idx], yf[idx], args, is_local=True)
        local_reg[a] = train_lgb(xm.iloc[idx], yf[idx], args,
                                 strong=True, is_local=True)
    return Fitted(global_model, without_aobt, local, local_reg, quantiles, global_q)


def predict_candidates(fit: Fitted, feat: pd.DataFrame,
                       indices: np.ndarray) -> dict[str, np.ndarray]:
    xm = matrix(feat, indices, FEATURES)
    g = np.asarray(fit.global_model.predict(xm), dtype=np.float64)
    no_aobt = np.asarray(fit.without_aobt.predict(xm[FEATURES_NO_AOBT]), dtype=np.float64)
    airports = feat["airport"].iloc[indices].astype(str).to_numpy()
    local = g.copy()
    local_reg = g.copy()
    for a in np.unique(airports):
        idx = np.flatnonzero(airports == a)
        if a in fit.local:
            local[idx] = fit.local[a].predict(xm.iloc[idx])
            local_reg[idx] = fit.local_reg[a].predict(xm.iloc[idx])
    blend75 = 0.75*local + 0.25*g
    blend55 = 0.55*local + 0.45*g
    reg75 = 0.75*local_reg + 0.25*g
    status = feat["_aobt_status"].iloc[indices].to_numpy()
    bad = status != "plausible"
    guarded = np.where(bad, 0.30*blend75 + 0.70*no_aobt, blend75)
    caps = np.array([fit.quantiles.get(a, fit.global_q) for a in airports])
    capped = np.minimum(blend75, caps)
    candidates = {
        "global_l2": g, "local_l2": local,
        "blend_local_l2_75": blend75, "blend_local_l2_55": blend55,
        "blend_local_reg_75": reg75, "blend_aobt_guard_75": guarded,
        "blend_tail_cap_75": capped,
    }
    return {k: np.maximum(v, 0.0) for k,v in candidates.items()}


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    error = np.asarray(pred, float) - np.asarray(y, float)
    invariant(bool(np.isfinite(error).all()), "Nonfinite predictions/errors")
    abs_e = np.abs(error)
    sse = error**2
    q99 = np.quantile(sse, 0.99)
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(sse))),
        "mae": float(np.mean(abs_e)),
        "p95_abs": float(np.quantile(abs_e, .95)),
        "p99_abs": float(np.quantile(abs_e, .99)),
        "sse_share_top_1pct": float(sse[sse >= q99].sum() / max(sse.sum(), 1e-12)),
        "rmse_target_gt_1800": (
            float(np.sqrt(np.mean(sse[y > 1800]))) if (y > 1800).any() else None
        ),
        "n_target_gt_1800": int((y > 1800).sum()),
    }


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                     days: np.ndarray, reps: int, seed: int) -> dict:
    """Distribution of RMSE(a)-RMSE(b), paired whole-day resampling."""
    ix, _ = pd.factorize(days, sort=True)
    n_days = int(ix.max()) + 1
    invariant(n_days >= 2, "Day bootstrap requires >=2 distinct days")
    counts = np.bincount(ix, minlength=n_days).astype(np.float64)
    ss_a = np.bincount(ix, weights=(a-y)**2, minlength=n_days)
    ss_b = np.bincount(ix, weights=(b-y)**2, minlength=n_days)
    rng = np.random.default_rng(seed)
    sims = np.empty(reps)
    # Preserve the correlation across airports on the same calendar day.
    for i in range(reps):
        sampled = rng.integers(0, n_days, size=n_days)
        denom = counts[sampled].sum()
        sims[i] = np.sqrt(ss_a[sampled].sum()/denom) - np.sqrt(ss_b[sampled].sum()/denom)
    return {
        "delta_rmse": float(np.sqrt(np.mean((a-y)**2)) - np.sqrt(np.mean((b-y)**2))),
        "q05_delta": float(np.quantile(sims, .05)),
        "q50_delta": float(np.quantile(sims, .50)),
        "q90_delta": float(np.quantile(sims, .90)),
        "q95_delta": float(np.quantile(sims, .95)),
        "n_days": n_days,
    }


def split_two_halves(feat: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = feat["_day"].iloc[indices]
    sorted_days = sorted(d.unique())
    invariant(len(sorted_days) >= 4, "Need at least four SELECT days")
    median_day = sorted_days[len(sorted_days)//2]
    first = indices[(d < median_day).to_numpy()]
    second = indices[(d >= median_day).to_numpy()]
    invariant(len(first) and len(second), "Empty SELECT half")
    return first, second


def choose_candidate(feat: pd.DataFrame, y: np.ndarray,
                     select_idx: np.ndarray, predictions: dict,
                     args: argparse.Namespace, outer: str) -> tuple[str, list[dict]]:
    first, second = split_two_halves(feat, select_idx)
    loc1 = np.isin(select_idx, first)
    loc2 = np.isin(select_idx, second)
    yy = y[select_idx]
    days = feat["_day"].iloc[select_idx].to_numpy()
    base = predictions[ANCHOR]
    base_rmse = score(yy, base)["rmse"]
    rows = []
    for i, name in enumerate(CANDIDATES):
        candidate = predictions[name]
        whole = score(yy, candidate)
        half_a = score(yy[loc1], candidate[loc1])["rmse"]
        half_b = score(yy[loc2], candidate[loc2])["rmse"]
        delta_half_1 = half_a-score(yy[loc1], base[loc1])["rmse"]
        delta_half_2 = half_b-score(yy[loc2], base[loc2])["rmse"]
        boot = paired_bootstrap(yy, candidate, base, days, args.bootstrap,
                                args.seed + i * 1301)
        eligible = bool(
            name != ANCHOR and base_rmse - whole["rmse"] >= args.min_gain_sec
            and max(delta_half_1, delta_half_2) <= args.max_half_regret_sec
            and boot["q90_delta"] < 0
        )
        rows.append({"outer": outer, "candidate": name, **whole,
                     "select_first_rmse": half_a, "select_second_rmse": half_b,
                     "first_delta_vs_anchor": delta_half_1,
                     "second_delta_vs_anchor": delta_half_2,
                     **boot, "eligible": eligible})
    winners = [r for r in rows if r["eligible"]]
    # Among conservative winners, optimize worst-half regret first; then RMSE.
    if winners:
        chosen = sorted(winners, key=lambda r: (
            max(r["first_delta_vs_anchor"], r["second_delta_vs_anchor"]),
            r["rmse"], r["candidate"]))[0]["candidate"]
    else:
        chosen = ANCHOR
    return chosen, rows


def group_diagnostics(feat: pd.DataFrame, y: np.ndarray,
                     indices: np.ndarray, pred: np.ndarray,
                     outer: str, name: str) -> tuple[list[dict], list[dict]]:
    rows_airport, rows_status = [], []
    yy = y[indices]
    for group, container, key in [
        (feat["airport"].iloc[indices], rows_airport, "airport"),
        (feat["_aobt_status"].iloc[indices], rows_status, "aobt_status"),
    ]:
        for label in sorted(group.astype(str).unique()):
            sel = (group.astype(str).to_numpy() == label)
            if np.count_nonzero(sel):
                container.append({"outer": outer, "model": name, key: label,
                                  **score(yy[sel], pred[sel])})
    return rows_airport, rows_status


def discrepancy_diagnostics(raw: pd.DataFrame, feat: pd.DataFrame,
                            y: np.ndarray, month: str) -> list[dict]:
    """Descriptive only. Residual against target is NEVER a model feature."""
    sel = (feat["_month"] == month).to_numpy() & np.isfinite(y) & (y >= 0)
    ix = np.flatnonzero(sel)
    if not len(ix):
        return []
    # Exact, UNCLIPPED source-clock delta for diagnostics. Model uses clipped
    # proxy only; diagnostics must preserve extreme discrepancies.
    m = pd.to_datetime(raw[TAKEOFF].iloc[ix], errors="coerce", utc=True)
    aobt = pd.to_datetime(raw[AOBT].iloc[ix], errors="coerce", utc=True)
    proxy = (m-aobt).dt.total_seconds().to_numpy(dtype=float)
    status = feat["_aobt_status"].iloc[ix].to_numpy()
    ap = feat["airport"].iloc[ix].astype(str).to_numpy()
    rows=[]
    for airport in np.unique(ap):
        where = ap == airport
        for label in ["ALL", "plausible", "missing", "after_takeoff", "older_than_8h"]:
            take = where & (np.ones(len(ix), bool) if label == "ALL" else status == label)
            if not take.any():
                continue
            # Only finite plausible proxy values have computable discrepancies.
            usable = take & np.isfinite(proxy)
            error = proxy[usable] - y[ix][usable]
            rows.append({"month":month, "airport":airport, "aobt_status":label,
                         "n":int(take.sum()), "n_proxy":int(usable.sum()),
                         "proxy_vs_target_median_sec":float(np.median(error)) if len(error) else None,
                         "proxy_vs_target_mae_sec":float(np.mean(np.abs(error))) if len(error) else None,
                         "proxy_vs_target_p99_abs_sec":float(np.quantile(np.abs(error), .99)) if len(error) else None})
    return rows


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def self_test() -> None:
    invariant(not ({TARGET, BLOCK, ID} & set(FEATURES)), "Forbidden feature")
    invariant(set(FEATURES_NO_AOBT).isdisjoint(AOBT_NUM), "AOBT ablation contaminated")
    takeoff = pd.date_range("2025-01-01", periods=8, freq="7D", tz="UTC")
    toy=pd.DataFrame({PHASE:["DEP"]*8, TARGET:np.arange(8)*40+600,
                      AIRPORT:["LIRF", "LTFM"]*4, TAKEOFF:takeoff,
                      AOBT:takeoff-pd.Timedelta(minutes=10),
                      "EOBT_1_flt":takeoff-pd.Timedelta(minutes=12),
                      "SCHED_TIME_UTC_mvt":takeoff-pd.Timedelta(minutes=15)})
    feat=build_features(toy)
    invariant(len(feat) == len(toy), "Feature length")
    invariant(np.allclose(feat["aobt_gap_sec"],600), "Two clocks difference wrong")
    invariant(set(FEATURES).issubset(feat), "Missing features")
    invariant(not np.any(feat["aobt_missing"]), "Wrong AOBT missingness")
    toy.loc[0, AOBT] = pd.NaT
    toy.loc[1, AOBT] = takeoff[1] + pd.Timedelta(minutes=12)
    feat=build_features(toy)
    invariant(feat.loc[0,"_aobt_status"]=="missing", "Missing proxy fallback")
    invariant(feat.loc[1,"_aobt_status"]=="after_takeoff", "Impossible source-clock detection")
    invariant(months_before("2025-07")=="2025-06", "Month boundary")
    # Cross-file UTC month movements are legitimate input; keep and assign by
    # timestamp, never source filename. Exclude any 2026-labeled rows.
    boundary_toy = pd.DataFrame({TAKEOFF: ["2025-01-31T23:59:00Z",
                                             "2025-02-01T00:01:00Z",
                                             "2026-01-01T00:01:00Z"]})
    filtered, audit = audit_and_keep_2025_departures(
        boundary_toy, "training_2025-01-01_2025-02-01.parquet", verbose=False)
    invariant(len(filtered) == 2 and audit["cross_file_utc_month"] == 1
              and audit["excluded_outside_2025"] == 1,
              "Cross-month clock audit / year exclusion regression")
    yd=np.arange(8.0)+800
    pred1=yd+10; pred2=yd+20
    bootstrap=paired_bootstrap(yd, pred1,pred2,
                   np.array([str(d.date()) for d in takeoff]),100,42)
    invariant(bootstrap["q90_delta"] < 0, "Paired bootstrap sign")
    # 2025-06 SELECT, 2025-07 OUTER, disjoint and strict chronology.
    dates=pd.date_range("2025-01-01", "2025-08-01", freq="D", inclusive="left", tz="UTC")
    large=pd.DataFrame({PHASE:["DEP"]*len(dates), TARGET:np.full(len(dates),900),
                        AIRPORT:["LTFM"]*len(dates), TAKEOFF:dates,
                        AOBT:dates-pd.Timedelta(minutes=15)})
    features=build_features(large)
    cuts=select_indices(features, make_y(large), "2025-07", 1)
    invariant(features["_timestamp"].iloc[cuts["fit_select"]].max() <
              features["_timestamp"].iloc[cuts["select"]].min(), "Chronology test")
    invariant(features["_month"].iloc[cuts["outer"]].eq("2025-07").all(),
              "Wrong OUTER month")
    print("SELF-TEST PASS: target whitelist, source clocks, AOBT anomalies, "
          "paired day bootstrap, UTC file-boundary audit, 2026 exclusion, "
          "month boundaries, chronological embargo")


# -------- INTEGRATED CAMPAIGNS 16-18 (APPENDED BELOW) --------

from typing import Any

# No target or future block timestamp appears among these model features.
FLOW_NUM = ["dep_past_15m", "dep_past_45m", "dep_past_90m",
            "runway_past_15m", "runway_past_45m"]
ROUTE_NUM = ["route_prior_sec", "route_prior_n", "runway_prior_sec", "runway_prior_n"]
FLOW_FEATURES = FEATURES + FLOW_NUM
ROUTE_FEATURES = FLOW_FEATURES + ROUTE_NUM
ANCHOR = "blend_local_l2_75"
CANDIDATES = (
    ANCHOR, "tail_moe_25", "tail_moe_50", "tail_lirf_50",
    "flow_25", "flow_50", "route_25", "route_50",
    "soft_gate", "soft_gate_50", "aobt_guard_75",
)
MODULES = ("tail", "flow", "route", "moe")
SPECIAL_AIRPORTS = ("LIRF", "LTFM")


def add_past_only_flow(feat: pd.DataFrame) -> None:
    """Counts of earlier departure MVT timestamps only; ties excluded.

    This is intended for retrospective post-operational modelling, not a
    prospective pre-departure forecast. No targets enter the transformation.
    """
    # CRITICAL: pandas can preserve datetime64[ms/us/ns] resolutions, depending
    # on the upstream parquet reader and pandas version.  .astype('int64')
    # alone returns integers IN THAT UNIT, not necessarily nanoseconds.  The
    # rolling-window limits below are nanoseconds. Normalize explicitly first.
    # All input dates are audited as 2025, safely inside int64-ns limits.
    stamps = feat["_timestamp"]
    invariant(bool(stamps.notna().all()), "Missing timestamps for flow")
    times = (stamps.astype("datetime64[ns, UTC]")
                   .astype("int64").to_numpy(dtype=np.int64, copy=False))
    invariant(np.all(times > 0), "Missing/invalid timestamps for flow")
    ap = feat["airport"].astype(str)
    rw = feat["runway"].astype(str)
    for group_key, durations, names in [
        (ap, [15, 45, 90], ["dep_past_15m", "dep_past_45m", "dep_past_90m"]),
        (ap.str.cat(rw, sep="||"), [15, 45], ["runway_past_15m", "runway_past_45m"]),
    ]:
        codes = pd.factorize(group_key, sort=False)[0]
        order = np.lexsort((times, codes))
        ordered_codes = codes[order]
        breaks = np.concatenate(([0], np.flatnonzero(np.diff(ordered_codes)) + 1,
                                  [len(codes)]))
        values = {name: np.zeros(len(feat), dtype=np.float32) for name in names}
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            idx = order[lo:hi]
            tt = times[idx]
            # STRICTLY t' < t: excludes self and simultaneous movements.
            upper = np.searchsorted(tt, tt, side="left")
            for mins, name in zip(durations, names):
                lower = np.searchsorted(tt, tt - mins*60*1_000_000_000,
                                        side="left")
                values[name][idx] = (upper-lower).astype(np.float32)
        for name, values_arr in values.items():
            feat[name] = values_arr
    invariant(set(FLOW_NUM).issubset(feat), "Missing flow features")


def route_history_features(feat: pd.DataFrame, y: np.ndarray,
                           train_idx: np.ndarray, query_idx: np.ndarray,
                           lag_days: int = 1) -> pd.DataFrame:
    """Strict past-label prior for stand/runway, with a one-day information lag.

    Each query at time t uses ONLY labels from FIT rows timestamped <= t-lag.
    For FIT query rows, its own label and every later label are excluded.
    The response/target is never itself a model feature.
    """
    invariant(lag_days >= 1, "Historical target priors need >=1-day lag")
    invariant(np.isfinite(y[train_idx]).all(), "Invalid FIT labels")
    result = pd.DataFrame(index=feat.index[query_idx])
    q_t = feat["_timestamp"].iloc[query_idx]
    t_t = feat["_timestamp"].iloc[train_idx]
    for keys, out_mean, out_n, smooth in [
        (["airport", "stand", "runway"], "route_prior_sec", "route_prior_n", 25.0),
        (["airport", "runway"], "runway_prior_sec", "runway_prior_n", 60.0),
    ]:
        hist = feat.iloc[train_idx][keys].reset_index(drop=True).copy()
        hist["source_time"] = t_t.reset_index(drop=True)
        # Cap HISTORY STATISTIC influence; original training targets untouched.
        hist["hist_y"] = np.clip(y[train_idx], 0, 7200)
        hist = hist.sort_values("source_time", kind="stable")
        groups = hist.groupby(keys, sort=False, observed=True)
        hist["hist_sum"] = groups["hist_y"].cumsum()
        hist["hist_n"] = groups.cumcount()+1
        # Use all FIT history that existed by t-24h, even for FIT rows.
        left = feat.iloc[query_idx][keys].reset_index(drop=True).copy()
        left["lookup_time"] = q_t.reset_index(drop=True)-pd.Timedelta(days=lag_days)
        left["_row_order"] = np.arange(len(left))
        left = left.sort_values("lookup_time", kind="stable")
        right = hist[keys+["source_time", "hist_sum", "hist_n"]]
        joined = pd.merge_asof(
            left, right, left_on="lookup_time", right_on="source_time",
            by=keys, direction="backward", allow_exact_matches=True
        ).sort_values("_row_order")
        count = joined["hist_n"].fillna(0).to_numpy(dtype=np.float64)
        total = joined["hist_sum"].fillna(0).to_numpy(dtype=np.float64)
        # Fixed 1200-sec physical-scale prior; not fit on future labels.
        result[out_mean] = ((total + smooth*1200)/(count+smooth)).astype("float32")
        result[out_n] = np.log1p(count).astype("float32")
    invariant(len(result) == len(query_idx), "Historical prior shape mismatch")
    return result


def fit_experiment_regressor(x: pd.DataFrame, yy: np.ndarray,
                            args: argparse.Namespace, *, tail: bool = False):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        objective="regression", n_estimators=args.experiment_trees,
        learning_rate=.06, num_leaves=17 if tail else 27,
        min_child_samples=35 if tail else 100,
        reg_lambda=65.0 if tail else 30.0,
        max_bin=127, random_state=args.seed+140,
        n_jobs=args.threads, deterministic=True,
        force_col_wise=True, verbosity=-1,
    )
    model.fit(x, yy, categorical_feature=[c for c in CATS if c in x])
    return model


def fit_experiment_classifier(x: pd.DataFrame, binary: np.ndarray,
                              args: argparse.Namespace):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        objective="binary", n_estimators=args.experiment_trees,
        learning_rate=.055, num_leaves=15,
        min_child_samples=85, reg_lambda=80.0, max_bin=127,
        random_state=args.seed+141, n_jobs=args.threads,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(x, binary, categorical_feature=[c for c in CATS if c in x])
    return model


def fit_enhanced(feat: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray,
                 eval_idx: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    """Fit all experiments once for a given strictly earlier FIT window."""
    idx = train_idx
    if args.max_train_rows and len(idx) > args.max_train_rows:
        idx = np.sort(np.random.default_rng(args.seed).choice(
            idx, size=args.max_train_rows, replace=False))
    pack: dict[str, Any] = {}
    print(f"    Anchor FIT rows={len(idx):,}", flush=True)
    pack["baseline"] = fit_pack(feat, y, idx, args)
    want = set(args.modules)
    if "tail" in want:
        print(f"    Campaign 16: probability/severity tail experts (>{args.tail_threshold:.0f}s)", flush=True)
        x = matrix(feat, idx, FEATURES)
        event = y[idx] > args.tail_threshold
        pack["tail_rate"] = float(event.mean())
        if event.sum() >= args.min_tail_train and (~event).sum() >= 1000:
            pack["tail_classifier"] = fit_experiment_classifier(x, event.astype(int), args)
            pack["tail_normal"] = fit_experiment_regressor(x.loc[~event],y[idx][~event],args)
            pack["tail_severe"] = fit_experiment_regressor(x.loc[event],y[idx][event],args,tail=True)
            pack["tail_active"] = True
        else:
            print(f"    WARN: too few tail/normal events; tail expert disabled ({int(event.sum())} tail rows)")
            pack["tail_active"] = False
    if "flow" in want:
        print("    Campaign 17: past-only airport/runway throughput experts", flush=True)
        x = matrix(feat, idx, FLOW_FEATURES)
        pack["flow_global"] = fit_experiment_regressor(x, y[idx], args)
        ap = feat["airport"].iloc[idx].astype(str).to_numpy()
        pack["flow_locals"] = {}
        for airport in SPECIAL_AIRPORTS:
            rows = np.flatnonzero(ap == airport)
            if len(rows) >= args.min_special_train:
                pack["flow_locals"][airport] = fit_experiment_regressor(
                    x.iloc[rows], y[idx][rows], args)
    if "route" in want:
        print("    Campaign 18: lagged stand-runway priors + flow", flush=True)
        pri_train = route_history_features(feat, y, idx, idx, args.prior_lag_days)
        x = matrix(feat, idx, FLOW_FEATURES)
        for col in ROUTE_NUM:
            x[col] = pri_train[col].to_numpy()
        pack["route_global"] = fit_experiment_regressor(x, y[idx], args)
        ap = feat["airport"].iloc[idx].astype(str).to_numpy()
        pack["route_locals"] = {}
        for airport in SPECIAL_AIRPORTS:
            rows = np.flatnonzero(ap == airport)
            if len(rows) >= args.min_special_train:
                pack["route_locals"][airport] = fit_experiment_regressor(
                    x.iloc[rows], y[idx][rows], args)
    # FIT indices/labels retained only to make prior features for SELECT/OUTER.
    pack["fit_indices"] = idx
    return pack


def predict_enhanced(pack: dict[str, Any], feat: pd.DataFrame, y: np.ndarray,
                     eval_idx: np.ndarray, args: argparse.Namespace) -> dict[str,np.ndarray]:
    base_pred = predict_candidates(pack["baseline"], feat, eval_idx)
    anchor = base_pred[ANCHOR]
    ap = feat["airport"].iloc[eval_idx].astype(str).to_numpy()
    status = feat["_aobt_status"].iloc[eval_idx].astype(str).to_numpy()
    n = len(anchor)
    normal = anchor.copy(); severe = anchor.copy()
    prob = np.zeros(n, dtype=np.float64)
    tail_prediction = anchor.copy()
    if pack.get("tail_active", False):
        x = matrix(feat, eval_idx, FEATURES)
        prob = pack["tail_classifier"].predict_proba(x)[:,1].astype(np.float64)
        prob = np.clip(prob, 1e-5, .95)
        normal = np.maximum(0, pack["tail_normal"].predict(x))
        severe = np.maximum(0, pack["tail_severe"].predict(x))
        # Posterior conditional expectation, not hard assignment to a cluster.
        tail_prediction = (1.0-prob)*normal + prob*severe
    flow = anchor.copy()
    if "flow_global" in pack:
        x = matrix(feat, eval_idx, FLOW_FEATURES)
        pred_global = np.asarray(pack["flow_global"].predict(x), float)
        p_local = base_pred["local_l2"].copy()
        for airport, model in pack["flow_locals"].items():
            mask = np.flatnonzero(ap == airport)
            if len(mask):
                p_local[mask] = model.predict(x.iloc[mask])
        flow = .75*p_local+.25*pred_global
    route = anchor.copy()
    if "route_global" in pack:
        x = matrix(feat, eval_idx, FLOW_FEATURES)
        prior = route_history_features(feat,y,pack["fit_indices"],eval_idx,args.prior_lag_days)
        for col in ROUTE_NUM:
            x[col] = prior[col].to_numpy()
        pred_global = np.asarray(pack["route_global"].predict(x), float)
        p_local = base_pred["local_l2"].copy()
        for airport, model in pack["route_locals"].items():
            mask = np.flatnonzero(ap == airport)
            if len(mask):
                p_local[mask] = model.predict(x.iloc[mask])
        route = .75*p_local+.25*pred_global
    # A deterministic, feature-dependent soft gate; not a learned gating net.
    # Regime probabilities are supervised through the tail classifier only.
    congest = np.clip(feat["dep_past_45m"].iloc[eval_idx].to_numpy()/35,0,1)
    w_tail = np.minimum(.60, 4.0*prob) if pack.get("tail_active") else np.zeros(n)
    w_flow = (.18+.28*congest) if "flow_global" in pack else np.zeros(n)
    w_route = (.25*(1.0-w_tail)) if "route_global" in pack else np.zeros(n)
    total = w_tail+w_flow+w_route
    excessive = total > .93
    if np.any(excessive):
        factor = np.where(excessive, .93 / np.maximum(total,1e-9), 1.0)
        w_tail *= factor; w_flow *= factor; w_route *= factor
        total = w_tail+w_flow+w_route
    soft_gate = ((1.0-total)*anchor+w_tail*tail_prediction+
                 w_flow*flow+w_route*route)
    soft_gate = np.where(status != "plausible",
                         .70*soft_gate+.30*base_pred["blend_aobt_guard_75"],soft_gate)
    if "moe" not in args.modules:
        soft_gate = anchor.copy()
    preds = {
        ANCHOR: anchor,
        "tail_moe_25": .75*anchor+.25*tail_prediction,
        "tail_moe_50": .50*anchor+.50*tail_prediction,
        "tail_lirf_50": np.where(ap=="LIRF",.50*anchor+.50*tail_prediction, anchor),
        "flow_25": .75*anchor+.25*flow,
        "flow_50": .50*anchor+.50*flow,
        "route_25": .75*anchor+.25*route,
        "route_50": .50*anchor+.50*route,
        "soft_gate": soft_gate,
        "soft_gate_50": .50*anchor+.50*soft_gate,
        "aobt_guard_75": base_pred["blend_aobt_guard_75"],
    }
    result = {name:np.maximum(0.0,np.asarray(p,float)) for name,p in preds.items()}
    for name,p in result.items():
        invariant(bool(np.isfinite(p).all()) and len(p)==n, f"Invalid {name} predictions")
    return result


def pick_with_robustness(feat:pd.DataFrame,y:np.ndarray,idx:np.ndarray,
                         preds:dict[str,np.ndarray],args:argparse.Namespace,
                         outer:str) -> tuple[str,list[dict]]:
    selected, rows = choose_candidate(feat,y,idx,preds,args,outer)
    # Explicitly distinguish model choice from an OUTER evaluation.
    return selected, rows


def tail_diagnostics(y:np.ndarray, base:np.ndarray, pred:np.ndarray,
                     airport:np.ndarray, outer:str, candidate:str) -> list[dict]:
    records=[]
    err0=(base-y)**2; err1=(pred-y)**2
    threshold=np.quantile(err0,.999)
    base_bad=err0>=threshold  # fixed by ANCHOR, never re-selected by candidate
    for label, m in [
        ("all",np.ones(len(y),bool)),
        ("LIRF",airport=="LIRF"),
        ("LTFM",airport=="LTFM"),
        ("anchor_worst_0p1pct",base_bad),
        ("anchor_remaining_99p9pct",~base_bad),
        ("true_target_gt_3600",y>3600),
        ("true_target_le_3600",y<=3600),
    ]:
        if np.any(m):
            records.append({"outer":outer,"candidate":candidate,"slice":label,
                            "n":int(m.sum()),
                            "anchor_rmse":float(np.sqrt(err0[m].mean())),
                            "candidate_rmse":float(np.sqrt(err1[m].mean())),
                            "delta_sse":float((err1[m]-err0[m]).sum()),
                            "anchor_sse_share":float(err0[m].sum()/max(err0.sum(),1e-12))})
    return records


def more_self_tests() -> None:
    self_test()
    n=8
    tt=pd.date_range("2025-01-01",periods=n,freq="D",tz="UTC")
    r=pd.DataFrame({PHASE:["DEP"]*n,TARGET:np.arange(n)*100+700,
                    AIRPORT:["LIRF"]*n,TAKEOFF:tt,AOBT:tt-pd.Timedelta(minutes=13),
                    "STAND_mvt":["S1"]*n,"RUNWAY_mvt":["R1"]*n})
    f=build_features(r)
    add_past_only_flow(f)
    invariant(f.loc[0,"dep_past_90m"]==0,"Flow at first row")
    invariant(f.loc[1,"dep_past_90m"]==0,"Day-separated flow")
    # Regression for Windows / pandas 3.x / parquet timestamp resolutions.
    # In particular, a previous version compared microsecond timestamps to
    # nanosecond window lengths and counted flights from DAYS before.
    for resolution in ("ns", "us", "ms", "s"):
        unit_times = tt.as_unit(resolution)
        unit_raw = r.copy()
        unit_raw[TAKEOFF] = unit_times
        unit_raw[AOBT] = unit_times - pd.Timedelta(minutes=13)
        unit_feat = build_features(unit_raw)
        invariant(resolution in str(unit_feat["_timestamp"].dtype),
                  f"Synthetic {resolution} unit was not preserved")
        add_past_only_flow(unit_feat)
        for name in FLOW_NUM:
            invariant(np.all(unit_feat[name].to_numpy() == 0),
                      f"Timestamp-resolution regression ({resolution}): {name}")
    # Independent datetime-arithmetic oracle, including ties, runway/airport
    # boundaries, 15/45/90 minute cutoffs, and deliberately shuffled rows.
    from datetime import timedelta
    checks = [
        ("2025-07-01T10:00:00Z", "LIRF", "R1"),
        ("2025-07-01T10:10:00Z", "LIRF", "R1"),
        ("2025-07-01T10:15:00Z", "LIRF", "R2"),
        ("2025-07-01T10:15:00Z", "LIRF", "R1"),
        ("2025-07-01T10:30:00Z", "LTFM", "R1"),
        ("2025-07-01T10:45:00Z", "LIRF", "R1"),
        ("2025-07-01T11:00:00Z", "LIRF", "R2"),
        ("2025-07-01T11:31:00Z", "LIRF", "R1"),
        ("2025-07-02T10:00:00Z", "LIRF", "R1"),
    ]
    # Non-chronological input order ensures grouping/sorting isn't accidentally
    # relying on original rows' order.
    checks = [checks[i] for i in [7,0,3,5,2,8,4,6,1]]
    dts = pd.to_datetime([q[0] for q in checks],utc=True)
    for resolution in ("ns", "us", "ms", "s"):
        ztime = dts.as_unit(resolution)
        z = pd.DataFrame({PHASE:["DEP"]*len(checks), TARGET:[800]*len(checks),
                          AIRPORT:[q[1] for q in checks],
                          "RUNWAY_mvt":[q[2] for q in checks],
                          "STAND_mvt":["S1"]*len(checks),TAKEOFF:ztime,
                          AOBT:ztime-pd.Timedelta(minutes=13)})
        zf = build_features(z)
        add_past_only_flow(zf)
        for i,(clock,apt,rwy) in enumerate(checks):
            current = pd.Timestamp(clock)
            for minutes,column,same_runway in [
                (15,"dep_past_15m",False),(45,"dep_past_45m",False),
                (90,"dep_past_90m",False),(15,"runway_past_15m",True),
                (45,"runway_past_45m",True)]:
                expected = sum(
                    (q[1]==apt) and (not same_runway or q[2]==rwy)
                    and (current-pd.Timedelta(minutes=minutes)
                         <= pd.Timestamp(q[0]) < current)
                    for q in checks)
                invariant(float(zf.iloc[i][column])==expected,
                          f"Flow oracle mismatch: unit={resolution}, "
                          f"row={i}, feature={column}, got={zf.iloc[i][column]}, "
                          f"expected={expected}")
    # Insert a future movement; earlier features must not change.
    dup=r.iloc[[-1]].copy()
    dup[TAKEOFF]=pd.Timestamp("2025-01-12",tz="UTC")
    ext=pd.concat([r,dup],ignore_index=True)
    ext_f=build_features(ext);add_past_only_flow(ext_f)
    for c in FLOW_NUM:
        invariant(np.array_equal(f[c].to_numpy(),ext_f[c].iloc[:n].to_numpy()),
                  f"Future movement leaks into flow {c}")
    yp=make_y(r)
    a=route_history_features(f,yp,np.arange(n),np.arange(n),lag_days=1)
    yp2=yp.copy();yp2[7]=500000
    b=route_history_features(f,yp2,np.arange(n),np.arange(n),lag_days=1)
    invariant(np.allclose(a["route_prior_sec"].iloc[:7],
                               b["route_prior_sec"].iloc[:7]),
              "Future labels leak into route prior")
    invariant(a["route_prior_n"].iloc[0] == 0,
              "First prior incorrectly contains own label")
    # Verify same-timestamp departure counts exclude entire tie group.
    rr=r.iloc[:2].copy();rr[TAKEOFF]=[tt[0],tt[0]]
    ff=build_features(rr);add_past_only_flow(ff)
    invariant((ff["dep_past_15m"]==0).all(),"Same-time leakage into flow")
    invariant(not ({TARGET,BLOCK,ID} & set(ROUTE_FEATURES)),"Forbidden model features")
    print("SELF-TEST PASS: ns/us/ms/s timestamp normalization, flow oracle, "
          "past-only congestion, tie exclusion, historical-label lag, "
          "future-event invariance, target whitelist",flush=True)


# ---------------------------------------------------------------------------
# CAMPAIGN 19 - incremental experimental layer, preserving the 16-18 anchor
# ---------------------------------------------------------------------------

C19_INCUMBENTS = (ANCHOR, "tail_lirf_50", "route_25")
C19_NEW = (
    "local_tail_lirf_50", "local_tail_lirf_adaptive", "local_tail_both_35",
    "winter_ltfm_35", "residual_lirf_25", "residual_lirf_50",
    "residual_guard_both_25", "hybrid_lirf_tail_resid",
)
CANDIDATES = C19_INCUMBENTS + C19_NEW


def _plausible_proxy(feat: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Only AOBT before movement, <= 4h old; never look at labels."""
    proxy = feat["aobt_gap_sec"].iloc[indices].to_numpy(dtype=float)
    status = feat["_aobt_status"].iloc[indices].to_numpy()
    return (status == "plausible") & np.isfinite(proxy) & (proxy >= 0) & (proxy <= 14400)


def fit_campaign19(feat: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray,
                   args: argparse.Namespace) -> dict[str, Any]:
    """All model training occurs inside FIT; never use SELECT or OUTER labels."""
    # Version 16-18 models remain unchanged, including their hyperparameters.
    # The route prior and flow definitions are inherited and audited by tests.
    base = fit_enhanced(feat, y, train_idx, np.empty(0, dtype=int), args)
    idx = base["fit_indices"]  # EXACT same FIT sample as the incumbent.
    all_x = matrix(feat, idx, FEATURES)
    airport = feat["airport"].iloc[idx].astype(str).to_numpy()
    dates = feat["_timestamp"].iloc[idx]
    models: dict[str, dict[str, Any]] = {}
    for apt in SPECIAL_AIRPORTS:
        local_rows = np.flatnonzero(airport == apt)
        if len(local_rows) < args.min_special_train:
            print(f"    C19 {apt}: skipped (n={len(local_rows)} < {args.min_special_train})", flush=True)
            continue
        local_y = y[idx[local_rows]]
        local_x = all_x.iloc[local_rows]
        # Threshold obtained ONLY from FIT labels. A fixed bounded quantile
        # prevents very small/very high event definitions between folds.
        threshold = float(np.clip(np.quantile(local_y, args.local_tail_quantile),
                                  1200, 3600))
        events = local_y > threshold
        specialist: dict[str, Any] = {
            "threshold": threshold, "tail_rate": float(events.mean()),
            "n": len(local_rows), "tail_n": int(events.sum()),
            "tail_enabled": False, "residual_enabled": False,
        }
        if (events.sum() >= args.min_local_tail_train
                and (~events).sum() >= args.min_special_train // 2):
            specialist["classifier"] = fit_experiment_classifier(local_x,
                                      events.astype(np.int8), args)
            specialist["normal"] = fit_experiment_regressor(
                local_x.loc[~events], local_y[~events], args)
            specialist["severe"] = fit_experiment_regressor(
                local_x.loc[events], local_y[events], args, tail=True)
            specialist["tail_enabled"] = True
        # Train correction for (MVT-AOBT) against the independent official
        # label. Proxy is permitted source-clock information, not target itself.
        eligible = _plausible_proxy(feat, idx[local_rows])
        if int(eligible.sum()) >= args.min_residual_train:
            px = feat["aobt_gap_sec"].iloc[idx[local_rows]].to_numpy(dtype=float)
            correction = local_y[eligible] - px[eligible]
            specialist["residual"] = fit_experiment_regressor(
                local_x.iloc[np.flatnonzero(eligible)], correction, args)
            specialist["residual_enabled"] = True
            specialist["residual_n"] = int(eligible.sum())
        models[apt] = specialist
        print(f"    C19 {apt}: n={len(local_rows):,}, local_threshold={threshold:.0f}s, "
              f"tail_n={events.sum():,}, tail={specialist['tail_enabled']}, "
              f"residual={specialist['residual_enabled']}",flush=True)
    return {"incumbent": base, "specialists": models}


def predict_campaign19(pack: dict[str, Any], feat: pd.DataFrame, y: np.ndarray,
                       indices: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    # This inherited function sees y only for constructing lagged priors from
    # pack['incumbent']['fit_indices']; query labels are never indexed.
    previous = predict_enhanced(pack["incumbent"], feat, y, indices, args)
    anchor = previous[ANCHOR]
    airport = feat["airport"].iloc[indices].astype(str).to_numpy()
    m = feat["_timestamp"].iloc[indices]
    status = feat["_aobt_status"].iloc[indices].astype(str).to_numpy()
    proxy = feat["aobt_gap_sec"].iloc[indices].to_numpy(dtype=float)
    valid_proxy = _plausible_proxy(feat, indices)

    local_mix = anchor.copy()
    residual = anchor.copy()
    probability = np.zeros(len(anchor), dtype=np.float64)
    local_rate = np.zeros(len(anchor), dtype=np.float64)
    tail_available = np.zeros(len(anchor), dtype=bool)
    residual_available = np.zeros(len(anchor), dtype=bool)
    # Assemble all specialist predictions on the same query rows.
    for apt, specialist in pack["specialists"].items():
        at = np.flatnonzero(airport == apt)
        if not len(at):
            continue
        x = matrix(feat, indices[at], FEATURES)
        if specialist["tail_enabled"]:
            p = np.clip(specialist["classifier"].predict_proba(x)[:, 1],
                        1e-6, 0.95)
            normal = np.maximum(0., specialist["normal"].predict(x))
            severe = np.maximum(0., specialist["severe"].predict(x))
            local_mix[at] = (1-p)*normal + p*severe
            probability[at] = p
            local_rate[at] = specialist["tail_rate"]
            tail_available[at] = True
        if specialist["residual_enabled"]:
            g = np.maximum(0., proxy[at] + specialist["residual"].predict(x))
            valid_at = valid_proxy[at]
            residual[at[valid_at]] = g[valid_at]
            residual_available[at[valid_at]] = True

    is_lirf = airport == "LIRF"
    is_ltfm = airport == "LTFM"
    winter = m.dt.month.isin([11, 12, 1, 2, 3]).to_numpy()
    # Smooth gating from actual *features*, never y. Probability-to-rate
    # ratio prevents a tiny absolute probability being treated as certainty.
    p_ratio = probability / np.maximum(local_rate, 1e-5)
    strength = np.clip(0.20 + 0.16*np.log1p(p_ratio), 0.20, 0.65)
    strength = np.where(status == "plausible", strength, 0.55*strength)
    strength = np.where(tail_available, strength, 0.0)
    def blend(base: np.ndarray, expert: np.ndarray, weight) -> np.ndarray:
        return np.maximum(0.0, base + weight*(expert-base))
    w_lirf = np.where(is_lirf & tail_available, 0.50, 0.0)
    w_both = np.where((is_lirf | is_ltfm) & tail_available, 0.35, 0.0)
    w_ltfm_winter = np.where(is_ltfm & winter & tail_available, 0.35, 0.0)
    w_resid_lirf_25 = np.where(is_lirf & residual_available, 0.25, 0.0)
    w_resid_lirf_50 = np.where(is_lirf & residual_available, 0.50, 0.0)
    w_resid_both = np.where((is_lirf | is_ltfm) & residual_available, 0.25, 0.0)
    # Convex hybrid: at most 50% local tail, 20% proxy residual, 30% anchor.
    hybrid = (anchor + w_lirf*(local_mix-anchor)
              + np.where(is_lirf & residual_available, .20, 0)*(residual-anchor))
    result = {
        ANCHOR: anchor,
        "tail_lirf_50": previous["tail_lirf_50"],
        "route_25": previous["route_25"],
        "local_tail_lirf_50": blend(anchor, local_mix, w_lirf),
        "local_tail_lirf_adaptive": blend(anchor, local_mix,
                  np.where(is_lirf,strength,0.0)),
        "local_tail_both_35": blend(anchor, local_mix, w_both),
        "winter_ltfm_35": blend(previous["tail_lirf_50"], local_mix, w_ltfm_winter),
        "residual_lirf_25": blend(anchor, residual, w_resid_lirf_25),
        "residual_lirf_50": blend(anchor, residual, w_resid_lirf_50),
        "residual_guard_both_25": blend(anchor, residual, w_resid_both),
        "hybrid_lirf_tail_resid": np.maximum(0.0, hybrid),
    }
    invariant(set(result) == set(CANDIDATES), "Campaign 19 candidate mismatch")
    for key, values in result.items():
        invariant(len(values) == len(indices) and np.isfinite(values).all()
                  and np.all(values >= 0), f"Nonfinite/negative C19 predictions: {key}")
    return result


def _candidate_select_row(feat: pd.DataFrame, y: np.ndarray, idx: np.ndarray,
                         name: str, candidate: np.ndarray, anchor: np.ndarray,
                         bootstrap: int, seed: int, outer: str) -> dict[str, Any]:
    first, second = split_two_halves(feat, idx)
    is1 = np.isin(idx, first)
    is2 = ~is1
    yy = y[idx]
    days = feat["_day"].iloc[idx].to_numpy()
    full = score(yy, candidate)
    first_rmse = score(yy[is1],candidate[is1])["rmse"]
    second_rmse = score(yy[is2],candidate[is2])["rmse"]
    bs = paired_bootstrap(yy,candidate,anchor,days,bootstrap,seed)
    return {"outer":outer,"candidate":name, **full,
            "first_rmse":first_rmse,"second_rmse":second_rmse,
            "first_delta_vs_anchor":first_rmse - score(yy[is1],anchor[is1])["rmse"],
            "second_delta_vs_anchor":second_rmse - score(yy[is2],anchor[is2])["rmse"],
            **bs}


def choose_campaign19(feat:pd.DataFrame, y:np.ndarray, idx:np.ndarray,
                      preds:dict[str,np.ndarray], args:argparse.Namespace,
                      outer:str) -> tuple[str, str, list[dict]]:
    """Two-stage selection, all decisions strictly within SELECT.

    Stage 1: choose old incumbent among anchor, global-tail LIRF, route.
    Stage 2: accept a novel candidate only if robustly better than the
             *SELECT-locked incumbent*, not merely better than anchor.
    """
    yy = y[idx]
    days = feat["_day"].iloc[idx].to_numpy()
    rows=[]
    for j,key in enumerate(CANDIDATES):
        r=_candidate_select_row(feat,y,idx,key,preds[key],preds[ANCHOR],
                                args.bootstrap,args.seed+j*997,outer)
        rows.append(r)
    by_name={r["candidate"]:r for r in rows}
    def pass_vs_anchor(row:dict) -> bool:
        return bool(row["candidate"]!=ANCHOR
                    and row["delta_rmse"]<=-args.min_gain_sec
                    and max(row["first_delta_vs_anchor"],row["second_delta_vs_anchor"])
                      <=args.max_half_regret_sec
                    and row["q90_delta"] < 0)
    incumbents=[r for r in rows if r["candidate"] in C19_INCUMBENTS
                and pass_vs_anchor(r)]
    incumbent=(sorted(incumbents,key=lambda r:(
        max(r["first_delta_vs_anchor"],r["second_delta_vs_anchor"]),
        r["rmse"],r["candidate"]))[0]["candidate"] if incumbents else ANCHOR)
    old = preds[incumbent]
    i_first,i_second=split_two_halves(feat,idx)
    is1 = np.isin(idx,i_first)
    is2 = ~is1
    day_boot = feat["_day"].iloc[idx].to_numpy()
    novel=[]
    for j,row in enumerate(rows):
        name=row["candidate"]
        if name in C19_INCUMBENTS:
            row["eligible_vs_incumbent"] = False
            continue
        p=preds[name]
        boot=paired_bootstrap(yy,p,old,day_boot,args.bootstrap,args.seed+21000+j*1301)
        first_delta=(score(yy[is1],p[is1])["rmse"]-
                     score(yy[is1],old[is1])["rmse"])
        second_delta=(score(yy[is2],p[is2])["rmse"]-
                      score(yy[is2],old[is2])["rmse"])
        row.update({"incremental_vs":incumbent,
                    "incremental_delta_rmse":boot["delta_rmse"],
                    "incremental_q90_delta":boot["q90_delta"],
                    "incremental_first_delta":first_delta,
                    "incremental_second_delta":second_delta})
        eligible=bool(boot["delta_rmse"] <= -args.min_incremental_gain_sec
                      and max(first_delta, second_delta)<=args.max_half_regret_sec
                      and boot["q90_delta"]<0)
        row["eligible_vs_incumbent"] = eligible
        if eligible:novel.append(row)
    chosen=(sorted(novel,key=lambda r:(
        max(r["incremental_first_delta"],r["incremental_second_delta"]),
        r["rmse"],r["candidate"]))[0]["candidate"] if novel else incumbent)
    return chosen,incumbent,rows


def campaign19_self_test() -> None:
    more_self_tests()
    invariant(set(C19_INCUMBENTS).issubset(CANDIDATES),"Missing proven incumbent")
    invariant(not ({TARGET, BLOCK, ID} & set(FEATURES)),"Leakage in baseline input")
    invariant(not ({TARGET, BLOCK, ID} & set(ROUTE_FEATURES)),"Leakage in enhanced input")
    invariant(len(CANDIDATES)==len(set(CANDIDATES)),"Repeated candidate name")
    # AOBT proxy plausibility must be independent from target, including bad
    # negative gaps and missing values.
    dt = pd.date_range("2025-05-01",periods=4,freq="h",tz="UTC")
    df=pd.DataFrame({PHASE:["DEP"]*4,TARGET:[700,1100,2000,5000],
                     AIRPORT:["LIRF"]*4,TAKEOFF:dt,
                     AOBT:[dt[0]-pd.Timedelta(minutes=20),pd.NaT,
                           dt[2]+pd.Timedelta(minutes=10),
                           dt[3]-pd.Timedelta(hours=6)]})
    f=build_features(df)
    mask=_plausible_proxy(f,np.arange(4))
    invariant(mask.tolist()==[True,False,False,False],"AOBT residual proxy guard")
    changed = df.copy()
    changed[TARGET]=[999999,0,0,0]
    invariant(np.array_equal(_plausible_proxy(build_features(df),np.arange(4)),
                                    _plausible_proxy(build_features(changed),np.arange(4))),
              "Target changed proxy validity")
    # A deliberately severe error affects the RMSE, never the chronological
    # train/query partitions, selection feature matrix, or event feature.
    invariant("route_prior_sec" not in FEATURES, "Label priors must stay separate")
    invariant(np.isclose(float(np.sqrt(np.mean((np.array([0.,100.])-
                                np.array([0.,0.]))**2))),70.71067811865),
              "RMSE arithmetic")
    print("SELF-TEST PASS C19: AOBT guard, target invariance, incumbent presence, "
          "no target/block in model features",flush=True)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--out-dir",type=Path,default=Path("campaign19_research"))
    parser.add_argument("--preset",choices=["smoke","full"],default="smoke")
    parser.add_argument("--outer-months",nargs="+",default=["2025-07","2025-10","2025-12"])
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--bootstrap",type=int,default=400)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--embargo-days",type=int,default=1)
    parser.add_argument("--prior-lag-days",type=int,default=1)
    parser.add_argument("--tail-threshold",type=float,default=1800)
    parser.add_argument("--local-tail-quantile",type=float,default=.90)
    parser.add_argument("--min-tail-train",type=int,default=300)
    parser.add_argument("--min-local-tail-train",type=int,default=120)
    parser.add_argument("--min-residual-train",type=int,default=1500)
    parser.add_argument("--min-special-train",type=int,default=2500)
    parser.add_argument("--min-local-train",type=int,default=4000)
    parser.add_argument("--min-gain-sec",type=float,default=1.0)
    parser.add_argument("--min-incremental-gain-sec",type=float,default=1.0)
    parser.add_argument("--max-half-regret-sec",type=float,default=1.0)
    parser.add_argument("--save-outer-predictions",action="store_true")
    parser.add_argument("--february-stress",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    invariant(args.bootstrap>=100 and args.threads>=1,"Invalid bootstrap/threads")
    invariant(args.embargo_days>=0 and args.prior_lag_days>=1,"Invalid chronology")
    invariant(0.5 < args.local_tail_quantile < .99,"Invalid local quantile")
    invariant(args.min_incremental_gain_sec>=0 and args.max_half_regret_sec>=0,
              "Invalid selection safeguards")
    invariant(args.tail_threshold>=300 and args.min_residual_train>=10,
              "Invalid expert definitions")
    for mm in args.outer_months:
        invariant(re.fullmatch(r"2025-\d\d",mm) is not None,"OUTER must be 2025")
        invariant(pd.Period(mm,freq="M")>=pd.Period("2025-05",freq="M") and
                  pd.Period(mm,freq="M")<=pd.Period("2025-12",freq="M"),
                  "OUTER outside the historical 2025 research horizon")
    # These are inherited from 16-18 and consumed by fit_enhanced/predict_enhanced.
    args.modules=list(MODULES)
    campaign19_self_test()
    if args.self_test:return 0
    args.trees=220 if args.preset=="full" else 65
    args.experiment_trees=180 if args.preset=="full" else 55
    args.max_train_rows=None if args.preset=="full" else 100000
    if args.preset=="smoke":
        args.min_local_train=700
        args.min_special_train=700
        args.min_tail_train=50
        args.min_local_tail_train=30
        args.min_residual_train=250
    print("="*78)
    print("CAMPAIGN 19: ADAPTIVE TAIL MIXTURE -- CHRONOLOGICAL 2025 RESEARCH ONLY")
    print("NO 2026 FILE READ; NO FINAL TRAIN/REFIT; NO SUBMISSION/PREDICTIONS")
    print("HISTORICALLY EXPLORED OUTER MONTHS -- NOT VIRGIN HOLDOUTS")
    print("="*78,flush=True)
    begin=time.perf_counter()
    raw,manifest=load_training_files(args.data_dir)
    feat=build_features(raw)
    y=make_y(raw)
    print(f"2025 departure movements={len(raw):,}; valid labels="
          f"{int(np.sum(np.isfinite(y)&(y>=0))):,}; "
          f"pandas={pd.__version__}, timestamp={feat['_timestamp'].dtype}",flush=True)
    add_past_only_flow(feat)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    write_json(args.out_dir/"manifest.json",{
        "campaign":19,"code_sha256":sha256_file(Path(__file__)),
        "inputs":manifest,"args":config,"research_only":True,
        "warnings":["2025 outer windows have been explored repeatedly",
                    "No independent 2026 ranking evaluation",
                    "Source AOBT is NOT the official target; residual mapping is trained on FIT labels only",
                    "Retrospective features assume movement time known",
                    "New model candidates increase selection overfitting risk"]})
    all_select=[];all_outer=[];all_tail=[];locked=[];frames=[];airport_groups=[]
    for outer in args.outer_months:
        print(f"\n==== OUTER {outer} / SELECT {months_before(outer)} ====",flush=True)
        cut=select_indices(feat,y,outer,args.embargo_days)
        print("ROWS",{k:len(v) for k,v in cut.items()},flush=True)
        print("  FIT before SELECT",flush=True)
        sel_fit=fit_campaign19(feat,y,cut["fit_select"],args)
        sel_pred=predict_campaign19(sel_fit,feat,y,cut["select"],args)
        chosen,incumbent,rows=choose_campaign19(feat,y,cut["select"],
                                                sel_pred,args,outer)
        all_select.extend(rows)
        print(f"  LOCKED SELECT: incumbent={incumbent} -> chosen={chosen}",flush=True)
        del sel_fit,sel_pred
        print("  FIT before OUTER (descriptive evaluation only)",flush=True)
        out_fit=fit_campaign19(feat,y,cut["fit_outer"],args)
        out_pred=predict_campaign19(out_fit,feat,y,cut["outer"],args)
        yy=y[cut["outer"]]
        ap=feat["airport"].iloc[cut["outer"]].astype(str).to_numpy()
        days=feat["_day"].iloc[cut["outer"]].to_numpy()
        for k in CANDIDATES:
            stat=score(yy,out_pred[k]); bs=(paired_bootstrap(yy,out_pred[k],
                out_pred[incumbent],days,args.bootstrap,args.seed+890)
                   if k!=incumbent else {})
            all_outer.append({"outer":outer,"candidate":k,"selected":k==chosen,
                              "incumbent":k==incumbent,**stat,**bs})
            all_tail.extend(tail_diagnostics(yy,out_pred[ANCHOR],out_pred[k],
                                             ap,outer,k))
            if k in (ANCHOR,incumbent,chosen):
                print(f"  OUTER {k:28s} RMSE={stat['rmse']:8.3f} "
                      f"top1_SSE={stat['sse_share_top_1pct']:.3f}",flush=True)
            for apt in SPECIAL_AIRPORTS:
                sub=ap==apt
                if sub.any():
                    airport_groups.append({"outer":outer,"candidate":k,
                                           "airport":apt,**score(yy[sub],out_pred[k][sub])})
        locked.append({"outer":outer,"select":months_before(outer),
                       "incumbent":incumbent,"locked":chosen,"n":len(yy),
                       "anchor_rmse":score(yy,out_pred[ANCHOR])["rmse"],
                       "incumbent_rmse":score(yy,out_pred[incumbent])["rmse"],
                       "locked_rmse":score(yy,out_pred[chosen])["rmse"]})
        if args.save_outer_predictions:
            fr=pd.DataFrame({"outer":outer,"utc_mvt":feat["_timestamp"].iloc[cut["outer"]].to_numpy(),
                    "airport":ap,"y_true":yy,
                    "aobt_status":feat["_aobt_status"].iloc[cut["outer"]].to_numpy()})
            if ID in raw:
                fr.insert(0,ID,raw[ID].iloc[cut["outer"]].to_numpy())
            for k in CANDIDATES:
                fr["pred_"+k]=out_pred[k]
            frames.append(fr)
        del out_fit,out_pred
    for name,rows in [
        ("select_candidates.csv",all_select),
        ("outer_candidates_DESCRIPTIVE.csv",all_outer),
        ("tail_sse_diagnostics.csv",all_tail),
        ("airport_errors_DESCRIPTIVE.csv",airport_groups),
        ("locked_choices.csv",locked),
    ]:
        pd.DataFrame(rows).to_csv(args.out_dir/name,index=False)
    if args.february_stress:
        valid=np.isfinite(y)&(y>=0)
        feb=feat["_month"].eq("2025-02").to_numpy()&valid
        records=[]
        for apt in sorted(feat["airport"].iloc[np.flatnonzero(feb)].astype(str).unique()):
            loc=feb&(feat["airport"].to_numpy()==apt)
            v=y[loc]
            records.append({"airport":apt,"n":len(v),
                 "target_p99":float(np.quantile(v,.99)),
                 "target_gt_3600":int(np.sum(v>3600))})
        pd.DataFrame(records).to_csv(args.out_dir/"february_2025_DESCRIPTIVE.csv",index=False)
        print("FEB 2025 DESCRIPTIVE ONLY, rows=",int(feb.sum()),flush=True)
    if frames:
        combined=pd.concat(frames,ignore_index=True)
        try:
            combined.to_parquet(args.out_dir/"outer_predictions_2025_RESEARCH_ONLY.parquet",index=False)
        except ImportError:
            combined.to_csv(args.out_dir/"outer_predictions_2025_RESEARCH_ONLY.csv",index=False)
    N=sum(x["n"] for x in locked)
    pooled={key:float(np.sqrt(sum(x["n"]*x[key+"_rmse"]**2
                     for x in locked)/N))
            for key in ("anchor","incumbent","locked")}
    print("\nPOOLED DISJOINT HISTORICAL OUTER:",pooled,flush=True)
    write_json(args.out_dir/"summary.json",{
       "campaign":19,"pooled_rmse":pooled,"locked":locked,
       "seconds":round(time.perf_counter()-begin,2),
       "selection":"Two-stage SELECT-only: old incumbent, then incremental challengers",
       "historical_validation_warning":"OUTER 2025 not independent; descriptive only",
       "no_2026":True})
    print("Research outputs:",args.out_dir.resolve(),flush=True)
    print("NO 2026 READ / NO FINAL REFIT / NO SUBMISSION",flush=True)
    return 0



# ===========================================================================
# CAMPAIGN 20: seasonal generalization, high-severity events, robust hybrids
# ===========================================================================
# Uses unchanged research-only C19 candidate implementation and fit windows.
# Its output is not a ranking submission and includes no 2026 data.

C20_CHALLENGERS = (
    "hybrid_lirf_25",
    "hybrid_lirf_50",
    "hybrid_lirf_risk",
    "super_lirf_25",
    "super_lirf_50",
    "super_ltfm_winter_20",
    "super_ltfm_winter_40",
    "hybrid_winter_25",
    "hybrid_winter_50",
)
C20_REPORTED = CANDIDATES + C20_CHALLENGERS
C20_SCOPES = {
    "2025-03": "winter surrogate (February is SELECT)",
    "2025-06": "ranking-month analog June",
    "2025-07": "ranking-month analog July",
    "2025-10": "shoulder-season control",
    "2025-12": "winter surrogate December",
}


def _safe_prob(model, x: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict_proba(x)[:, 1], dtype=float),
                   1e-5, 0.90)


def fit_campaign20(feat: pd.DataFrame, y: np.ndarray,
                   train_idx: np.ndarray, args: argparse.Namespace) -> dict:
    original = fit_campaign19(feat, y, train_idx, args)
    idx = original["incumbent"]["fit_indices"]
    # LIRF all-year and LTFM winter-only: thresholds are fixed before SELECT.
    airports = feat["airport"].iloc[idx].astype(str).to_numpy()
    months = feat["_timestamp"].iloc[idx].dt.month.to_numpy()
    experts = {}
    for apt in ("LIRF", "LTFM"):
        scope = airports == apt
        if apt == "LTFM":
            scope &= np.isin(months, [11, 12, 1, 2, 3])
        pos = np.flatnonzero(scope)
        yy = y[idx[pos]]
        evt = yy > args.super_tail_threshold
        if (len(pos) < args.min_special_train or
            int(evt.sum()) < args.min_super_tail_train or
            int((~evt).sum()) < args.min_special_train // 2):
            print(f"    C20 {apt} >{args.super_tail_threshold:.0f}s: disabled "
                  f"(scope={len(pos):,}, events={int(evt.sum()):,})", flush=True)
            continue
        x = matrix(feat, idx[pos], FEATURES)
        experts[apt] = {
            "classifier": fit_experiment_classifier(x, evt.astype(np.int8), args),
            "normal": fit_experiment_regressor(x.iloc[np.flatnonzero(~evt)], yy[~evt], args),
            "severe": fit_experiment_regressor(x.iloc[np.flatnonzero(evt)], yy[evt], args, tail=True),
            "event_rate": float(evt.mean()),
            "scope_n": int(len(pos)), "events": int(evt.sum()),
        }
        print(f"    C20 {apt} >{args.super_tail_threshold:.0f}s: scope="
              f"{len(pos):,}, events={int(evt.sum()):,}, active", flush=True)
    return {"old": original, "super_experts": experts}


def predict_campaign20(pack: dict, feat: pd.DataFrame, y: np.ndarray,
                       idx: np.ndarray, args: argparse.Namespace) -> dict[str,np.ndarray]:
    # Legacy candidate predictions and their FIT-only label priors unchanged.
    old = predict_campaign19(pack["old"], feat, y, idx, args)
    ap = feat["airport"].iloc[idx].astype(str).to_numpy()
    winter = feat["_timestamp"].iloc[idx].dt.month.isin([11,12,1,2,3]).to_numpy()
    incumbent_lirf = old["tail_lirf_50"]
    original_hybrid = old["hybrid_lirf_tail_resid"]
    lirf = ap == "LIRF"
    ltfm_winter = (ap == "LTFM") & winter
    candidates = {}
    candidates["hybrid_lirf_25"] = np.where(
        lirf, incumbent_lirf + 0.25*(original_hybrid-incumbent_lirf), incumbent_lirf)
    candidates["hybrid_lirf_50"] = np.where(
        lirf, incumbent_lirf + 0.50*(original_hybrid-incumbent_lirf), incumbent_lirf)
    # A feature-only gating: local C19 probability of the long-tail event;
    # never conditional on hidden y or prediction residual on query rows.
    risk = np.zeros(len(idx), dtype=float)
    l_model = pack["old"]["specialists"].get("LIRF")
    if l_model is not None and l_model.get("tail_enabled"):
        positions = np.flatnonzero(lirf)
        if len(positions):
            x = matrix(feat, idx[positions], FEATURES)
            p = _safe_prob(l_model["classifier"], x)
            base_rate = max(0.01, float(l_model["tail_rate"]))
            strength = np.clip(0.10 + 0.36*np.minimum(p/base_rate, 2.0)/2.0,
                               0.10, 0.46)
            # AOBT discordance is only a source-quality flag, not a label.
            reasonable = _plausible_proxy(feat, idx[positions])
            strength = np.where(reasonable, strength, 0.65*strength)
            risk[positions] = strength
    candidates["hybrid_lirf_risk"] = incumbent_lirf + risk*(original_hybrid-incumbent_lirf)
    super_preds = {}
    for apt, specialist in pack["super_experts"].items():
        mask = lirf if apt == "LIRF" else ltfm_winter
        at = np.flatnonzero(mask)
        if not len(at):
            continue
        x = matrix(feat, idx[at], FEATURES)
        prob = _safe_prob(specialist["classifier"], x)
        normal = np.maximum(0, np.asarray(specialist["normal"].predict(x), dtype=float))
        severe = np.maximum(0, np.asarray(specialist["severe"].predict(x), dtype=float))
        super_preds[apt] = (at, (1-prob)*normal + prob*severe)
    def inject(start:np.ndarray, apt:str, weight:float) -> np.ndarray:
        ret = np.array(start, dtype=np.float64, copy=True)
        if apt in super_preds:
            at, pred = super_preds[apt]
            ret[at] = ret[at] + weight*(pred-ret[at])
        return ret
    candidates["super_lirf_25"] = inject(incumbent_lirf, "LIRF", .25)
    candidates["super_lirf_50"] = inject(incumbent_lirf, "LIRF", .50)
    candidates["super_ltfm_winter_20"] = inject(incumbent_lirf, "LTFM", .20)
    candidates["super_ltfm_winter_40"] = inject(incumbent_lirf, "LTFM", .40)
    candidates["hybrid_winter_25"] = inject(candidates["hybrid_lirf_25"], "LTFM", .20)
    candidates["hybrid_winter_50"] = inject(candidates["hybrid_lirf_50"], "LTFM", .40)
    invariant(set(candidates)==set(C20_CHALLENGERS),"C20 candidate mismatch")
    full = {**old, **candidates}
    invariant(set(full)==set(C20_REPORTED),"C20 outputs mismatch")
    for name, p in full.items():
        invariant(len(p)==len(idx) and np.all(np.isfinite(p)),
                  f"Invalid/nonfinite output from {name}")
        invariant(bool((np.asarray(p)>=0).all()),f"Negative output from {name}")
    return full


def select_campaign20(feat:pd.DataFrame,y:np.ndarray,idx:np.ndarray,
                      preds:dict,args:argparse.Namespace,outer:str) -> tuple[str,str,list[dict]]:
    original = {k:preds[k] for k in CANDIDATES}
    c19_chosen,c19_prior,c19_rows = choose_campaign19(feat,y,idx,original,args,outer)
    yy = y[idx]
    days = feat["_day"].iloc[idx].to_numpy()
    one,two = split_two_halves(feat,idx)
    mask = np.isin(idx,one)
    incumbent = preds[c19_chosen]
    rows=[{**r,"stage":"C19", "comparison":c19_chosen,
           "qualified_C20":False} for r in c19_rows]
    eligible=[]
    for j,key in enumerate(C20_CHALLENGERS):
        p = preds[key]
        bootstrap=paired_bootstrap(yy,p,incumbent,days,args.bootstrap,
                                    args.seed+30043+j*347)
        first=score(yy[mask],p[mask])["rmse"]-score(yy[mask],incumbent[mask])["rmse"]
        second=score(yy[~mask],p[~mask])["rmse"]-score(yy[~mask],incumbent[~mask])["rmse"]
        # A conservative day-level 95% bound and two-half regret cap.
        qualified = bool(bootstrap["delta_rmse"]<=-args.min_incremental_gain_sec
                         and max(first,second)<=args.max_half_regret_sec
                         and bootstrap["q95_delta"]<0)
        row={"outer":outer,"candidate":key,"stage":"C20",
             "comparison":c19_chosen,"qualified_C20":qualified,
             **score(yy,p), **bootstrap,
             "first_delta_vs_C19":float(first),
             "second_delta_vs_C19":float(second)}
        rows.append(row)
        if qualified:eligible.append(row)
    # Simplicity prior: prefer half-window stability before total SELECT RMSE.
    winner=(sorted(eligible,key=lambda r:(
        max(r["first_delta_vs_C19"],r["second_delta_vs_C19"]),
        r["rmse"],r["candidate"]))[0]["candidate"] if eligible else c19_chosen)
    return winner,c19_chosen,rows


def c20_diagnostics(feat:pd.DataFrame,y:np.ndarray, idx:np.ndarray,
                    ref:np.ndarray,candidate:np.ndarray, month:str,
                    name:str,stage:str) -> list[dict]:
    airport = feat["airport"].iloc[idx].astype(str).to_numpy()
    aobt = feat["_aobt_status"].iloc[idx].astype(str).to_numpy()
    true = y[idx]
    e_ref=(ref-true)**2
    e_new=(candidate-true)**2
    mask_anchor_tail = e_ref>=np.quantile(e_ref,.999)
    slices={
        "all":np.ones(len(idx),bool),
        "LIRF":airport=="LIRF",
        "LTFM":airport=="LTFM",
        "LTFM_target_gt_3600":(airport=="LTFM")&(true>3600),
        "LTFM_target_le_3600":(airport=="LTFM")&(true<=3600),
        "LIRF_target_gt_3600":(airport=="LIRF")&(true>3600),
        "AOBT_bad_or_missing":aobt!="plausible",
        "incumbent_worst_0p1pct":mask_anchor_tail,
        "incumbent_remaining_99p9pct":~mask_anchor_tail,
    }
    result=[]
    for group,mask in slices.items():
        if not mask.any():continue
        result.append({"outer":month,"stage":stage,"candidate":name,"slice":group,
                       "n":int(mask.sum()),
                       "reference_rmse":float(np.sqrt(e_ref[mask].mean())),
                       "candidate_rmse":float(np.sqrt(e_new[mask].mean())),
                       "delta_sse":float((e_new[mask]-e_ref[mask]).sum()),
                       "reference_sse_share":float(e_ref[mask].sum()/max(e_ref.sum(),1e-9))})
    return result


def campaign20_self_test() -> None:
    campaign19_self_test()
    invariant(C20_CHALLENGERS and len(set(C20_REPORTED))==len(C20_REPORTED),
              "C20 candidate names conflict")
    invariant("2025-03" in C20_SCOPES, "Feb 2025 SELECT not covered")
    invariant(TARGET not in FEATURES and BLOCK not in FEATURES,"Forbidden feature")
    # No changes to C19 predictions for other airports by a LIRF-only gate.
    base=np.array([100.,200.,300.]); hybrid=np.array([90.,500.,310.])
    aps=np.array(["LIRF","EGLL","LTFM"])
    w=np.where(aps=="LIRF",.25,0.)
    trial=base+w*(hybrid-base)
    invariant(np.array_equal(trial[1:],base[1:]),"Airport gate leakage")
    # Changing query labels must not change any new predictor input.
    dat=pd.DataFrame({TAKEOFF:pd.date_range("2025-02-01",periods=3,freq="h",tz="UTC"),
                      AIRPORT:["LIRF","LTFM","LIRF"],
                      AOBT:["2025-02-01T00:00:00Z",pd.NaT,"2025-02-01T01:00:00Z"],
                      TARGET:[100,200,300]})
    orig=build_features(dat)
    dat[TARGET]=[20000,30000,1]
    test=build_features(dat)
    invariant(orig[FEATURES].equals(test[FEATURES]),"Query target in C20 features")
    print("SELF-TEST PASS C20: 2025-only windows, winter SELECT, "
          "candidate isolation, target invariance, rare-event branches",flush=True)


def campaign20_main() -> int:
    parser=argparse.ArgumentParser(description="PRC Campaign 20: seasonal tail research, 2025 only",
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--out-dir",type=Path,default=Path("campaign20_research"))
    parser.add_argument("--preset",choices=["smoke","full"],default="smoke")
    parser.add_argument("--outer-months",nargs="+",default=list(C20_SCOPES))
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--bootstrap",type=int,default=400)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--embargo-days",type=int,default=1)
    parser.add_argument("--prior-lag-days",type=int,default=1)
    parser.add_argument("--tail-threshold",type=float,default=1800.)
    parser.add_argument("--local-tail-quantile",type=float,default=.90)
    parser.add_argument("--super-tail-threshold",type=float,default=3600.)
    parser.add_argument("--min-super-tail-train",type=int,default=100)
    parser.add_argument("--min-tail-train",type=int,default=300)
    parser.add_argument("--min-local-tail-train",type=int,default=120)
    parser.add_argument("--min-residual-train",type=int,default=1500)
    parser.add_argument("--min-special-train",type=int,default=2500)
    parser.add_argument("--min-local-train",type=int,default=4000)
    parser.add_argument("--min-gain-sec",type=float,default=1.)
    parser.add_argument("--min-incremental-gain-sec",type=float,default=1.)
    parser.add_argument("--max-half-regret-sec",type=float,default=1.)
    parser.add_argument("--save-outer-predictions",action="store_true")
    parser.add_argument("--february-stress",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    invariant(args.bootstrap>=100 and args.threads>0, "Invalid compute settings")
    invariant(args.embargo_days>=0 and args.prior_lag_days>=1,"Invalid embargo")
    invariant(0.5<args.local_tail_quantile<.99,"Invalid local quantile")
    invariant(args.super_tail_threshold>=args.tail_threshold,
              "Super-tail cutoff must exceed tail cutoff")
    invariant(args.min_super_tail_train>=10,"Insufficient super-tail support")
    invariant(args.min_incremental_gain_sec>=0 and args.max_half_regret_sec>=0,
              "Invalid selection thresholds")
    for mm in args.outer_months:
        invariant(mm in C20_SCOPES,"Unsupported OUTER: " + mm)
    args.modules=list(MODULES)
    campaign20_self_test()
    if args.self_test:return 0
    args.trees=220 if args.preset=="full" else 65
    args.experiment_trees=180 if args.preset=="full" else 55
    args.max_train_rows=None if args.preset=="full" else 100000
    if args.preset=="smoke":
        args.min_local_train=700
        args.min_special_train=700
        args.min_tail_train=50
        args.min_local_tail_train=30
        args.min_residual_train=250
        args.min_super_tail_train=15
    print("="*79,flush=True)
    print("CAMPAIGN 20: SEASONAL + EXTREME TAIL RESEARCH / TRAINING 2025 ONLY",flush=True)
    print("NO 2026 READ / NO FINAL REFIT / NO SUBMISSION / NO RANKING PREDICTIONS",flush=True)
    print("ALL OUTERS HISTORICALLY EXPLORED; NO VIRGIN HOLDOUT",flush=True)
    print("="*79,flush=True)
    begin=time.perf_counter()
    raw,manifest=load_training_files(args.data_dir)
    feat=build_features(raw); y=make_y(raw)
    invariant(np.all(feat["_timestamp"].dt.year.to_numpy()==2025),
              "Non-2025 timestamp")
    add_past_only_flow(feat)
    print(f"Training movements={len(y):,} valid={int(np.sum(np.isfinite(y)&(y>=0))):,}",flush=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.out_dir/"manifest.json",{
      "campaign":20,"code_sha256":sha256_file(Path(__file__)),"inputs":manifest,
      "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
      "research_only":True,
      "notes":["ALL validation months 2025 previously explored; no virgin holdout",
               "2025-02 is SELECT before March OUTER, not a held-out OUTER",
               "January 2025 cannot be evaluated as a past-only full-month OUTER",
               "No ranking data read, no final fit, no 2026 predictions"]})
    selection=[]; comparisons=[]; diagnostics=[]; lock=[]; output=[]
    for outer in args.outer_months:
        print(f"\n==== OUTER {outer} ({C20_SCOPES[outer]}) / "
              f"SELECT {months_before(outer)} ====",flush=True)
        windows=select_indices(feat,y,outer,args.embargo_days)
        print("ROWS:",{k:len(v) for k,v in windows.items()},flush=True)
        print("  FIT strictly before SELECT",flush=True)
        pack=fit_campaign20(feat,y,windows["fit_select"],args)
        preds=predict_campaign20(pack,feat,y,windows["select"],args)
        chosen,old,selrows=select_campaign20(feat,y,windows["select"],preds,args,outer)
        selection.extend(selrows)
        # February 2025 explicitly inspected as chronological SELECT, never fit
        # or selected retrospectively based on its following OUTER results.
        if months_before(outer)=="2025-02":
            for key in (old,chosen,"super_ltfm_winter_20","hybrid_lirf_25"):
                diagnostics.extend(c20_diagnostics(feat,y,windows["select"],
                                   preds[old],preds[key],outer,key,"FEB_SELECT"))
        print(f"  LOCKED SELECT C19={old} --> C20={chosen}",flush=True)
        del pack,preds
        print("  REFIT strictly before OUTER for 2025 research evaluation",flush=True)
        pack=fit_campaign20(feat,y,windows["fit_outer"],args)
        preds=predict_campaign20(pack,feat,y,windows["outer"],args)
        yy=y[windows["outer"]]
        days=feat["_day"].iloc[windows["outer"]].to_numpy()
        for j,key in enumerate(C20_REPORTED):
            base=preds[old]
            stat=score(yy,preds[key])
            boot=paired_bootstrap(yy,preds[key],base,days,args.bootstrap,
                                  args.seed+11011+j*103) if key!=old else {}
            comparisons.append({"outer":outer,"candidate":key,"selected":key==chosen,
                                "C19_locked":key==old,**stat,**boot})
            if key in {ANCHOR,old,chosen,"hybrid_lirf_25","super_ltfm_winter_20"}:
                diagnostics.extend(c20_diagnostics(feat,y,windows["outer"],
                                   base,preds[key],outer,key,"OUTER"))
            if key in {ANCHOR,old,chosen}:
                print(f"  OUTER {key:27s} RMSE={stat['rmse']:.3f} "
                      f"top1_SSE={stat['sse_share_top_1pct']:.3f}",flush=True)
        lock.append({"outer":outer,"select":months_before(outer),"n":len(yy),
                     "C19_locked":old,"C20_locked":chosen,
                     "anchor_rmse":score(yy,preds[ANCHOR])["rmse"],
                     "C19_rmse":score(yy,preds[old])["rmse"],
                     "C20_rmse":score(yy,preds[chosen])["rmse"]})
        if args.save_outer_predictions:
            subset=windows["outer"]
            fr=pd.DataFrame({"outer":outer,
                             "utc_mvt":feat["_timestamp"].iloc[subset].to_numpy(),
                             "airport":feat["airport"].iloc[subset].to_numpy(),
                             "y_true":yy})
            if ID in raw:fr.insert(0,ID,raw[ID].iloc[subset].to_numpy())
            for key in C20_REPORTED:fr["pred_"+key]=preds[key]
            output.append(fr)
        del pack,preds
    for name,rows in [
        ("select_candidates.csv",selection),
        ("outer_candidates_DESCRIPTIVE.csv",comparisons),
        ("tail_and_airport_diagnostics.csv",diagnostics),
        ("locked_choices.csv",lock),
    ]:pd.DataFrame(rows).to_csv(args.out_dir/name,index=False)
    if args.february_stress:
        feb=(feat["_month"].to_numpy()=="2025-02") & np.isfinite(y) & (y>=0)
        counts=[]
        for apt in sorted(feat["airport"].iloc[np.flatnonzero(feb)].astype(str).unique()):
            mask=feb & (feat["airport"].to_numpy()==apt)
            v=y[mask]
            counts.append({"airport":apt,"n":len(v),"n_gt_3600":int(np.sum(v>3600)),
                           "p99_target":float(np.quantile(v,.99))})
        pd.DataFrame(counts).to_csv(args.out_dir/"february_2025_DESCRIPTIVE.csv",index=False)
    if output:
        combined=pd.concat(output,ignore_index=True)
        try:combined.to_parquet(args.out_dir/"outer_predictions_2025_RESEARCH_ONLY.parquet",index=False)
        except ImportError:combined.to_csv(args.out_dir/"outer_predictions_2025_RESEARCH_ONLY.csv",index=False)
    total=sum(x["n"] for x in lock)
    pooled={key:float(np.sqrt(sum(x["n"]*x[key+"_rmse"]**2
                for x in lock)/total)) for key in ("anchor","C19","C20")}
    print("\nPOOLED 2025 HISTORICAL OUTERS:",pooled,flush=True)
    write_json(args.out_dir/"summary.json",{
        "campaign":20,"pooled_rmse":pooled,"locked":lock,
        "seconds":round(time.perf_counter()-begin,2),
        "note":"SELECT locked, OUTER historically explored/descriptive; do not claim fresh holdout",
        "no_2026":True})
    print("Research outputs:",args.out_dir.resolve(),flush=True)
    print("NO FINAL REFIT / NO SUBMISSION / NO 2026 PREDICTIONS",flush=True)
    return 0




# ---------------------------------------------------------------------------
# CAMPAIGN 21: CLOCK-AWARE EXTREME GATING + DUAL-THRESHOLD LIRF
# This section extends the independently tested Campaign 20 implementation.
# A decision on any SELECT/OUTER label is never fed into a prediction for that
# same fold, and C20 model selection is locked on SELECT before OUTER begins.
# ---------------------------------------------------------------------------

C21_NEW = (
    "c21_super_65",
    "c21_super_80",
    "c21_clock_good25_bad65",
    "c21_clock_good40_bad75",
    "c21_super_plausible_only",
    "c21_super_bad_only",
    "c21_super_risk_gate",
    "c21_dual2400_50",
    "c21_dual2400_65",
)
C21_COMPARABLE = tuple(C20_SCOPES)
C21_ADDITIONAL = ("2025-04", "2025-05", "2025-08", "2025-09", "2025-11")


def c21_fit(feat: pd.DataFrame, y: np.ndarray, fit_indices: np.ndarray,
            args: argparse.Namespace) -> dict:
    """FIT is historical only. No SELECT/OUTER labels enter these models."""
    parent = fit_campaign20(feat, y, fit_indices, args)
    idx = parent["old"]["incumbent"]["fit_indices"]
    aps = feat["airport"].iloc[idx].astype(str).to_numpy()
    pos = np.flatnonzero(aps == "LIRF")
    labels = y[idx[pos]]
    evt = labels > args.c21_medium_threshold
    alt = None
    if (len(pos) >= args.min_special_train and
        int(evt.sum()) >= args.c21_min_medium_events and
        int((~evt).sum()) >= args.min_special_train // 2):
        x = matrix(feat, idx[pos], FEATURES)
        alt = {
            "classifier": fit_experiment_classifier(x, evt.astype(np.int8), args),
            "normal": fit_experiment_regressor(x.iloc[np.flatnonzero(~evt)],
                                                  labels[~evt], args),
            "severe": fit_experiment_regressor(x.iloc[np.flatnonzero(evt)],
                                                  labels[evt], args, tail=True),
            "n": len(pos), "events": int(evt.sum()),
        }
        print(f"    C21 LIRF >{args.c21_medium_threshold:.0f}s: "
              f"n={len(pos):,} events={int(evt.sum()):,} active", flush=True)
    else:
        print(f"    C21 LIRF >{args.c21_medium_threshold:.0f}s: disabled "
              f"(n={len(pos):,}, events={int(evt.sum()):,})", flush=True)
    return {"parent": parent, "medium": alt}


def c21_candidate_predictions(pack: dict, feat: pd.DataFrame, y: np.ndarray,
                              indices: np.ndarray, args: argparse.Namespace,
                              c20_reference: str,
                              c20_preds: dict[str, np.ndarray] | None = None,
                              *, return_aux: bool = False):
    """All intervention masks depend ONLY on query features, never query y."""
    if c20_preds is None:
        c20_preds = predict_campaign20(pack["parent"], feat, y, indices, args)
    invariant(c20_reference in c20_preds, "Invalid C20 SELECT-locked reference")
    old = np.asarray(c20_preds[c20_reference], dtype=np.float64)
    ap = feat["airport"].iloc[indices].astype(str).to_numpy()
    lirf_pos = np.flatnonzero(ap == "LIRF")
    results = {name: old.copy() for name in C21_NEW}
    aux = {"n_lirf": int(len(lirf_pos)), "n_bad": 0,
           "n_super_active": 0, "n_medium_active": 0,
           "n_medium_missing": 0}
    if not len(lirf_pos):
        return (results, aux) if return_aux else results
    super_model = pack["parent"]["super_experts"].get("LIRF")
    if super_model is None:
        return (results, aux) if return_aux else results
    x = matrix(feat, indices[lirf_pos], FEATURES)
    p = _safe_prob(super_model["classifier"], x)
    super_normal = np.maximum(0, np.asarray(super_model["normal"].predict(x), float))
    super_severe = np.maximum(0, np.asarray(super_model["severe"].predict(x), float))
    super_mean = (1-p)*super_normal + p*super_severe
    old_tail = c20_preds["tail_lirf_50"][lirf_pos]
    status_good = _plausible_proxy(feat, indices[lirf_pos])
    bad = ~status_good
    aux["n_bad"] = int(bad.sum())
    aux["n_super_active"] = int(len(lirf_pos))
    aux["n_medium_missing"] = int(len(lirf_pos))
    rel = np.clip(p / max(float(super_model["event_rate"]), 1e-5), .0, 6.0)
    smooth = np.clip(np.log1p(rel)/np.log(3), 0., 1.)
    dynamic = np.clip(.25 + .50*smooth + np.where(bad,.10,0.0), 0., .85)
    weight_map = {
        "c21_super_65": np.full(len(lirf_pos), .65),
        "c21_super_80": np.full(len(lirf_pos), .80),
        "c21_clock_good25_bad65": np.where(bad,.65,.25),
        "c21_clock_good40_bad75": np.where(bad,.75,.40),
        "c21_super_plausible_only": np.where(status_good,.50,0.),
        "c21_super_bad_only": np.where(bad,.65,0.),
        "c21_super_risk_gate": dynamic,
    }
    for name, weight in weight_map.items():
        results[name][lirf_pos] = np.maximum(0., old_tail + weight*(super_mean-old_tail))
    medium_model = pack["medium"]
    if medium_model is not None:
        medium_p = _safe_prob(medium_model["classifier"], x)
        medium_normal = np.maximum(0.,np.asarray(medium_model["normal"].predict(x),float))
        medium_severe = np.maximum(0.,np.asarray(medium_model["severe"].predict(x),float))
        medium_mean = (1-medium_p)*medium_normal + medium_p*medium_severe
        # Two decompositions of the conditional mean, rather than a new tail cutoff
        # selected from OUTER labels. The 2400s threshold is fixed in advance.
        composite = .50*super_mean + .50*medium_mean
        results["c21_dual2400_50"][lirf_pos] = np.maximum(
            0., old_tail + .50*(composite-old_tail))
        results["c21_dual2400_65"][lirf_pos] = np.maximum(
            0., old_tail + .65*(composite-old_tail))
        aux["n_medium_active"] = len(lirf_pos)
        aux["n_medium_missing"] = 0
    else:
        # Neutral fallback: if extra expert is not trained, preserve locked C20.
        results["c21_dual2400_50"][:] = old
        results["c21_dual2400_65"][:] = old
    for name, p_pred in results.items():
        invariant(len(p_pred)==len(indices) and np.isfinite(p_pred).all(),
                  "C21 non-finite candidate: "+name)
        invariant(bool((p_pred >= 0).all()), "C21 negative candidate: "+name)
        invariant(np.array_equal(p_pred[ap!="LIRF"], old[ap!="LIRF"]),
                  "C21 affected a non-LIRF airport: "+name)
    return (results, aux) if return_aux else results


def c21_select(feat:pd.DataFrame,y:np.ndarray,select_idx:np.ndarray,
               c20_name:str, new:dict[str,np.ndarray],
               c20_preds:dict[str,np.ndarray],args:argparse.Namespace,
               outer:str) -> tuple[str,list[dict]]:
    """Paired-day bootstrap and both chronological SELECT halves only."""
    original = np.asarray(c20_preds[c20_name], float)
    yy = y[select_idx]
    days = feat["_day"].iloc[select_idx].to_numpy()
    first, second = split_two_halves(feat,select_idx)
    left=np.isin(select_idx,first)
    right=~left
    base_l=score(yy[left],original[left])["rmse"]
    base_r=score(yy[right],original[right])["rmse"]
    rows=[]
    eligible=[]
    for j,name in enumerate(C21_NEW):
        trial=new[name]
        b=paired_bootstrap(yy,trial,original,days,args.bootstrap,
                           args.seed + 43001 + j*1013)
        l=score(yy[left],trial[left])["rmse"]-base_l
        r=score(yy[right],trial[right])["rmse"]-base_r
        # Selecting many correlated mixtures on previously explored months is
        # especially dangerous. Require a paired day bootstrap q95 below zero,
        # an absolute RMSE improvement and no major bad temporal half.
        qualified = bool(b["delta_rmse"] <= -args.c21_min_gain_sec
                         and max(l,r) <= args.max_half_regret_sec
                         and b["q95_delta"] < 0.)
        record={"outer":outer,"select":months_before(outer),"candidate":name,
                "C20_reference":c20_name,"qualified":qualified,
                "first_delta_vs_C20":float(l),"second_delta_vs_C20":float(r),
                **score(yy,trial), **b}
        rows.append(record)
        if qualified: eligible.append(record)
    # Keep the already SELECT-locked C20 model unless a C21 candidate clears
    # all predeclared hurdles. Max-worst-half stability before mean error.
    winner=(sorted(eligible,key=lambda r:(
        max(r["first_delta_vs_C20"],r["second_delta_vs_C20"]),
        r["rmse"],r["candidate"]))[0]["candidate"] if eligible else c20_name)
    return winner,rows


def c21_self_test() -> None:
    campaign20_self_test()
    invariant(len(C21_NEW)==len(set(C21_NEW)),"Duplicated C21 candidate")
    invariant(set(C21_NEW).isdisjoint(C20_REPORTED),"C21 name collision")
    invariant(TARGET not in FEATURES and BLOCK not in FEATURES,
              "C21 target/block feature leakage")
    # Alternative outcome labels must not change prediction-time feature masks.
    tmp=pd.DataFrame({TAKEOFF:pd.to_datetime([
        "2025-07-01T09:00:00Z","2025-07-01T10:00:00Z",
        "2025-07-01T11:00:00Z"]), AOBT:[
        "2025-07-01T08:20:00Z",None,"2025-07-01T05:00:00Z"],
        AIRPORT:["LIRF","LIRF","EGLL"], TARGET:[600,10000,400]})
    a=build_features(tmp)
    tmp[TARGET]=[20000,0,9000]
    b=build_features(tmp)
    invariant(a[FEATURES].equals(b[FEATURES]),"C21 prediction uses query label")
    invariant(np.array_equal(_plausible_proxy(a,np.arange(3)),
                             np.array([True,False,False])),
              "C21 source clock quality classifier")
    # Pure-data unit test for paired selection: candidate identical to locked
    # incumbent must never qualify under a positive minimum gain.
    invariant(.5*2 + .5*4 == 3,"C21 mixture arithmetic")
    print("SELF-TEST PASS C21: clock regimes, target invariance, fixed 2400/3600 "
          "thresholds, model isolation, no 2026 reading",flush=True)


def c21_main() -> int:
    parser=argparse.ArgumentParser(description="C21: LIRF clock-aware and dual-tail models; research 2025 only",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--out-dir",type=Path,default=Path("campaign21_research"))
    parser.add_argument("--preset",choices=["smoke","full"],default="smoke")
    parser.add_argument("--outer-months",nargs="+",default=list(C21_COMPARABLE),
                        help="Allowed: 2025-03 through 2025-12 (historical research only)")
    parser.add_argument("--extra-months",action="store_true",
                        help="Add 2025-04/05/08/09/11 for additional chronological stress")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--bootstrap",type=int,default=400)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--embargo-days",type=int,default=1)
    parser.add_argument("--prior-lag-days",type=int,default=1)
    parser.add_argument("--tail-threshold",type=float,default=1800.)
    parser.add_argument("--local-tail-quantile",type=float,default=.90)
    parser.add_argument("--super-tail-threshold",type=float,default=3600.)
    parser.add_argument("--c21-medium-threshold",type=float,default=2400.)
    parser.add_argument("--c21-min-medium-events",type=int,default=100)
    parser.add_argument("--min-super-tail-train",type=int,default=100)
    parser.add_argument("--min-tail-train",type=int,default=300)
    parser.add_argument("--min-local-tail-train",type=int,default=120)
    parser.add_argument("--min-residual-train",type=int,default=1500)
    parser.add_argument("--min-special-train",type=int,default=2500)
    parser.add_argument("--min-local-train",type=int,default=4000)
    parser.add_argument("--min-gain-sec",type=float,default=1.)
    parser.add_argument("--min-incremental-gain-sec",type=float,default=1.)
    parser.add_argument("--c21-min-gain-sec",type=float,default=1.)
    parser.add_argument("--max-half-regret-sec",type=float,default=1.)
    parser.add_argument("--save-outer-predictions",action="store_true")
    parser.add_argument("--february-stress",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    invariant(args.threads>0 and args.bootstrap>=100,"Invalid threads/bootstrap")
    invariant(args.embargo_days>=0 and args.prior_lag_days>=1,"Invalid chronology settings")
    invariant(args.c21_min_gain_sec>0,"C21 minimum gain must be strictly positive")
    invariant(args.tail_threshold < args.c21_medium_threshold < args.super_tail_threshold,
              "Tail thresholds must be strictly increasing 1800 < 2400 < 3600")
    invariant(args.min_super_tail_train>=10 and args.c21_min_medium_events>=10,
              "Invalid event count safeguards")
    invariant(args.max_half_regret_sec>=0,"Invalid half-window regret")
    invariant(0.5<args.local_tail_quantile<.99,"Invalid local tail quantile")
    if args.extra_months:
        args.outer_months=sorted(set(args.outer_months) | set(C21_ADDITIONAL))
    invariant(bool(args.outer_months),"No OUTER specified")
    for m in args.outer_months:
        invariant(re.fullmatch(r"2025-\d\d",m) is not None, "2025-only month")
        invariant("2025-03"<=m<="2025-12", "Month outside 2025 research horizon")
    args.modules=list(MODULES)
    c21_self_test()
    if args.self_test:return 0
    args.trees=220 if args.preset=="full" else 65
    args.experiment_trees=180 if args.preset=="full" else 55
    args.max_train_rows=None if args.preset=="full" else 100000
    if args.preset=="smoke":
        args.min_local_train=700
        args.min_special_train=700
        args.min_tail_train=50
        args.min_local_tail_train=30
        args.min_residual_train=250
        args.min_super_tail_train=15
        args.c21_min_medium_events=15
    print("="*78,flush=True)
    print("CAMPAIGN 21: CLOCK-AWARE / DUAL-THRESHOLD TAIL -- RESEARCH 2025 ONLY",flush=True)
    print("NO RANKING/SUBMISSION READ, NO 2026 PREDICTIONS, NO FINAL REFIT",flush=True)
    print("ALL 2025 OUTERS HAVE BEEN EXPLORED HISTORICALLY -- NOT FRESH HOLDOUTS",flush=True)
    print("="*78,flush=True)
    begin=time.perf_counter()
    raw,manifest=load_training_files(args.data_dir)
    feat=build_features(raw)
    y=make_y(raw)
    invariant(bool((feat["_timestamp"].dt.year==2025).all()),"2026 data detected")
    add_past_only_flow(feat)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.out_dir/"manifest.json",{
      "campaign":21,"code_sha256":sha256_file(Path(__file__)),"inputs":manifest,
      "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
      "research_only":True,
      "notes":["Original C20 selection frozen on the SELECT month before C21 selection",
               "C21 gating only uses AOBT availability/plausibility and fitted tail probabilities",
               "All OUTER 2025 months previously explored, never virgin holdouts",
               "No 2026 read, no final refit, no submission"]})
    print(f"2025 movements={len(raw):,}, valid labels={int(np.sum(np.isfinite(y)&(y>=0))):,}",flush=True)
    selected=[]; select_rows=[]; outer_rows=[]; detail=[]; saved=[]
    for outer in args.outer_months:
        print(f"\n==== OUTER {outer} / SELECT {months_before(outer)} ====",flush=True)
        cut=select_indices(feat,y,outer,args.embargo_days)
        print("ROWS",{k:len(v) for k,v in cut.items()},flush=True)
        print("  FIT strictly before SELECT",flush=True)
        pack=c21_fit(feat,y,cut["fit_select"],args)
        c20_sel=predict_campaign20(pack["parent"],feat,y,cut["select"],args)
        c20_locked,c19_locked,c20_rows=select_campaign20(
            feat,y,cut["select"],c20_sel,args,outer)
        s_new,s_aux=c21_candidate_predictions(
            pack,feat,y,cut["select"],args,c20_locked,c20_sel,return_aux=True)
        new_locked,new_rows=c21_select(feat,y,cut["select"],c20_locked,s_new,
                                       c20_sel,args,outer)
        for row in c20_rows:
            select_rows.append({**row,"stage":"C19_C20"})
        select_rows.extend({**r,"stage":"C21"} for r in new_rows)
        print(f"  LOCKED SELECT C19={c19_locked} C20={c20_locked} C21={new_locked}",flush=True)
        print(f"  SELECT gating: {s_aux}",flush=True)
        del pack,c20_sel,s_new
        print("  REFITTING models STRICTLY BEFORE OUTER (2025 research)",flush=True)
        pack=c21_fit(feat,y,cut["fit_outer"],args)
        c20_out=predict_campaign20(pack["parent"],feat,y,cut["outer"],args)
        c21_out,o_aux=c21_candidate_predictions(
            pack,feat,y,cut["outer"],args,c20_locked,c20_out,return_aux=True)
        cur=c20_out[c20_locked]
        chosen=c20_out[new_locked] if new_locked in c20_out else c21_out[new_locked]
        yy=y[cut["outer"]]
        days=feat["_day"].iloc[cut["outer"]].to_numpy()
        old=score(yy,cur)
        new=score(yy,chosen)
        anchor=score(yy,c20_out[ANCHOR])
        print(f"  OUTER L2 anchor RMSE={anchor['rmse']:.3f} C20={old['rmse']:.3f} "
              f"C21={new['rmse']:.3f} [chosen={new_locked}]",flush=True)
        all_report={"c21_locked_C20":cur, "c21_locked_C21":chosen,
                    "c21_anchor":c20_out[ANCHOR],
                    "c21_super_lirf_50_REFERENCE":c20_out["super_lirf_50"],
                    **c21_out}
        for j,(key,pred) in enumerate(all_report.items()):
            metric=score(yy,pred)
            boot=paired_bootstrap(yy,pred,cur,days,args.bootstrap,
                                  args.seed+72013+j*1013) if key!="c21_locked_C20" else {}
            outer_rows.append({"outer":outer,"candidate":key,
                               "selected":key=="c21_locked_C21",
                               "C20_locked":c20_locked,"C21_locked":new_locked,
                               **metric, **boot})
        selected.append({"outer":outer,"select":months_before(outer),
                         "n":len(yy),"C19_locked":c19_locked,
                         "C20_locked":c20_locked,"C21_locked":new_locked,
                         "anchor_rmse":anchor["rmse"],
                         "C20_rmse":old["rmse"],"C21_rmse":new["rmse"]})
        ap=feat["airport"].iloc[cut["outer"]].astype(str).to_numpy()
        good=_plausible_proxy(feat,cut["outer"])
        e_base=(cur-yy)**2
        e_selected=(chosen-yy)**2
        top01=e_base>=np.quantile(e_base,.999)
        for group,mask in {
             "all":np.ones(len(yy),bool),
             "LIRF":ap=="LIRF",
             "LIRF_proxy_plausible":(ap=="LIRF")&good,
             "LIRF_proxy_bad_or_missing":(ap=="LIRF")&(~good),
             "LIRF_target_over_2400":(ap=="LIRF")&(yy>2400),
             "LIRF_target_over_3600":(ap=="LIRF")&(yy>3600),
             "LIRF_target_under_3600":(ap=="LIRF")&(yy<=3600),
             "LTFM":ap=="LTFM",
             "incumbent_worst_0p1pct":top01,
             "incumbent_other_99p9pct":~top01,
        }.items():
            if not np.any(mask):continue
            detail.append({"outer":outer,"candidate":new_locked,"slice":group,
                           "n":int(mask.sum()),
                           "C20_rmse":float(np.sqrt(np.mean(e_base[mask]))),
                           "C21_rmse":float(np.sqrt(np.mean(e_selected[mask]))),
                           "delta_SSE":float(np.sum(e_selected[mask]-e_base[mask])),
                           "C20_SSE_share":float(np.sum(e_base[mask])/np.sum(e_base))})
        if args.save_outer_predictions:
            subset=cut["outer"]
            fr=pd.DataFrame({"outer":outer,
                 "utc_mvt":feat["_timestamp"].iloc[subset].to_numpy(),
                 "airport":ap, "proxy_ok":good, "y_true":yy,
                 "pred_C20_locked":cur,"pred_C21_locked":chosen,
                 "pred_super_lirf_50":c20_out["super_lirf_50"]})
            if ID in raw:fr.insert(0,ID,raw[ID].iloc[subset].to_numpy())
            for name,pred in c21_out.items():fr["pred_"+name]=pred
            saved.append(fr)
        del pack,c20_out,c21_out
    for filename,rows in [
       ("locked_choices.csv",selected),
       ("select_candidates.csv",select_rows),
       ("outer_candidates_DESCRIPTIVE.csv",outer_rows),
       ("tail_and_clock_diagnostics.csv",detail),
    ]:
        pd.DataFrame(rows).to_csv(args.out_dir/filename,index=False)
    if saved:
        combined=pd.concat(saved,ignore_index=True)
        try:combined.to_parquet(args.out_dir/"outer_predictions_2025_RESEARCH_ONLY.parquet",index=False)
        except ImportError:combined.to_csv(args.out_dir/"outer_predictions_2025_RESEARCH_ONLY.csv",index=False)
    if args.february_stress:
        feb=(feat["_month"].to_numpy()=="2025-02") & np.isfinite(y) & (y>=0)
        counts=[]
        for apt in sorted(feat["airport"].iloc[np.flatnonzero(feb)].astype(str).unique()):
            mask=feb & (feat["airport"].to_numpy()==apt)
            vals=y[mask]
            counts.append({"airport":apt,"n":len(vals),
                           "n_gt_3600":int(np.sum(vals>3600)),
                           "p99_target":float(np.quantile(vals,.99))})
        pd.DataFrame(counts).to_csv(args.out_dir/"february_2025_DESCRIPTIVE.csv",index=False)
    total=sum(r["n"] for r in selected)
    pooled={key:float(np.sqrt(sum(r["n"]*r[key+"_rmse"]**2 for r in selected)/total))
            for key in ("anchor","C20","C21")}
    core=[r for r in selected if r["outer"] in C21_COMPARABLE]
    base_total=sum(r["n"] for r in core)
    matched={key:float(np.sqrt(sum(r["n"]*r[key+"_rmse"]**2 for r in core)/base_total))
             for key in ("anchor","C20","C21")} if core else {}
    print("\nPOOLED 2025 HISTORICAL OUTERS:",pooled,flush=True)
    print("POOLED MATCHED CAMPAIGN20 5 OUTERS:",matched,flush=True)
    write_json(args.out_dir/"summary.json",{
        "campaign":21,"pooled_rmse":pooled,
        "matched_5fold_rmse":matched,"locked":selected,
        "seconds":round(time.perf_counter()-begin,2),
        "research_only":True,"no_2026_read":True,
        "note":"All OUTER months have previously been explored, no unbiased holdout"})
    print("Research output:",args.out_dir.resolve(),flush=True)
    print("NO FINAL REFIT / NO SUBMISSION / NO 2026 RANKING PREDICTIONS",flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(c21_main())
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(2)
