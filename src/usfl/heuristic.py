# src/usfl/heuristic.py
#
# Heuristic dynamic dual-cut selector for USFL.
# - Loads offline tables (static_costs + optional proxies + scales)
# - Given current net_state/comp_state, estimates dynamic comm/edge-comp costs
# - Selects (cut1, cut2) by minimizing a proxy objective J_hat
#
# Notes:
# 1) This module is intentionally robust to missing proxy columns. If comm/edge-comp
#    proxies are absent, it will gracefully fall back to static-only selection.
# 2) It does NOT assume any LinUCB "context" usage. If you later add a proxy table
#    (e.g., comm_bytes, comp_time_server), it will pick it up automatically.
#
# Expected CSV conventions (minimum):
# - static_costs.csv columns: cut1, cut2, acc_cost, privacy_cost, comp_cost_client, static_cost
# - scales.csv rows: {"metric": "comm_time_scale", "value": ...}, {"metric": "comp_time_scale", "value": ...}
#
# Optional proxy columns (either in feature_csv or static_costs.csv):
# - comm_bytes: bytes transferred per decision unit (e.g., per epoch) under this cut
# - comm_time_base: comm time under a known baseline network (seconds)
# - msg_count: number of RTT-bearing messages per decision unit (default: 1)
# - comp_time_server: server-side compute time proxy under this cut (seconds)
# - t_middle: alternative name for comp_time_server

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class OfflineTables:
    static_df: pd.DataFrame
    proxy_df: Optional[pd.DataFrame]
    scales: Dict[str, float]


def _ns_get(obj: Any, dotted: str, default: Any = None) -> Any:
    """Get dotted attribute from a SimpleNamespace-like config."""
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return default
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_offline_tables(cfg: Any) -> OfflineTables:
    """
    Load offline tables from cfg.experiment.output_dir using cfg.logging.* filenames.

    It will look for:
      - static_cost_csv (required)
      - feature_csv (optional; used as proxy table if present)
      - scale_csv (optional but recommended)
    """
    out_dir = _ns_get(cfg, "experiment.output_dir", ".")
    logging = _ns_get(cfg, "logging", None)

    if logging is None:
        raise ValueError("cfg.logging is required to locate offline CSVs.")

    static_name = _ns_get(cfg, "logging.static_cost_csv", None)
    feature_name = _ns_get(cfg, "logging.feature_csv", None)
    scale_name = _ns_get(cfg, "logging.scale_csv", None)

    static_path = os.path.join(out_dir, static_name) if static_name else ""
    feature_path = os.path.join(out_dir, feature_name) if feature_name else ""
    scale_path = os.path.join(out_dir, scale_name) if scale_name else ""

    static_df = _read_csv_if_exists(static_path)
    if static_df is None:
        raise FileNotFoundError(
            f"static_costs CSV not found at: {static_path}. "
            f"Check cfg.experiment.output_dir and cfg.logging.static_cost_csv."
        )

    # Optional proxy table: you can store comm_bytes / comp_time_server / etc. here.
    proxy_df = _read_csv_if_exists(feature_path)

    # scales
    scales_df = _read_csv_if_exists(scale_path)
    scales: Dict[str, float] = {}
    if scales_df is not None and len(scales_df) > 0:
        # expected columns: metric,value
        if "metric" in scales_df.columns and "value" in scales_df.columns:
            for _, row in scales_df.iterrows():
                try:
                    scales[str(row["metric"])] = float(row["value"])
                except Exception:
                    continue

    # Basic validation
    for col in ("cut1", "cut2"):
        if col not in static_df.columns:
            raise ValueError(f"static_costs CSV must include column '{col}'.")

    # Ensure int cuts for safe joins
    static_df["cut1"] = static_df["cut1"].astype(int)
    static_df["cut2"] = static_df["cut2"].astype(int)

    if proxy_df is not None and "cut1" in proxy_df.columns and "cut2" in proxy_df.columns:
        proxy_df = proxy_df.copy()
        proxy_df["cut1"] = proxy_df["cut1"].astype(int)
        proxy_df["cut2"] = proxy_df["cut2"].astype(int)
    else:
        proxy_df = None  # do not attempt to merge if it lacks keys

    return OfflineTables(static_df=static_df, proxy_df=proxy_df, scales=scales)


def _bps_from_mbps(mbps: float) -> float:
    # 1 Mbps = 1e6 bits per second
    return max(float(mbps), 1e-12) * 1e6


def estimate_comm_time_seconds(row: pd.Series, net_state: Dict[str, float]) -> float:
    """
    Estimate comm time (seconds) for a given cut under current network state.

    Supports two modes:
    - If 'comm_bytes' exists: comm_time = comm_bytes / bw_eff + msg_count * rtt
    - Else if 'comm_time_base' exists: comm_time = comm_time_base * (bw_base / bw_eff) + msg_count * rtt_adjust
    - Else: 0.0 (fallback)
    """
    bw_up = float(net_state.get("bw_up_mbps", net_state.get("bw_mbps", 0.0)) or 0.0)
    bw_down = float(net_state.get("bw_down_mbps", net_state.get("bw_mbps", 0.0)) or 0.0)
    rtt_ms = float(net_state.get("rtt_ms", 0.0) or 0.0)

    # Use a conservative symmetric effective bandwidth if both directions provided
    if bw_up > 0.0 and bw_down > 0.0:
        bw_eff_mbps = min(bw_up, bw_down)
    else:
        bw_eff_mbps = max(bw_up, bw_down)

    bw_eff_bps = _bps_from_mbps(bw_eff_mbps)
    rtt_s = max(rtt_ms, 0.0) / 1000.0

    msg_count = 1.0
    if "msg_count" in row and pd.notna(row["msg_count"]):
        try:
            msg_count = float(row["msg_count"])
        except Exception:
            msg_count = 1.0

    # Primary: comm_bytes
    if "comm_bytes" in row and pd.notna(row["comm_bytes"]):
        comm_bytes = float(row["comm_bytes"])
        # bytes -> bits
        tx_bits = max(comm_bytes, 0.0) * 8.0
        return (tx_bits / bw_eff_bps) + (msg_count * rtt_s)

    # Secondary: comm_time_base (requires bw_base_mbps if provided)
    if "comm_time_base" in row and pd.notna(row["comm_time_base"]):
        comm_time_base = float(row["comm_time_base"])
        bw_base = float(net_state.get("bw_base_mbps", net_state.get("bw_mbps_base", bw_eff_mbps)) or bw_eff_mbps)
        bw_base = max(bw_base, 1e-12)
        scale = bw_base / max(bw_eff_mbps, 1e-12)
        return (comm_time_base * scale) + (msg_count * rtt_s)

    return 0.0


def estimate_edge_comp_time_seconds(row: pd.Series, comp_state: Dict[str, float]) -> float:
    """
    Estimate server-side compute time (seconds) for a given cut.

    If 'comp_time_server' exists, use it; else try 't_middle'. Then apply
    kappa_server scaling if provided.

    Convention here (default): larger kappa_server means faster server, so:
      effective_time = raw_time / kappa_server
    """
    raw = None
    if "comp_time_server" in row and pd.notna(row["comp_time_server"]):
        raw = float(row["comp_time_server"])
    elif "t_middle" in row and pd.notna(row["t_middle"]):
        raw = float(row["t_middle"])

    if raw is None:
        return 0.0

    kappa_server = float(comp_state.get("kappa_server", 1.0) or 1.0)
    kappa_server = max(kappa_server, 1e-12)
    return max(raw, 0.0) / kappa_server


def _norm01_by_scale(x: float, scale: float) -> float:
    """
    Simple scale-based normalization to [0,1], robust to unknown maxima.
    Uses clip(x/scale, 0, 1).
    """
    s = max(float(scale), 1e-12)
    v = x / s
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _get_weights(cfg: Any) -> Dict[str, float]:
    w = _ns_get(cfg, "objective.weights", None)
    if w is None:
        return {"acc_cost": 1.0, "privacy": 1.0, "comm": 1.0, "comp": 1.0}
    # tolerate either dict or namespace
    return {
        "acc_cost": float(_ns_get(w, "acc_cost", 1.0)),
        "privacy": float(_ns_get(w, "privacy", 1.0)),
        "comm": float(_ns_get(w, "comm", 1.0)),
        "comp": float(_ns_get(w, "comp", 1.0)),
    }


def _merge_tables(tables: OfflineTables) -> pd.DataFrame:
    df = tables.static_df.copy()
    if tables.proxy_df is not None:
        # left join to keep all candidates
        df = df.merge(tables.proxy_df, on=["cut1", "cut2"], how="left", suffixes=("", "_proxy"))
    return df


def select_cut_heuristic(
    cfg: Any,
    tables: OfflineTables,
    net_state: Dict[str, float],
    comp_state: Dict[str, float],
    prev_cut: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Select (cut1, cut2) by minimizing J_hat.

    J_hat is constructed as:
      J_hat = w_acc * acc_cost
            + w_priv * privacy_cost
            + w_comp * (comp_cost_client + comp_edge_norm)
            + w_comm * comm_norm

    where acc_cost/privacy_cost are assumed already in [0,1],
    comp_cost_client is assumed already normalized (as in your offline result),
    and comm/edge-comp are normalized using scales.csv:
      comm_time_scale, comp_time_scale

    Hysteresis (optional):
      cfg.usfl.heuristic.hysteresis_delta (default 0.0)
      If prev_cut is provided, switch only if improvement > delta.
    """
    df = _merge_tables(tables)

    # Required columns (static)
    for c in ("acc_cost", "privacy_cost"):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in static_costs table.")

    # Optional: comp_cost_client
    if "comp_cost_client" not in df.columns:
        df["comp_cost_client"] = 0.0

    # scales
    comm_scale = float(tables.scales.get("comm_time_scale", 0.0) or 0.0)
    comp_scale = float(tables.scales.get("comp_time_scale", 0.0) or 0.0)

    # If no scale provided, fall back to data-driven scale to avoid div-by-zero.
    # (This keeps heuristic usable even if scales.csv is not generated yet.)
    if comm_scale <= 0.0:
        comm_scale = 1.0
    if comp_scale <= 0.0:
        comp_scale = 1.0

    weights = _get_weights(cfg)

    # Compute J_hat per candidate
    J_vals = []
    for _, row in df.iterrows():
        comm_s = estimate_comm_time_seconds(row, net_state)
        comp_edge_s = estimate_edge_comp_time_seconds(row, comp_state)

        comm_norm = _norm01_by_scale(comm_s, comm_scale)
        comp_edge_norm = _norm01_by_scale(comp_edge_s, comp_scale)

        acc_cost = float(row["acc_cost"])
        priv_cost = float(row["privacy_cost"])
        comp_client = float(row.get("comp_cost_client", 0.0) or 0.0)

        J_hat = (
            weights["acc_cost"] * acc_cost
            + weights["privacy"] * priv_cost
            + weights["comm"] * comm_norm
            + weights["comp"] * (comp_client + comp_edge_norm)
        )

        J_vals.append(J_hat)

    df = df.copy()
    df["J_hat"] = J_vals

    # Best candidate by proxy
    best_idx = int(df["J_hat"].idxmin())
    best_row = df.loc[best_idx]
    best_cut = (int(best_row["cut1"]), int(best_row["cut2"]))
    best_J = float(best_row["J_hat"])

    # Optional hysteresis to avoid oscillation
    delta = float(_ns_get(cfg, "usfl.heuristic.hysteresis_delta", 0.0) or 0.0)
    if prev_cut is not None and delta > 0.0:
        prev_df = df[(df["cut1"] == prev_cut[0]) & (df["cut2"] == prev_cut[1])]
        if len(prev_df) > 0:
            prev_J = float(prev_df["J_hat"].iloc[0])
            # Switch only if improvement is significant
            if (prev_J - best_J) <= delta:
                return prev_cut

    return best_cut
