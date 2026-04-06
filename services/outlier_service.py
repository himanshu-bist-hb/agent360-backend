import numpy as np
import pandas as pd
from fastapi import HTTPException
from store import store
from utils.helpers import sanitize_record


def detect_outliers(tier: str = None, method: str = "iqr") -> dict:
    """
    Detect outlier agents using IQR or Z-score method.
    Optionally filter by tier.
    """
    if store.clustered_df is None or "tier" not in store.clustered_df.columns:
        raise HTTPException(status_code=400, detail="Tier assignments not found. Complete Step 4 first.")

    df = store.clustered_df.copy()
    if tier and tier != "All":
        df = df[df["tier"] == tier]
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No agents found for tier '{tier}'.")

    numeric_cols = [
        c for c in store.numeric_columns
        if c in df.columns and c not in ("cluster_no",)
    ]

    outlier_flags = {col: _flag_column(df[col], method) for col in numeric_cols}

    # Build result: agents that are outliers in at least one column
    any_outlier = np.zeros(len(df), dtype=bool)
    for flags in outlier_flags.values():
        any_outlier |= flags

    outlier_df = df[any_outlier].copy()

    # Attach outlier columns list to each row
    records = []
    for idx, row in outlier_df.iterrows():
        flagged_cols = [
            col for col in numeric_cols
            if outlier_flags[col][df.index.get_loc(idx)]
        ]
        rec = sanitize_record(row.to_dict())
        rec["outlier_metrics"] = flagged_cols
        rec["outlier_count"] = len(flagged_cols)
        records.append(rec)

    # Sort by outlier count descending
    records.sort(key=lambda r: r["outlier_count"], reverse=True)

    return {
        "total_agents_scanned": int(len(df)),
        "outlier_count": int(len(records)),
        "method": method,
        "tier_filter": tier or "All",
        "agents": records[:100],  # cap at 100 for response size
    }


def _flag_column(series: pd.Series, method: str) -> np.ndarray:
    """Return boolean array marking outliers in a series."""
    s = pd.to_numeric(series, errors="coerce")
    flags = np.zeros(len(s), dtype=bool)
    valid = s.notna()

    if method == "zscore":
        mean, std = s[valid].mean(), s[valid].std()
        if std > 0:
            z = np.abs((s - mean) / std)
            flags = (z > 2.8).values
    else:  # IQR
        q1, q3 = s[valid].quantile(0.25), s[valid].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags = ((s < lower) | (s > upper)).values

    return flags
