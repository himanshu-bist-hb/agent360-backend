import numpy as np
import pandas as pd
from typing import Any


def sanitize_value(val: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, pd.Timestamp):
        return str(val)
    return val


def sanitize_record(record: dict) -> dict:
    """Sanitize all values in a record dict."""
    return {k: sanitize_value(v) for k, v in record.items()}


def compute_quality_score(df: pd.DataFrame) -> tuple[float, str]:
    """
    Compute a data quality score based on null percentages.
    Returns (score 0-100, label).
    """
    if df.empty:
        return 0.0, "Empty"
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    null_pct = (null_cells / total_cells) * 100 if total_cells > 0 else 0
    score = max(0.0, 100.0 - null_pct)
    if score >= 95:
        label = "Excellent"
    elif score >= 85:
        label = "Good"
    elif score >= 70:
        label = "Fair"
    else:
        label = "Needs Cleaning"
    return round(score, 2), label


def find_optimal_k(k_values: list, inertias: list) -> int:
    """
    Find optimal K using the elbow point (maximum perpendicular distance).
    Uses the kneedle algorithm concept.
    """
    k_arr = np.array(k_values, dtype=float)
    inertia_arr = np.array(inertias, dtype=float)

    if len(k_arr) < 3:
        return int(k_values[0])

    k_range = k_arr.max() - k_arr.min()
    i_range = inertia_arr.max() - inertia_arr.min()

    if k_range == 0 or i_range == 0:
        return int(k_values[0])

    k_norm = (k_arr - k_arr.min()) / k_range
    inertia_norm = (inertia_arr - inertia_arr.min()) / i_range

    # Direction vector of the line from first to last point
    dx = k_norm[-1] - k_norm[0]
    dy = inertia_norm[-1] - inertia_norm[0]
    line_len = np.sqrt(dx ** 2 + dy ** 2)

    if line_len == 0:
        return int(k_values[0])

    distances = []
    for i in range(len(k_norm)):
        px = k_norm[i] - k_norm[0]
        py = inertia_norm[i] - inertia_norm[0]
        cross = abs(dx * py - dy * px)
        distances.append(cross / line_len)

    optimal_idx = int(np.argmax(distances))
    return int(k_values[optimal_idx])
