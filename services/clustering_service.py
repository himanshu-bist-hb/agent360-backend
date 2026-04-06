import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from fastapi import HTTPException

from store import store
from utils.helpers import find_optimal_k, sanitize_record

TIER_PALETTE = {
    "Bronze": "#CD7F32",
    "Silver": "#A8A9AD",
    "Gold": "#FFD700",
    "Diamond": "#B9F2FF",
    "Platinum": "#E5E4E2",
}

CLUSTER_COLORS = [
    "#4F46E5",
    "#F59E0B",
    "#10B981",
    "#EF4444",
    "#8B5CF6",
    "#EC4899",
    "#06B6D4",
    "#84CC16",
]


def _prepare_matrix(df: pd.DataFrame, features: list) -> np.ndarray:
    """Extract, impute, and scale numeric features for clustering."""
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise HTTPException(status_code=400, detail="No valid numeric features found.")

    X = df[valid_features].copy()
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled


def run_elbow(features: list) -> dict:
    """Compute KMeans inertia for K=2..10 and return elbow plot + optimal K."""
    if store.raw_df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")

    df = store.raw_df
    valid_features = [f for f in features if f in df.columns and f in store.numeric_columns]
    if len(valid_features) < 1:
        raise HTTPException(status_code=400, detail="Select at least one numeric feature.")

    store.selected_features = valid_features

    X = _prepare_matrix(df, valid_features)
    max_k = min(10, len(df) - 1)
    k_range = list(range(2, max_k + 1))

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(float(km.inertia_))

    optimal_k = find_optimal_k(k_range, inertias)
    store.optimal_k = optimal_k

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=inertias,
            mode="lines+markers",
            name="Inertia",
            line=dict(color="#4F46E5", width=2.5),
            marker=dict(size=8, color="#4F46E5", symbol="circle"),
        )
    )
    # Highlight optimal K
    optimal_inertia = inertias[k_range.index(optimal_k)]
    fig.add_trace(
        go.Scatter(
            x=[optimal_k],
            y=[optimal_inertia],
            mode="markers",
            name=f"Optimal K = {optimal_k}",
            marker=dict(size=14, color="#EF4444", symbol="star"),
        )
    )
    fig.add_vline(
        x=optimal_k,
        line_dash="dash",
        line_color="#EF4444",
        opacity=0.6,
        annotation_text=f"K = {optimal_k}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text="Elbow Method — Optimal Cluster Count", font=dict(size=16)),
        xaxis=dict(title="Number of Clusters (K)", dtick=1, gridcolor="#F1F5F9"),
        yaxis=dict(title="Inertia (Within-Cluster Sum of Squares)", gridcolor="#F1F5F9"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=60),
        font=dict(family="Inter, sans-serif", color="#1E293B"),
    )

    return {
        "k_values": k_range,
        "inertias": inertias,
        "optimal_k": optimal_k,
        "plot_json": fig.to_dict(),
    }


def run_clustering(features: list, k: int) -> dict:
    """Run KMeans with the given features and K, attach cluster_no to dataframe."""
    if store.raw_df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")

    df = store.raw_df.copy()
    valid_features = [f for f in features if f in df.columns and f in store.numeric_columns]
    if not valid_features:
        raise HTTPException(status_code=400, detail="No valid numeric features selected.")

    if k < 2:
        raise HTTPException(status_code=400, detail="K must be at least 2.")

    store.selected_features = valid_features
    store.optimal_k = k

    X = _prepare_matrix(df, valid_features)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    df["cluster_no"] = labels.astype(int)
    store.clustered_df = df
    store.cluster_count = k

    # Build display records (all columns)
    records = df.fillna("").to_dict(orient="records")
    sanitized = [sanitize_record(r) for r in records]

    return {
        "data": sanitized,
        "cluster_count": k,
        "columns": list(df.columns),
    }


def get_cluster_profiles() -> dict:
    """Compute per-cluster aggregate statistics."""
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="Clustering has not been run yet.")

    df = store.clustered_df
    numeric_cols = [
        c for c in store.numeric_columns
        if c in df.columns and c != "cluster_no"
    ]

    clusters = []
    for cluster_no in sorted(df["cluster_no"].unique()):
        group = df[df["cluster_no"] == cluster_no]
        metrics = {
            col: round(float(group[col].mean()), 4)
            for col in numeric_cols
            if group[col].notna().any()
        }
        clusters.append(
            {
                "cluster_no": int(cluster_no),
                "count": int(len(group)),
                "metrics": metrics,
            }
        )

    return {"clusters": clusters, "numeric_columns": numeric_cols}


def submit_tiers(tier_mapping: dict) -> dict:
    """
    Assign tiers to clusters and add a 'tier' column to the dataframe.
    Returns scatter plot ready data.
    """
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="Clustering has not been run yet.")

    # tier_mapping: {"0": "Gold", "1": "Bronze", ...}
    int_mapping = {int(k): v for k, v in tier_mapping.items()}
    store.tier_mapping = int_mapping

    df = store.clustered_df.copy()
    df["tier"] = df["cluster_no"].map(int_mapping)
    store.clustered_df = df

    # Build scatter plot data (mean per cluster)
    numeric_cols = [
        c for c in store.numeric_columns
        if c in df.columns and c not in ("cluster_no",)
    ]

    # Axes options: exclude agent_id, cluster_no, tier
    excluded = {store.agent_id_col, "cluster_no", "tier"}
    axes_options = [c for c in numeric_cols if c not in excluded]

    # Build initial scatter data (use first two axes as default)
    x_axis = axes_options[0] if len(axes_options) > 0 else None
    y_axis = axes_options[1] if len(axes_options) > 1 else axes_options[0] if axes_options else None

    scatter_json = _build_scatter(df, x_axis, y_axis, int_mapping) if x_axis and y_axis else {}

    return {
        "available_axes": axes_options,
        "plot_json": scatter_json,
    }


def get_scatter(x_axis: str, y_axis: str) -> dict:
    """Build scatter plot for given axes."""
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="Clustering has not been run yet.")

    df = store.clustered_df
    excluded = {store.agent_id_col, "cluster_no", "tier"}
    if x_axis in excluded or y_axis in excluded:
        raise HTTPException(status_code=400, detail="Cannot use agent_id, cluster_no, or tier as axes.")

    scatter_json = _build_scatter(df, x_axis, y_axis, store.tier_mapping)
    return {"plot_json": scatter_json}


def _build_scatter(df: pd.DataFrame, x_col: str, y_col: str, tier_mapping: dict) -> dict:
    """Build a Plotly scatter figure using all individual data points, grouped by tier."""
    if x_col not in df.columns or y_col not in df.columns:
        return {}

    plot_df = df[[x_col, y_col, "cluster_no", "tier"]].dropna(subset=[x_col, y_col]).copy()

    fig = go.Figure()
    # One trace per tier so the legend shows tiers
    for cluster_no in sorted(plot_df["cluster_no"].unique()):
        group = plot_df[plot_df["cluster_no"] == cluster_no]
        tier = str(tier_mapping.get(int(cluster_no), "Unknown"))
        color = TIER_PALETTE.get(tier, "#4F46E5")
        fig.add_trace(
            go.Scatter(
                x=group[x_col].tolist(),
                y=group[y_col].tolist(),
                mode="markers",
                name=f"{tier} (Cluster {cluster_no})",
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    f"<b>Cluster {cluster_no} — {tier}</b><br>"
                    f"{x_col}: %{{x:.2f}}<br>"
                    f"{y_col}: %{{y:.2f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=f"{x_col} vs {y_col} — All Agents by Tier", font=dict(size=16)),
        xaxis=dict(title=x_col, gridcolor="#F1F5F9"),
        yaxis=dict(title=y_col, gridcolor="#F1F5F9"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="Tier", orientation="v"),
        margin=dict(l=60, r=30, t=60, b=60),
        font=dict(family="Inter, sans-serif", color="#1E293B"),
    )
    return fig.to_dict()
