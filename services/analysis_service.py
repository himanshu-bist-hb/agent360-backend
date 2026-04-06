import plotly.graph_objects as go
from fastapi import HTTPException

from store import store
from utils.helpers import sanitize_record

TIER_PALETTE = {
    "Bronze": "#CD7F32",
    "Silver": "#A8A9AD",
    "Gold": "#FFD700",
    "Diamond": "#B9F2FF",
    "Platinum": "#E5E4E2",
}
TIER_ORDER = ["Bronze", "Silver", "Gold", "Diamond", "Platinum"]


def _apply_hierarchy_filter(df, state: str = None, district: str = None):
    """Filter dataframe by state and/or district if provided."""
    if state and state != "All" and "state" in df.columns:
        df = df[df["state"] == state]
    if district and district != "All" and "district" in df.columns:
        df = df[df["district"] == district]
    return df


def get_metrics_analysis(state: str = None, district: str = None) -> dict:
    """Compare key numeric metrics across tiers, optionally filtered by geography."""
    if store.clustered_df is None or "tier" not in store.clustered_df.columns:
        raise HTTPException(status_code=400, detail="Tier assignments not found. Complete cluster profiling first.")

    df = _apply_hierarchy_filter(store.clustered_df.copy(), state, district)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data for the selected geography filter.")

    numeric_cols = [c for c in store.numeric_columns if c in df.columns and c not in ("cluster_no",)]
    present_tiers = [t for t in TIER_ORDER if t in df["tier"].unique()]
    tier_group = df.groupby("tier")[numeric_cols].mean()

    display_cols = numeric_cols[:8]

    fig = go.Figure()
    for col in display_cols:
        values = [float(tier_group.loc[t, col]) if t in tier_group.index else 0 for t in present_tiers]
        fig.add_trace(go.Bar(
            name=col, x=present_tiers, y=values,
            hovertemplate=f"<b>{col}</b><br>Tier: %{{x}}<br>Avg: %{{y:.2f}}<extra></extra>",
        ))

    geo_suffix = ""
    if state and state != "All":
        geo_suffix = f" — {district if district and district != 'All' else state}"

    fig.update_layout(
        title=dict(text=f"Average Metrics by Performance Tier{geo_suffix}", font=dict(size=16)),
        barmode="group",
        xaxis=dict(title="Tier", gridcolor="#F1F5F9"),
        yaxis=dict(title="Average Value", gridcolor="#F1F5F9"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="Metric", orientation="v", x=1.02),
        margin=dict(l=60, r=180, t=60, b=60),
        font=dict(family="Inter, sans-serif", color="#1E293B"),
        height=500,
    )

    return {"plot_json": fig.to_dict(), "tiers": present_tiers, "numeric_columns": numeric_cols}


def get_top_agents(tier: str, metric: str, state: str = None, district: str = None, metric2: str = None) -> dict:
    """Top 10 agents in the given tier ranked by primary metric and optional secondary metric."""
    if store.clustered_df is None or "tier" not in store.clustered_df.columns:
        raise HTTPException(status_code=400, detail="Tier assignments not found. Complete cluster profiling first.")

    df = _apply_hierarchy_filter(store.clustered_df.copy(), state, district)

    if tier not in df["tier"].unique():
        raise HTTPException(status_code=400, detail=f"Tier '{tier}' not found in filtered data.")
    if metric not in df.columns:
        raise HTTPException(status_code=400, detail=f"Metric '{metric}' not found in data.")

    filtered = df[df["tier"] == tier].copy()

    # Sort by primary metric, then secondary metric if provided
    if metric2 and metric2 in filtered.columns:
        top10 = filtered.sort_values([metric, metric2], ascending=[False, False]).head(10)
    else:
        top10 = filtered.nlargest(10, metric)
        metric2 = None  # ensure None if not valid

    key_cols = [store.agent_id_col, metric]
    if metric2:
        key_cols.append(metric2)
    key_cols += ["cluster_no", "tier"]
    if "state" in top10.columns:
        key_cols.append("state")
    if "district" in top10.columns:
        key_cols.append("district")
    extra_cols = [c for c in store.numeric_columns if c not in key_cols and c in top10.columns][:3]
    display_cols = [c for c in key_cols + extra_cols if c in top10.columns]

    records = top10[display_cols].fillna("").to_dict(orient="records")
    return {"agents": [sanitize_record(r) for r in records], "tier": tier, "metric": metric, "metric2": metric2}


def get_hierarchy_stats(state: str = None, district: str = None) -> dict:
    """Return tier distribution and GWP stats for the selected geography."""
    if store.clustered_df is None or "tier" not in store.clustered_df.columns:
        raise HTTPException(status_code=400, detail="Tier assignments not found.")

    df = _apply_hierarchy_filter(store.clustered_df.copy(), state, district)
    if df.empty:
        return {"tier_counts": {}, "total_agents": 0, "total_gwp": 0}

    tier_counts = df["tier"].value_counts().to_dict()
    total_gwp = float(df["gwp_total"].sum()) if "gwp_total" in df.columns else 0

    return {
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "total_agents": int(len(df)),
        "total_gwp": round(total_gwp, 2),
        "state": state or "All",
        "district": district or "All",
    }
