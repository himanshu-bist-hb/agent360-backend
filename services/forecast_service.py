import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from fastapi import HTTPException
from store import store

TIER_PALETTE = {
    "Bronze": "#CD7F32",
    "Silver": "#A8A9AD",
    "Gold": "#FFD700",
    "Diamond": "#B9F2FF",
    "Platinum": "#E5E4E2",
}


def get_impact_ranking() -> dict:
    """
    Rank numeric features by their absolute Pearson correlation with gwp_total.
    Returns a Plotly bar chart + ranked list.
    """
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="No clustered data. Complete Step 3 first.")

    df = store.clustered_df
    if "gwp_total" not in df.columns:
        raise HTTPException(status_code=400, detail="gwp_total column not found in dataset.")

    exclude = {"gwp_total", "gwp_new_business", "gwp_renewals",
               "commission_earned", "cluster_no", "avg_premium_per_policy"}

    numeric_cols = [
        c for c in store.numeric_columns
        if c in df.columns and c not in exclude
    ]

    correlations = []
    for col in numeric_cols:
        if df[col].std() > 0:
            corr = float(df[col].corr(df["gwp_total"]))
            if not np.isnan(corr):
                correlations.append({"feature": col, "correlation": round(corr, 4)})

    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    top = correlations[:12]

    colors = ["#22c55e" if c["correlation"] > 0 else "#ef4444" for c in top]

    fig = go.Figure(go.Bar(
        x=[c["correlation"] for c in top],
        y=[c["feature"] for c in top],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Correlation with GWP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Feature Correlation with GWP Total (Top 12)", font=dict(size=16)),
        xaxis=dict(title="Pearson Correlation", gridcolor="#F1F5F9", range=[-1, 1]),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=200, r=40, t=60, b=60),
        font=dict(family="Inter, sans-serif", color="#1E293B"),
        height=420,
    )

    return {"ranking": correlations, "plot_json": fig.to_dict()}


def run_shock(feature: str, change_pct: float, scope: str) -> dict:
    """
    Estimate GWP impact if `feature` changes by `change_pct` %.
    scope: 'all' | tier name | state name
    """
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="No clustered data. Complete Step 3 first.")

    df = store.clustered_df.copy()
    if "gwp_total" not in df.columns:
        raise HTTPException(status_code=400, detail="gwp_total not found in dataset.")
    if feature not in df.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found.")

    # ── Filter by scope ───────────────────────────────────────────────────────
    filtered = df
    if scope != "all":
        if "tier" in df.columns and scope in df["tier"].unique():
            filtered = df[df["tier"] == scope]
        elif store.agent_id_col and "state" in df.columns and scope in df["state"].unique():
            filtered = df[df["state"] == scope]

    if filtered.empty:
        raise HTTPException(status_code=400, detail=f"No agents found for scope '{scope}'.")

    # ── Linear regression: feature → gwp_total ───────────────────────────────
    valid = filtered[[feature, "gwp_total"]].dropna()
    X = valid[[feature]].values
    y = valid["gwp_total"].values

    reg = LinearRegression()
    reg.fit(X, y)
    coef = float(reg.coef_[0])
    r2   = float(reg.score(X, y))

    # ── Shock calculation ─────────────────────────────────────────────────────
    delta_feature     = filtered[feature] * (change_pct / 100.0)
    estimated_gwp_delta = (delta_feature * coef).sum()
    current_gwp_total = float(filtered["gwp_total"].sum())
    projected_gwp     = current_gwp_total + estimated_gwp_delta
    delta_pct         = (estimated_gwp_delta / current_gwp_total * 100) if current_gwp_total else 0

    # ── Per-tier breakdown (if tier column exists) ────────────────────────────
    tier_breakdown = []
    if "tier" in filtered.columns:
        for tier, grp in filtered.groupby("tier"):
            d_feat   = grp[feature] * (change_pct / 100.0)
            d_gwp    = float((d_feat * coef).sum())
            cur_gwp  = float(grp["gwp_total"].sum())
            tier_breakdown.append({
                "tier": str(tier),
                "current_gwp": cur_gwp,
                "estimated_delta": d_gwp,
                "projected_gwp": cur_gwp + d_gwp,
                "delta_pct": round((d_gwp / cur_gwp * 100) if cur_gwp else 0, 2),
            })

    # ── Build waterfall / bar chart ───────────────────────────────────────────
    plot_json = _build_shock_chart(tier_breakdown, feature, change_pct)

    return {
        "feature": feature,
        "change_pct": change_pct,
        "scope": scope,
        "agents_in_scope": int(len(filtered)),
        "regression_r2": round(r2, 4),
        "regression_coef": round(coef, 4),
        "current_gwp_total": round(current_gwp_total, 2),
        "estimated_gwp_delta": round(float(estimated_gwp_delta), 2),
        "projected_gwp_total": round(projected_gwp, 2),
        "delta_pct": round(float(delta_pct), 2),
        "tier_breakdown": tier_breakdown,
        "plot_json": plot_json,
    }


def _build_shock_chart(tier_breakdown: list, feature: str, change_pct: float) -> dict:
    if not tier_breakdown:
        return {}

    tiers    = [t["tier"] for t in tier_breakdown]
    cur_gwp  = [t["current_gwp"] / 1_000_000 for t in tier_breakdown]
    proj_gwp = [t["projected_gwp"] / 1_000_000 for t in tier_breakdown]
    colors   = [TIER_PALETTE.get(t, "#4F46E5") for t in tiers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current GWP",
        x=tiers, y=cur_gwp,
        marker_color=colors, opacity=0.5,
        hovertemplate="<b>%{x}</b><br>Current GWP: $%{y:.2f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Projected GWP",
        x=tiers, y=proj_gwp,
        marker_color=colors, opacity=1.0,
        hovertemplate="<b>%{x}</b><br>Projected GWP: $%{y:.2f}M<extra></extra>",
    ))
    sign = "+" if change_pct >= 0 else ""
    fig.update_layout(
        title=dict(
            text=f"GWP Impact: {feature} {sign}{change_pct}% shock by Tier",
            font=dict(size=16),
        ),
        barmode="group",
        xaxis=dict(title="Tier", gridcolor="#F1F5F9"),
        yaxis=dict(title="GWP ($ Millions)", gridcolor="#F1F5F9"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=70, b=60),
        font=dict(family="Inter, sans-serif", color="#1E293B"),
    )
    return fig.to_dict()
