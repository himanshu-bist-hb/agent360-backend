from fastapi import APIRouter, Query
from services.analysis_service import get_metrics_analysis, get_top_agents, get_hierarchy_stats

router = APIRouter()


@router.get("/analysis/metrics")
def metrics_analysis(
    state: str = Query(default="All"),
    district: str = Query(default="All"),
):
    """Return bar chart comparing metrics across tiers, with optional geo filter."""
    return get_metrics_analysis(state=state, district=district)


@router.get("/analysis/top-agents")
def top_agents(
    tier: str = Query(...),
    metric: str = Query(...),
    state: str = Query(default="All"),
    district: str = Query(default="All"),
    metric2: str = Query(default=None),
):
    """Return top 10 agents in a given tier, ranked by metric (+ optional secondary metric)."""
    return get_top_agents(tier=tier, metric=metric, state=state, district=district, metric2=metric2)


@router.get("/analysis/hierarchy-stats")
def hierarchy_stats(
    state: str = Query(default="All"),
    district: str = Query(default="All"),
):
    """Return tier distribution and GWP totals for a geography."""
    return get_hierarchy_stats(state=state, district=district)
