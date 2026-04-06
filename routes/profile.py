import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from models.schemas import TierSubmitRequest
from services.clustering_service import (
    get_cluster_profiles,
    submit_tiers,
    get_scatter,
)
from store import store

router = APIRouter()


@router.get("/profile")
def profile():
    """Get per-cluster aggregated metrics."""
    return get_cluster_profiles()


@router.post("/profile/tiers")
def assign_tiers(req: TierSubmitRequest):
    """Submit tier assignments and return scatter plot data."""
    return submit_tiers(req.tier_mapping)


@router.get("/profile/scatter")
def scatter(
    x_axis: str = Query(..., description="X-axis feature"),
    y_axis: str = Query(..., description="Y-axis feature"),
):
    """Return scatter plot for selected axes."""
    return get_scatter(x_axis, y_axis)


@router.get("/profile/download")
def download_profiled():
    """Download the fully profiled dataset (cluster_no + tier columns) as CSV."""
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="No profiled data available.")
    if "tier" not in store.clustered_df.columns:
        raise HTTPException(status_code=400, detail="Tier assignments not submitted yet.")
    buf = io.StringIO()
    store.clustered_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=agent360_profiled.csv"},
    )
