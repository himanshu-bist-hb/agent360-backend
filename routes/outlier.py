from fastapi import APIRouter, Query
from services.outlier_service import detect_outliers

router = APIRouter()


@router.get("/analysis/outliers")
def outliers(
    tier: str = Query(default="All", description="Tier to filter (or 'All')"),
    method: str = Query(default="iqr", description="Detection method: 'iqr' or 'zscore'"),
):
    """Detect statistical outlier agents within a tier."""
    return detect_outliers(tier=tier, method=method)
