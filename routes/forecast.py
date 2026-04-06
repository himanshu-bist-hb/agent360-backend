from fastapi import APIRouter
from pydantic import BaseModel
from services.forecast_service import get_impact_ranking, run_shock

router = APIRouter()


class ShockRequest(BaseModel):
    feature: str
    change_pct: float
    scope: str = "all"


@router.get("/forecast/impact-ranking")
def impact_ranking():
    """Return feature correlation ranking with GWP total."""
    return get_impact_ranking()


@router.post("/forecast/shock")
def shock_analysis(req: ShockRequest):
    """Run what-if shock simulation for a feature change."""
    return run_shock(req.feature, req.change_pct, req.scope)
