from fastapi import APIRouter
from models.schemas import ElbowRequest
from services.clustering_service import run_elbow

router = APIRouter()


@router.post("/elbow")
def elbow_method(req: ElbowRequest):
    """Run elbow method and return inertia plot + optimal K."""
    return run_elbow(req.features)
