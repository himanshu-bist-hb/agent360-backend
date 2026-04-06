import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import ClusterRequest
from services.clustering_service import run_clustering
from store import store

router = APIRouter()


@router.post("/cluster")
def cluster(req: ClusterRequest):
    """Run K-Means clustering with the specified features and K."""
    return run_clustering(req.features, req.k)


@router.get("/cluster/download")
def download_clustered():
    """Download the clustered dataset (with cluster_no column) as CSV."""
    if store.clustered_df is None:
        raise HTTPException(status_code=400, detail="No clustered data available.")
    buf = io.StringIO()
    store.clustered_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=agent360_clustered.csv"},
    )
