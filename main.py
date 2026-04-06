from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import upload, elbow, cluster, profile, analysis, outlier, forecast

app = FastAPI(
    title="Agent360 API",
    description="Insurance Agent Performance Clustering Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router,   prefix="/api", tags=["Upload"])
app.include_router(elbow.router,    prefix="/api", tags=["Elbow"])
app.include_router(cluster.router,  prefix="/api", tags=["Cluster"])
app.include_router(profile.router,  prefix="/api", tags=["Profile"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(outlier.router,  prefix="/api", tags=["Outliers"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])


@app.get("/")
def root():
    return {"status": "ok", "app": "Agent360 API", "version": "1.0.0"}
