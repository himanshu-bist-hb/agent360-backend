from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ValidationResult(BaseModel):
    success: bool
    message: str
    rows: Optional[int] = None
    columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    numeric_columns: Optional[List[str]] = None
    null_summary: Optional[Dict[str, int]] = None
    quality_label: Optional[str] = None
    quality_score: Optional[float] = None
    preview: Optional[List[Dict[str, Any]]] = None
    agent_id_col: Optional[str] = None


class ElbowRequest(BaseModel):
    features: List[str]


class ElbowResult(BaseModel):
    k_values: List[int]
    inertias: List[float]
    optimal_k: int
    plot_json: Dict[str, Any]


class ClusterRequest(BaseModel):
    features: List[str]
    k: int


class ClusterResult(BaseModel):
    data: List[Dict[str, Any]]
    cluster_count: int
    columns: List[str]


class ClusterProfile(BaseModel):
    cluster_no: int
    count: int
    metrics: Dict[str, float]


class ProfileResult(BaseModel):
    clusters: List[ClusterProfile]
    numeric_columns: List[str]


class TierSubmitRequest(BaseModel):
    tier_mapping: Dict[str, str]


class ScatterData(BaseModel):
    plot_json: Dict[str, Any]
    available_axes: List[str]


class AnalysisMetricsResult(BaseModel):
    plot_json: Dict[str, Any]
    tiers: List[str]
    numeric_columns: List[str]


class TopAgentsRequest(BaseModel):
    tier: str
    metric: str


class TopAgentsResult(BaseModel):
    agents: List[Dict[str, Any]]
    tier: str
    metric: str
