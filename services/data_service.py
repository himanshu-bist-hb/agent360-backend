import io
import pandas as pd
from fastapi import UploadFile, HTTPException
from utils.helpers import compute_quality_score, sanitize_record
from store import store


ACCEPTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# ── Column category mapping (keyword-based auto-tagging) ──────────────────────
CATEGORY_KEYWORDS = {
    "New Business": [
        "nb_", "quote", "conversion", "business_quality", "discount_utilization",
        "cross_sell", "producers", "commission_rate", "nb_premium", "nb_policy",
    ],
    "Retention": [
        "retention", "lapse", "cancellation", "renewal",
    ],
    "Financial / GWP": [
        "gwp", "commission_earned", "avg_premium", "premium_per_policy",
    ],
    "Pricing": [
        "rate_competitiveness", "competitor_rate", "pricing",
    ],
    "Agent Profile": [
        "experience", "digital_adoption", "days_since", "producers_count",
        "agency_size",
    ],
}


def categorize_columns(columns: list, numeric_cols: list) -> dict:
    """
    Auto-assign each numeric column to a display category.
    Returns {category: [col, col, ...]}
    """
    result = {cat: [] for cat in CATEGORY_KEYWORDS}
    result["Other"] = []
    assigned = set()

    for col in numeric_cols:
        col_lower = col.lower()
        matched = False
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in col_lower for kw in keywords):
                result[cat].append(col)
                assigned.add(col)
                matched = True
                break
        if not matched:
            result["Other"].append(col)
            assigned.add(col)

    # Remove empty categories
    return {k: v for k, v in result.items() if v}


async def process_upload(file: UploadFile) -> dict:
    """
    Read, validate, and store the uploaded file.
    Returns validation summary for the frontend.
    """
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a CSV or Excel file.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file contains no data rows.")

    # Find agent_id column (case-insensitive)
    agent_id_col = next(
        (col for col in df.columns if col.strip().lower() == "agent_id"), None
    )
    if agent_id_col is None:
        raise HTTPException(
            status_code=422,
            detail="Validation failed: dataset must contain an 'agent_id' column.",
        )

    # Identify numeric columns (exclude agent_id)
    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns
        if col != agent_id_col
    ]

    null_summary = {col: int(df[col].isnull().sum()) for col in df.columns}
    quality_score, quality_label = compute_quality_score(df)

    # Store in global store
    store.reset()
    store.raw_df = df
    store.agent_id_col = agent_id_col
    store.all_columns = list(df.columns)
    store.numeric_columns = numeric_cols
    store.selected_features = numeric_cols.copy()

    preview_records = df.head(10).fillna("").to_dict(orient="records")
    preview = [sanitize_record(r) for r in preview_records]

    column_categories = categorize_columns(list(df.columns), numeric_cols)

    # Collect available states and districts for hierarchy filters
    states    = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    districts = sorted(df["district"].dropna().unique().tolist()) if "district" in df.columns else []

    return {
        "success": True,
        "message": "File uploaded and validated successfully.",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "numeric_columns": numeric_cols,
        "null_summary": null_summary,
        "quality_label": quality_label,
        "quality_score": quality_score,
        "preview": preview,
        "agent_id_col": agent_id_col,
        "column_categories": column_categories,
        "states": states,
        "districts": districts,
    }
