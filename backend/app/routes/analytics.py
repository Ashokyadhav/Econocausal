from io import BytesIO
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/api/analytics",
    tags=["Analytics"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_DATASETS = 5

REQUIRED_COLUMNS = [
    "recency_days",
    "visit_frequency",
    "historical_spend",
    "cart_value",
    "competitor_browse_index",
    "churn_risk_score",
    "tenure_months",
    "is_mobile",
    "discount_offered",
    "converted",
]

FEATURE_COLUMNS = [
    "recency_days",
    "visit_frequency",
    "historical_spend",
    "cart_value",
    "competitor_browse_index",
    "churn_risk_score",
    "tenure_months",
    "is_mobile",
]

DATASET_STORE = {}
TRAINED_STORE = {}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 10 MB upload limit.",
        )

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {str(exc)}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows.",
        )

    if len(df.columns) != len(set(df.columns)):
        raise HTTPException(
            status_code=400,
            detail="CSV contains duplicate column names.",
        )

    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required columns are missing.",
                "missing_columns": missing_columns,
                "required_columns": REQUIRED_COLUMNS,
            },
        )

    if len(df) < 50:
        raise HTTPException(
            status_code=422,
            detail="At least 50 customer records are required.",
        )

    missing_values = df[REQUIRED_COLUMNS].isnull().sum()
    missing_values = {
        column: int(count)
        for column, count in missing_values.items()
        if count > 0
    }

    if missing_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Dataset contains missing required values.",
                "missing_values": missing_values,
            },
        )

    numeric_columns = FEATURE_COLUMNS + [
        "discount_offered",
        "converted",
    ]

    try:
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Non-numeric value found in a numeric column: {str(exc)}",
        ) from exc

    if not df[numeric_columns].apply(
        lambda column: column.map(
            lambda value: pd.notna(value) and abs(float(value)) != float("inf")
        )
    ).all().all():
        raise HTTPException(
            status_code=422,
            detail="Dataset contains non-finite numeric values.",
        )

    if not set(df["discount_offered"].unique()).issubset({0, 10, 20}):
        raise HTTPException(
            status_code=422,
            detail="discount_offered must contain only 0, 10, or 20.",
        )

    treatment_counts = df["discount_offered"].value_counts().to_dict()

    insufficient_treatments = {
        str(treatment): int(count)
        for treatment, count in treatment_counts.items()
        if count < 5
    }

    if insufficient_treatments:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Each treatment group must contain at least 5 records.",
                "treatment_counts": {
                    str(key): int(value)
                    for key, value in treatment_counts.items()
                },
            },
        )

    if not set(df["discount_offered"].unique()) == {0, 10, 20}:
        raise HTTPException(
            status_code=422,
            detail="Dataset must contain all three treatment groups: 0, 10, and 20.",
        )

    if not set(df["converted"].unique()).issubset({0, 1}):
        raise HTTPException(
            status_code=422,
            detail="converted must contain only binary values: 0 or 1.",
        )

    if not set(df["is_mobile"].unique()).issubset({0, 1}):
        raise HTTPException(
            status_code=422,
            detail="is_mobile must contain only 0 or 1.",
        )

    if (
        (df["recency_days"] < 0).any()
        or (df["visit_frequency"] < 0).any()
        or (df["historical_spend"] < 0).any()
        or (df["cart_value"] < 0).any()
        or (df["tenure_months"] < 0).any()
        or (~df["competitor_browse_index"].between(0, 1)).any()
        or (~df["churn_risk_score"].between(0, 1)).any()
    ):
        raise HTTPException(
            status_code=422,
            detail="One or more customer features contain invalid ranges.",
        )

    if len(DATASET_STORE) >= MAX_DATASETS:
        DATASET_STORE.clear()

    dataset_id = str(uuid4())
    DATASET_STORE[dataset_id] = df.copy()

    return {
        "message": "Dataset uploaded and validated successfully.",
        "dataset_id": dataset_id,
        "filename": file.filename,
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "treatment_distribution": {
            str(key): int(value)
            for key, value in treatment_counts.items()
        },
        "conversion_rate": round(float(df["converted"].mean()), 4),
        "missing_values": 0,
        "feature_columns": FEATURE_COLUMNS,
        "preview": df.head(5).to_dict(orient="records"),
    }

from core.dml_engine import DoubleMachineLearningEngine


@router.post("/{dataset_id}/train")
async def train_dml_model(dataset_id: str):
    # Step 1: Retrieve uploaded dataset
    if dataset_id not in DATASET_STORE:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Please upload the dataset again.",
        )

    df = DATASET_STORE[dataset_id].copy()

    try:
        # Step 2: Initialize and train DML model
        model = DoubleMachineLearningEngine(
            n_folds=5,
            random_seed=42,
        )

        model.fit(
            df,
            feature_cols=FEATURE_COLUMNS,
            outcome_col="converted",
            treatment_col="discount_offered",
        )

        # Step 3: Generate customer-level predictions
        predictions_df = model.classify_uplift_quadrants(df)

        # Step 4: Generate model summary
        summary = model.get_summary_metrics()

        # Step 5: Store trained model and predictions
        TRAINED_STORE[dataset_id] = {
            "model": model,
            "predictions": predictions_df,
            "summary": summary,
        }

        # Step 6: Return training results
        return {
            "message": "DML model trained successfully.",
            "dataset_id": dataset_id,
            "training_status": "completed",
            "total_records": int(len(df)),
            "ate_results": summary["ate_results"],
            "diagnostics": summary["diagnostics"],
            "feature_coefficients": summary["feature_coefficients"],
            "prediction_summary": {
                "total_customers": int(len(predictions_df)),
                "average_tau_10": round(
                    float(predictions_df["pred_tau_10"].mean()), 4
                ),
                "average_tau_20": round(
                    float(predictions_df["pred_tau_20"].mean()), 4
                ),
                "uplift_quadrants": {
                    str(key): int(value)
                    for key, value in predictions_df[
                        "uplift_quadrant"
                    ].value_counts().items()
                },
            },
        }

    except HTTPException:
        raise

    except Exception as exc:
        TRAINED_STORE.pop(dataset_id, None)

        raise HTTPException(
            status_code=500,
            detail=f"DML model training failed: {str(exc)}",
        ) from exc

@router.get("/{dataset_id}/results")
async def get_dml_results(dataset_id: str):
    # Check whether the dataset exists
    if dataset_id not in DATASET_STORE:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Please upload the dataset again.",
        )

    # Check whether DML training is completed
    if dataset_id not in TRAINED_STORE:
        raise HTTPException(
            status_code=409,
            detail="DML model has not been trained for this dataset.",
        )

    trained_data = TRAINED_STORE[dataset_id]
    predictions_df = trained_data["predictions"]
    summary = trained_data["summary"]

    return {
        "message": "DML results retrieved successfully.",
        "dataset_id": dataset_id,
        "training_status": "completed",
        "total_records": int(len(predictions_df)),
        "ate_results": summary["ate_results"],
        "diagnostics": summary["diagnostics"],
        "feature_coefficients": summary["feature_coefficients"],
        "prediction_summary": {
            "total_customers": int(len(predictions_df)),
            "average_tau_10": round(
                float(predictions_df["pred_tau_10"].mean()), 4
            ),
            "average_tau_20": round(
                float(predictions_df["pred_tau_20"].mean()), 4
            ),
            "uplift_quadrants": {
                str(key): int(value)
                for key, value in predictions_df[
                    "uplift_quadrant"
                ].value_counts().items()
            },
        },
    }
