from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Literal, Optional
import pandas as pd
import os
from config.settings import Config 
from data.io import load_phenotype_data

router = APIRouter()

class PatientEntry(BaseModel):
    id: str
    age: float
    stage: str
    status: Literal["Alive", "Dead"]

# Module-level cache
cached_eda_data: Optional[List[PatientEntry]] = None

@router.get("/eda", response_model=List[PatientEntry])
def get_patient_eda_data(request: Request):
    global cached_eda_data
    if cached_eda_data is not None:
        return cached_eda_data

    ctx = request.app.state.ctx
    phen_df = ctx.phen_df_raw

    required_cols = ["id", "vital_status.demographic", "age_at_index.demographic", "ajcc_pathologic_stage.diagnoses"]
    for col in required_cols:
        if col not in phen_df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = phen_df[required_cols].dropna()

    def map_to_main_stage(stage_raw):
        stage_raw = str(stage_raw).strip().upper()
        if "IV" in stage_raw:
            return "Stage IV"
        elif "III" in stage_raw:
            return "Stage III"
        elif "II" in stage_raw:
            return "Stage II"
        elif "I" in stage_raw:
            return "Stage I"
        return None

    df["stage_mapped"] = df["ajcc_pathologic_stage.diagnoses"].apply(map_to_main_stage)
    df = df[df["stage_mapped"].isin(["Stage I", "Stage II", "Stage III", "Stage IV"])]

    df["status_mapped"] = df["vital_status.demographic"].str.strip().str.lower().map({
        "alive": "Alive",
        "dead": "Dead"
    })
    df = df[df["status_mapped"].isin(["Alive", "Dead"])]

    # Convert to list of Pydantic objects
    cached_eda_data = [
        PatientEntry(
            id=row["id"],
            age=row["age_at_index.demographic"],
            stage=row["stage_mapped"],
            status=row["status_mapped"]
        )
        for _, row in df.iterrows()
    ]
    return cached_eda_data
