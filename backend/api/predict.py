from fastapi import APIRouter, Request, HTTPException
import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel

router = APIRouter()

class PatientConfig(BaseModel):
    patient_id: str

@router.post("/predict")
def predict(request: Request, req: PatientConfig):
    pid = req.patient_id.strip()
    ctx = request.app.state.ctx

    if pid not in ctx.protein_df.index:
        raise HTTPException(status_code=404, detail=f"Patient ID '{pid}' not found")

    x = ctx.protein_df.loc[pid].values.astype(np.float32).reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        logits = ctx.flwr_model(x_tensor, edge_index=None)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return {
        "patient_id": pid,
        "prediction": "Alive" if pred_class == 1 else "Dead",
        "confidence": round(confidence * 100, 2)
    }
