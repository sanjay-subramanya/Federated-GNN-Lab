from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/patients")
def get_patient_ids(request: Request):
    ctx = request.app.state.ctx
    return {"patient_ids": ctx.protein_df.index.tolist()}