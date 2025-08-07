from fastapi import APIRouter

from .patients import router as patients_router
from .train import router as train_router
from .status import router as status_router
from .explore import router as explore_router
from .predict import router as predict_router
from .metadata import router as metadata_router
from .embeddings import router as embeddings_router
from .importance import router as importance_router
from .divergence import router as divergence_router
from .delete_run import router as deletion_router

router = APIRouter()
router.include_router(train_router)
router.include_router(status_router)
router.include_router(explore_router)
router.include_router(patients_router)
router.include_router(predict_router)
router.include_router(metadata_router)
router.include_router(divergence_router)
router.include_router(importance_router)
router.include_router(embeddings_router)
router.include_router(deletion_router)
