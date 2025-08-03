
import torch
import logging
import torch.nn.functional as F
import numpy as np
from captum.attr import Saliency
from sklearn.neighbors import kneighbors_graph
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from utils.blob_utils import load_file_from_blob_if_needed
from utils.logging_utils import configure_logging
from config.settings import Config
from data.preprocess import preprocess_data
from api.context import AppContext

configure_logging()
logger = logging.getLogger(__name__)
router = APIRouter()

class FeatureImportanceEntry(BaseModel):
    feature_name: str
    importance: float

class FeatureOverlap(BaseModel):
    overlap_percentage: float
    common_features: List[str]

class FeatureImportanceResponse(BaseModel):
    model_name: str
    feature_importances: List[FeatureImportanceEntry]
    overlap_with_global: Optional[FeatureOverlap] = None


def calculate_saliency_and_top_features(
    current_model_path: Path,
    blob_key: str,
    X_tensor: torch.Tensor,
    edge_index_tensor: torch.Tensor,
    feature_column_names: List[str],
    k_features: int,
    ctx: AppContext,
) -> List[FeatureImportanceEntry]:
    try:
        current_model_path = load_file_from_blob_if_needed(blob_key, current_model_path)
        model = ctx._load_model(current_model_path, blob_key)

        saliency = Saliency(model)
        input_tensor_for_saliency = X_tensor.clone().detach().requires_grad_(True).to(Config.device)
        
        with torch.no_grad():
            logits = model(input_tensor_for_saliency, edge_index=edge_index_tensor)
            predicted_classes = torch.argmax(F.softmax(logits, dim=1), dim=1)

        attributions = saliency.attribute(input_tensor_for_saliency, target=predicted_classes)
        feature_attributions_mean = attributions.abs().mean(dim=0)
        
        importance_scores = {}
        for i, feature_name in enumerate(feature_column_names):
            importance_scores[feature_name] = feature_attributions_mean[i].item()

        sorted_features = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
        return [{"feature_name": k, "importance": v} for k, v in sorted_features[:k_features]]
    except Exception as e:
        logger.error(f"Error calculating saliency for model {current_model_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing model {current_model_path}: {e}")


@router.get("/dissect/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance(
    request: Request, 
    model_name: str = "global", 
    top_k: int = 20,
    run_id: Optional[str] = None
    ):
    try:
        ctx = request.app.state.ctx

        X_processed_np, y_processed_np, class_names_list = preprocess_data(
                protein_df=ctx.protein_df_raw, phen_df=ctx.phen_df_raw)
        logger.info("Data prepared and preprocessed locally for this request.")
        X_inference_tensor_local = torch.tensor(X_processed_np, dtype=torch.float32).to(Config.device)
        y_inference_tensor_local = torch.tensor(y_processed_np, dtype=torch.long).to(Config.device)

        adjacency_matrix = kneighbors_graph(
                X_processed_np, 
                n_neighbors=Config.n_neighbors_knn_imputer, 
                mode='connectivity', 
                include_self=False,
                metric='euclidean'
            )
        coo_matrix = adjacency_matrix.tocoo()
        edge_index_np = np.stack([coo_matrix.row, coo_matrix.col], axis=0)
        edge_index_tensor_local = torch.from_numpy(edge_index_np).long().to(Config.device)
        logger.info("Graph edge_index created locally.")

    except Exception as e:
        logger.error(f"Error during data loading/preprocessing/graph creation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare data for feature importance: {e}")

    if run_id:
        current_model_dir = Config.model_dir / run_id
        blob_prefix = f"saved_models/{run_id}"
    else:
        current_model_dir = Config.model_dir  
        blob_prefix = "saved_models"

    if model_name == "global":
        current_model_path = current_model_dir / "global_model_manual.pt"
        blob_key = f"{blob_prefix}/global_model_manual.pt"
    else:
        client_id = model_name.split('_')[-1]
        current_model_path = current_model_dir / f"client_{client_id}_model.pt"
        blob_key = f"{blob_prefix}/client_{client_id}_model.pt"
        global_model_path = current_model_dir / "global_model_manual.pt"
        global_blob_key = f"{blob_prefix}/global_model_manual.pt"

    current_model_importances = calculate_saliency_and_top_features(
        current_model_path,
        blob_key,
        X_inference_tensor_local,
        edge_index_tensor_local,
        ctx.feature_cols,
        top_k,
        ctx,
    )

    overlap_data: Optional[FeatureOverlap] = None
    if model_name != "global":
        global_model_importances = calculate_saliency_and_top_features(
            global_model_path,
            global_blob_key,
            X_inference_tensor_local,
            edge_index_tensor_local,
            ctx.feature_cols,
            top_k,
            ctx,
        )
        global_top_features_names = {entry["feature_name"] for entry in global_model_importances}
        current_client_top_features_names = {entry["feature_name"] for entry in current_model_importances}
        common_features = list(global_top_features_names.intersection(current_client_top_features_names))
        
        # Calculate overlap percentage, handle division by zero if top_k is 0
        overlap_percentage = (len(common_features) / top_k) * 100 if top_k > 0 else 0.0
        overlap_data = FeatureOverlap(
            overlap_percentage=round(overlap_percentage, 2),
            common_features=common_features
        )
        logger.info(f"Overlap with global model calculated: {overlap_percentage:.2f}%")

    logger.info(f"Feature importance calculated for {model_name}. Top {top_k} features retrieved.")
    return FeatureImportanceResponse(
        model_name=model_name,
        feature_importances=current_model_importances,
        overlap_with_global=overlap_data
    )
