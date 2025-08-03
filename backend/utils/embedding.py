import torch
import pandas as pd
from pathlib import Path
from config.settings import Config
from models.gnn_model import SAGENet
from typing import List

def extract_patient_embeddings(model: SAGENet, protein_df: pd.DataFrame, feature_cols: List[str]=None) -> pd.DataFrame:
    
    input_dim = protein_df.shape[1]
    X = protein_df.fillna(0.0).values.astype("float32")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(Config.device)

    with torch.no_grad():
        embeddings = model(X_tensor, edge_index=None, return_embeddings=True).cpu().numpy()

    return pd.DataFrame(embeddings, index=protein_df.index)
