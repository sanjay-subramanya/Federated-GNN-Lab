import torch
import pandas as pd
from typing import Optional, List
from pathlib import Path
from config.settings import Config
from models.gnn_model import SAGENet
from utils.blob_utils import load_file_from_blob_if_needed
from data.preprocess import prepare_filtered_data
from data.io import load_protein_data, load_phenotype_data

class AppContext:
    def __init__(self):
        self.global_model_path = str(Config.model_dir / "global_model_manual.pt")
        self.flwr_model_path = str(Config.model_dir / "flower_fl_model.pth")
        self.protein_df_raw = load_protein_data()
        self.phen_df_raw = load_phenotype_data()
        X_raw, Y_raw, self.merged_df, self.feature_cols = prepare_filtered_data(self.protein_df_raw, self.phen_df_raw, exclude_columns=['id', 'case_id'])
        self.protein_df = self.merged_df[self.feature_cols]
        self.flwr_model = self._load_model(self.flwr_model_path, "saved_models/flower_fl_model.pth")
        self.global_model = self._load_model(self.global_model_path, "saved_models/global_model_manual.pt")

    def _load_model(self, model_path, blob_key: str) -> SAGENet:
        if not isinstance(model_path, str):
            model_path = str(model_path)
        input_dim = len(self.feature_cols)
        model = SAGENet(in_dim=input_dim, hidden_dim=Config.hidden_dim, out_dim=Config.out_dim, dropout=Config.dropout)
        local_path = load_file_from_blob_if_needed(blob_key, model_path)
        checkpoint = torch.load(local_path, map_location=torch.device(Config.device))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model