import os
import torch
from pathlib import Path

class Config:
    # General settings
    parent_dir = Path(__file__).resolve().parent.parent
    model_dir = parent_dir / "saved_models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    n_neighbors_knn_imputer = 5
    random_seed = 30
    n_neighbors_graph = 5
    test_size = 0.2

    # FL Training setup
    n_clients = 3
    n_rounds = 10
    local_epochs = 5
    label_map = {'Alive': 1, 'Dead': 0}

    # Model hyperparameters
    dropout = 0.5
    weight_decay = 1e-3
    learning_rate = 0.001
    hidden_dim = 64
    out_dim = 2

    # Flower server config
    fraction_fit = 1.0
    fraction_evaluate = 1.0
    min_fit_clients = 2
    min_evaluate_clients = 2
    min_available_clients = 3

    # Simulation mode toggle
    realistic_flower_simulation = False

    # Vercel Blob config
    vercel_blob_upload_url = "https://blob.vercel-storage.com/upload"
    vercel_blob_delete_url = "https://blob.vercel-storage.com/delete"
    vercel_blob_download_url = "https://blob.vercel-storage.com"
    vercel_blob_token = os.getenv("VERCEL_BLOB_TOKEN")
    if not vercel_blob_token:
        raise RuntimeError("Missing VERCEL_BLOB_TOKEN in environment variables")
