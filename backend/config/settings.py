import os
from pathlib import Path

class Config:
    # General settings
    parent_dir = Path(__file__).resolve().parent.parent
    model_dir = parent_dir / "saved_models"
    device = "cpu"
    
    # Data preprocessing
    n_neighbors_knn_imputer = 5
    random_seed = 30
    n_neighbors_graph = 5
    test_size = 0.2

    # FL Training setup
    n_clients = 3
    n_rounds = 8
    local_epochs = 4
    label_map = {'Alive': 1, 'Dead': 0}

    # Model hyperparameters
    dropout = 0.5
    weight_decay = 1e-3
    learning_rate = 0.001
    hidden_dim = 32
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
    vercel_blob_token = os.getenv("BLOB_READ_WRITE_TOKEN")
    vercel_blob_store_url = os.getenv("BLOB_STORE_BASE_URL")
    if not vercel_blob_token:
        raise ValueError("BLOB_READ_WRITE_TOKEN environment variable not set")
