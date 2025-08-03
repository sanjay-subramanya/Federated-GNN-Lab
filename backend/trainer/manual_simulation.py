import os
import gc
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from pathlib import Path
from torch import nn

from config.settings import Config
from models.gnn_model import SAGENet
from data.loader import DataObj
from utils.metrics import calculate_model_divergence
from utils.blob_utils import upload_file_to_blob
from utils.logging_utils import configure_logging
from utils.seeding import set_seeds

configure_logging()
logger = logging.getLogger(__name__)

def train_one_client(model: nn.Module, client_data_obj: DataObj, local_epochs_val: int = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    criterion = nn.CrossEntropyLoss(weight=client_data_obj.class_weights)

    if client_data_obj.train_mask.sum().item() == 0:
        logger.warning(f"Skipping training for a client with no training samples in manual simulation.")
        return model.state_dict(), float("nan"), float("nan")

    model.train()
    current_train_loss = float("nan")
    for _ in range(local_epochs_val):
        optimizer.zero_grad()
        out = model(client_data_obj.x, client_data_obj.edge_index)
        loss = criterion(out[client_data_obj.train_mask], client_data_obj.y[client_data_obj.train_mask])
        loss.backward()
        optimizer.step()
        current_train_loss = loss.item()

    model.eval()
    current_val_loss = float("nan")
    if client_data_obj.val_mask.sum().item() > 0:
        with torch.no_grad():
            out_val = model(client_data_obj.x, client_data_obj.edge_index)
            current_val_loss = criterion(out_val[client_data_obj.val_mask], client_data_obj.y[client_data_obj.val_mask]).item()
    else:
        logger.warning(f"Skipping validation for a client with no validation samples in manual simulation.")

    return model.state_dict(), current_train_loss, current_val_loss


def average_weights(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        if isinstance(weights_list[0][key], torch.Tensor):
            avg_weights[key] = sum([client[key].cpu() for client in weights_list]) / len(weights_list)
            avg_weights[key] = avg_weights[key].to(Config.device)
        else:
            avg_weights[key] = sum([client[key] for client in weights_list]) / len(weights_list)
    return avg_weights

def plot_client_losses(client_train_losses: List[List[float]], client_val_losses: List[List[float]]):
    plt.figure(figsize=(12, 6))
    for i in range(len(client_train_losses)):
        if any(not np.isnan(loss) for loss in client_train_losses[i]):
            plt.plot(client_train_losses[i], label=f'Client {i+1} Train Loss')
            plt.plot(client_val_losses[i], label=f'Client {i+1} Val Loss', linestyle='--')
    plt.xlabel('Federated Rounds')
    plt.ylabel('Loss')
    plt.title('Client-wise Training and Validation Loss (Manual Simulation)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_final_models_and_metadata(
    global_model,
    client_models,
    client_datasets,
    num_rounds,
    divergence_history,
    run_id: str = None
):
    
    if not run_id:
        model_dir = Config.model_dir
        blob_prefix = "saved_models"
    else:
        model_dir = Config.model_dir / run_id
        blob_prefix = f"saved_models/{run_id}"

    os.makedirs(model_dir, exist_ok=True)

    global_model_path = model_dir / "global_model_manual.pt"
    torch.save({"model_state_dict": global_model.state_dict()}, global_model_path)
    upload_file_to_blob(f"{blob_prefix}/global_model_manual.pt", str(global_model_path))
    logger.info(f"Saved final global model (manual simulation) to {global_model_path}")

    for i, client_model in enumerate(client_models):
        client_model_path = model_dir / f"client_{i+1}_model.pt"
        torch.save({"model_state_dict": client_model.state_dict()}, client_model_path)
        upload_file_to_blob(f"{blob_prefix}/client_{i+1}_model.pt", str(client_model_path))
        logger.info(f"Saved client {i+1} model to {client_model_path}")

    metadata_path = model_dir / "_train_metadata.json"
    metadata = {
        "num_clients": len(client_datasets),
        "num_rounds": num_rounds,
        "last_training_time": datetime.now().isoformat(),
        "run_id": run_id
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    upload_file_to_blob(f"{blob_prefix}/_train_metadata.json", str(metadata_path))

    divergence_path = model_dir / "_divergence_metrics.json"
    with open(divergence_path, "w") as f:
        json.dump(divergence_history, f, indent=4)
    upload_file_to_blob(f"{blob_prefix}/_divergence_metrics.json", str(divergence_path))
    logger.info(f"Saved divergence metrics to {divergence_path}")


def run_manual_simulation(
    client_datasets: List[DataObj], 
    num_features: int, 
    num_classes: int, 
    num_rounds: int = Config.n_rounds, 
    yield_per_round: bool = False,
    run_id: str = None,
    ):
    set_seeds()

    # Initialize global model
    global_model = SAGENet(num_features, hidden_dim=Config.hidden_dim, out_dim=num_classes, dropout=Config.dropout).to(Config.device)
    global_model_state = global_model.state_dict()

    divergence_history = []
    client_models = [None for _ in client_datasets]

    # Track losses for each client
    client_train_losses = [[] for _ in range(len(client_datasets))]
    client_val_losses = [[] for _ in range(len(client_datasets))]
    
    # Run federated learning rounds
    for round_num in range(num_rounds):
        # print(f"\n--- Federated Round {round_num+1} (Manual Simulation) ---")
        local_states = []
        
        for client_id_idx, client_data in enumerate(client_datasets):
            if client_data.train_mask.sum().item() > 0:

                client_model = SAGENet(num_features, hidden_dim=Config.hidden_dim, out_dim=num_classes, dropout=Config.dropout).to(Config.device)
                client_model.load_state_dict(global_model_state)

                local_state, train_loss, val_loss = train_one_client(
                    client_model, client_data, Config.local_epochs
                )
                local_states.append(local_state)
                client_train_losses[client_id_idx].append(train_loss)
                client_val_losses[client_id_idx].append(val_loss)

                # If final round, save client model for downstream tasks
                if round_num == num_rounds - 1:
                    client_models[client_id_idx] = client_model
                
                del client_model, local_state

            else:
                logger.warning(f"Client {client_id_idx+1} has no training data, skipping this round.")
                client_train_losses[client_id_idx].append(float("nan"))
                client_val_losses[client_id_idx].append(float("nan"))
        
        gc.collect()

        if local_states:
            global_model_state = average_weights(local_states)
            global_model.load_state_dict(global_model_state)
        else:
            logger.warning("No clients participated in this round, global model state not updated.")

        global_loss_round = float(np.nanmean([v[-1] for v in client_val_losses if v]))

        round_divergence = {
            "round": round_num + 1,
            "global_loss": global_loss_round,
            "client_divergence": {}
        }

        # for i in range(len(client_models)):
        for i, state_dict in enumerate(local_states):
            client_id = f"client_{i+1}"
            layer_divergences = calculate_model_divergence(state_dict, global_model_state)
            round_divergence["client_divergence"][client_id] = layer_divergences

        divergence_history.append(round_divergence)

        if yield_per_round:
            yield {
                "round": round_num + 1,
                "global_loss": round(global_loss_round, 5),
                "client_val": {str(i+1): round(client_val_losses[i][-1], 5) for i in range(len(client_val_losses))},
                "client_train": {str(i+1): round(client_train_losses[i][-1], 5) for i in range(len(client_train_losses))},
                "run_id": run_id,
            }

    if not yield_per_round:
        run_id = None
        plot_client_losses(client_train_losses, client_val_losses)       

    save_final_models_and_metadata(global_model, client_models, client_datasets, num_rounds, divergence_history, run_id=run_id)
    return global_model, client_train_losses, client_val_losses