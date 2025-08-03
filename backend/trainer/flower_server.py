import os
import torch
import logging
import json
import flwr as fl
from typing import Dict, Tuple, List
from flwr.common import Scalar
import numpy as np

from config.settings import Config
from models.gnn_model import SAGENet
from trainer.flower_client import FlowerClient
from data.loader import DataObj
from utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration for the current round."""
    config = {
        "local_epochs": Config.local_epochs,
        "server_round": server_round,
    }
    logger.info(f"Server: Configuring fit for round {server_round} with {Config.local_epochs} local epochs.")
    return config

def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration for the current round's evaluation."""
    config = {
        "server_round": server_round,
    }
    logger.info(f"Server: Configuring evaluation for round {server_round}.")
    return config

def evaluate_metrics_aggregation(metrics: List[Tuple[int, Dict]]) -> Dict[str, Scalar]:
    """Aggregate metrics from clients, weighted by number of examples."""
    accuracies = []
    losses = []
    roc_aucs = []
    total_examples = 0

    for num_examples, m in metrics:
        if num_examples > 0:
            total_examples += num_examples
            if not np.isnan(m.get("accuracy", float("nan"))):
                accuracies.append(num_examples * m["accuracy"])
            if not np.isnan(m.get("loss", float("nan"))):
                losses.append(num_examples * m["loss"])
            if not np.isnan(m.get("roc_auc", float("nan"))):
                roc_aucs.append(num_examples * m["roc_auc"])
        else:
            logger.warning(f"Server: Skipping client metrics with 0 examples or NaN values.")

    aggregated_metrics = {}
    if total_examples > 0:
        if accuracies:
            aggregated_metrics["accuracy"] = sum(accuracies) / total_examples
        if losses:
            aggregated_metrics["loss"] = sum(losses) / total_examples
        if roc_aucs:
            aggregated_metrics["roc_auc"] = sum(roc_aucs) / total_examples
    
    logger.info(f"Server: Aggregated evaluation metrics: {aggregated_metrics}")
    return aggregated_metrics

class SaveableFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None
    
    def aggregate_fit(self, server_round, results, failures):
        """Call parent aggregate_fit method."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Store the final parameters
        if aggregated_parameters is not None:
            self.final_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        return aggregated_parameters, aggregated_metrics

# Function to save the federated model
def save_federated_model(strategy, client_datasets: List[DataObj], num_features: int, num_classes: int, save_path: str = "saved_models/federated_model.pth"):
    """Save the final federated model after training."""
    try:
        save_dir = os.path.dirname(save_path)
        save_path = os.path.join(Config.parent_dir, save_path)
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True) # exist_ok=True prevents an error if the directory already exists
            logger.info(f"Created directory: {save_dir}")

        # Get the final global model parameters from the custom strategy
        if hasattr(strategy, 'final_parameters') and strategy.final_parameters is not None:
            final_parameters = strategy.final_parameters
        else:
            # Fallback: create a dummy client to get the model structure
            logger.warning("Cannot access final parameters from strategy. Creating model with initial structure.")
            dummy_client = FlowerClient(0, client_datasets[0], num_features, num_classes)
            final_parameters = dummy_client.get_parameters({})
        
        # Create a temporary model instance to load the parameters
        temp_model = SAGENet(num_features, Config.hidden_dim, num_classes, Config.dropout).to(Config.device)
        
        # Convert parameters back to tensors and load into model
        params_dict = zip(temp_model.state_dict().keys(), final_parameters)
        state_dict = {k: torch.tensor(v).to(Config.device) for k, v in params_dict}
        temp_model.load_state_dict(state_dict, strict=True)
        
        # Save the model
        torch.save({
            'model_state_dict': temp_model.state_dict(),
            'num_features': num_features,
            'num_classes': num_classes,
            'hidden_dim': Config.hidden_dim,
            'dropout': Config.dropout,
            'model_architecture': 'SAGENet'
        }, save_path)
        
        logger.info(f"Federated model saved successfully to {save_path}")

    except Exception as e:
        logger.error(f"Error saving federated model: {str(e)}")

