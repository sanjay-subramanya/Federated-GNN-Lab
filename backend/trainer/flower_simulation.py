import logging
import flwr as fl
from typing import List, Callable
from data.loader import DataObj
from config.settings import Config
from trainer.flower_client import FlowerClient
from trainer.flower_server import (
    fit_config,
    evaluate_config,
    evaluate_metrics_aggregation,
    SaveableFedAvg,
    save_federated_model
)
from utils.logging_utils import configure_logging
from utils.seeding import set_seeds

configure_logging()
logger = logging.getLogger(__name__)

def create_client(client_datasets: List[DataObj], num_features: int, num_classes: int) -> Callable[[str], FlowerClient]:
    """Return a function that creates a FlowerClient object for a given client ID."""
    def client_fn(cid: str) -> FlowerClient:
        """Instantiate and return a FlowerClient for a given client ID."""
        client_id_int = int(cid)
        
        if client_id_int < len(client_datasets):
            client_data = client_datasets[client_id_int]
            logger.info(f"Creating FlowerClient object for CID {cid}")
            return FlowerClient(client_id_int, client_data, num_features, num_classes)
        else:
            logger.error(f"Attempted to create client with CID {cid} but no data available.")
            raise ValueError(f"No data for client ID: {cid}")
    return client_fn
    

def run_flower_simulation(client_datasets: List[DataObj], num_features: int, num_classes: int, num_rounds: int):
    set_seeds()
    logger.info("Starting Flower Federated Learning simulation...")

    # Get the client_fn, which is a closure over the data
    client_fn_closure = create_client(client_datasets, num_features, num_classes)

    # Create a custom strategy using SaveableFedAvg
    strategy = SaveableFedAvg(
        fraction_fit=Config.fraction_fit,
        fraction_evaluate=Config.fraction_evaluate,
        min_fit_clients=Config.min_fit_clients,
        min_evaluate_clients=Config.min_evaluate_clients,
        min_available_clients=Config.min_available_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn_closure,
        num_clients=len(client_datasets),
        client_resources={"num_cpus": 1},
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
    )

    save_federated_model(strategy, client_datasets, num_features, num_classes, Config.model_dir / "flower_fl_model.pth")
    logger.info("Flower-based simulation finished.")