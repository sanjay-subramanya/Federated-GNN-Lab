import json
import logging
from pathlib import Path
from config.settings import Config
from utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

def load_metadata(metadata_path: Path) -> tuple[int, int]:
    
    try:
        with open(metadata_path, "r") as f:
            train_metadata = json.load(f)
        # Use default values from config as fallback if a key is missing
        num_clients_trained = train_metadata.get("num_clients", Config.n_clients) 
        num_rounds_trained = train_metadata.get("num_rounds", Config.n_rounds)
    except FileNotFoundError:
        logger.warning(f"Metadata file not found at {metadata_path}. Using default num_clients_trained={Config.n_clients} and num_rounds_trained={Config.n_rounds}.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {metadata_path}. Using default num_clients_trained={Config.n_clients} and num_rounds_trained={Config.n_rounds}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading metadata: {e}. Using default num_clients_trained={Config.n_clients} and num_rounds_trained={Config.n_rounds}.")
    
    logger.info(f"Loaded training metadata. Num clients trained: {num_clients_trained}, Num rounds trained: {num_rounds_trained}")
    return num_clients_trained, num_rounds_trained
