import logging
from config.settings import Config
from utils.seeding import set_seeds
from utils.logging_utils import configure_logging
from data.io import load_protein_data, load_phenotype_data
from data.preprocess import preprocess_data
from data.loader import load_and_partition_data
from trainer.manual_simulation import run_manual_simulation
from trainer.flower_simulation import run_flower_simulation


def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    set_seeds()
    
    logger.info("Loading data...")
    protein_df = load_protein_data()
    phen_df = load_phenotype_data()

    logger.info("Running classical ML preprocessing and analysis...")
    X, y, class_names = preprocess_data(protein_df, phen_df)
    logger.info(f"Classical ML preprocessing complete. X shape: {X.shape}, classes: {class_names}")

    logger.info("Preparing data for federated learning...")
    client_datasets, num_features, num_classes = load_and_partition_data(protein_df, phen_df)
    logger.info(f"Created {len(client_datasets)} client datasets with {num_features} features and {num_classes} classes")

    valid_client_ids = [str(i) for i, data_obj in enumerate(client_datasets) 
                        if data_obj.train_mask.sum().item() > 0]
    logger.info(f"Total clients with valid training data: {len(valid_client_ids)} out of {Config.n_clients}")
    
    if Config.flower_simulation:
        logger.info("Starting Flower FL simulation...")
        run_flower_simulation(client_datasets, num_features, num_classes, Config.n_rounds)
        logger.info("Flower-based FL simulation complete")
    else:
        logger.info("Starting manual FL simulation...")
        global_model, train_losses, val_losses = run_manual_simulation(client_datasets, num_features, num_classes, num_rounds=Config.n_rounds)
        logger.info("Manual FL simulation complete")

if __name__ == "__main__":
    main()