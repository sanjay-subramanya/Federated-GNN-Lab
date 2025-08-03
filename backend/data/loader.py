
import torch
import logging
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch_geometric.data import Data
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from utils.logging_utils import configure_logging
from config.settings import Config

configure_logging()
logger = logging.getLogger(__name__)

class DataObj:
    """A simple container for client-specific data."""
    def __init__(self, x, edge_index, y, train_mask, val_mask, class_weights):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.class_weights = class_weights

def load_and_partition_data(protein_df, phen_df, num_clients=Config.n_clients) -> Tuple[List[DataObj], int, int]:
    """
    Loads and partitions the protein and phenotypic data for federated learning.
    """
    logger.info("Loading and partitioning data...")
    
    # Preprocessing steps
    protein_df.index = protein_df.index.str.strip()
    phen_df.index = phen_df.index.str.strip()
    merged_df = protein_df.join(phen_df, how='inner')
    logger.info(f"Data: Shape after initial merge: {merged_df.shape}")

    # Use "vital_status.demographic" as binary label
    merged_df = merged_df[merged_df['vital_status.demographic'].isin(['Alive', 'Dead'])].copy()
    label_map = {'Dead': 0, 'Alive': 1}
    merged_df['target_label'] = merged_df['vital_status.demographic'].map(label_map)
    logger.info(f"Data: Shape after filtering 'Alive'/'Dead': {merged_df.shape}")
    logger.info(f"Data: Class distribution after filtering: \n{merged_df['vital_status.demographic'].value_counts().to_string()}")

    # Determine protein columns (numerical/float) and other columns
    protein_cols = [col for col in merged_df.columns if merged_df[col].dtype.kind in ['i', 'f'] and col != 'target_label']
    
    # Exclude non-feature columns
    columns_to_exclude_from_features = ['id', 'case_id']
    protein_cols = [col for col in protein_cols if col not in columns_to_exclude_from_features]
    
    X = merged_df[protein_cols].values
    y = merged_df['target_label'].values

    logger.info(f"Data: X shape before imputation: {X.shape}")

    # Imputation
    imputer = KNNImputer(n_neighbors=Config.n_neighbors_knn_imputer)
    X = imputer.fit_transform(X)
    logger.info(f"Data: X shape after KNN Imputation: {X.shape}")

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    logger.info(f"Data: X shape after StandardScaler: {X.shape}")

    # Apply SMOTE globally for client splitting
    smote = SMOTE(random_state=Config.random_seed)
    
    unique_labels_before_smote, counts_before_smote = np.unique(y, return_counts=True)
    logger.info(f"Data: Original class distribution before SMOTE: {list(zip(unique_labels_before_smote, counts_before_smote))}")

    if len(unique_labels_before_smote) < 2 or (len(unique_labels_before_smote) == 2 and np.min(counts_before_smote) < 2):
        logger.warning("Data: SMOTE skipped. Not enough classes or minority samples (min 2 required per class).")
        X_balanced, y_balanced = X, y
    else:
        X_balanced, y_balanced = smote.fit_resample(X, y)
    
    logger.info(f"Data: Original data shape: {X.shape}, Balanced data shape: {X_balanced.shape}")
    logger.info(f"Data: Class distribution after SMOTE: {np.bincount(y_balanced).tolist()}")

    # Create client partitions
    client_indices = defaultdict(list)
    all_indices = list(range(len(X_balanced)))
    random.shuffle(all_indices)

    for i, idx in enumerate(all_indices):
        client_indices[i % num_clients].append(idx)

    client_datasets = []
    for i in range(num_clients):
        logger.info(f"--- Preparing Client {i+1} data ---")
        idxs = client_indices[i]
        
        # Check if client has data assigned
        if not idxs:
            logger.warning(f"Client {i+1} has no data assigned. Skipping this client.")
            continue

        X_c = X_balanced[idxs]
        y_c = y_balanced[idxs]
        
        if len(y_c) < 2:
            logger.warning(f"Client {i+1}: Only {len(y_c)} samples. Train/val split will be non-stratified or all samples to train.")
            idx_train = np.arange(len(y_c))
            idx_val = np.array([])
            logger.warning(f"Client {i+1}: Not enough unique classes or samples for stratified split. Assigning all to train.")
        else:
            try:
                idx_train, idx_val = train_test_split(np.arange(len(y_c)), test_size=Config.test_size, stratify=y_c, random_state=Config.random_seed)
            except ValueError as e:
                logger.warning(f"Client {i+1}: Stratified split failed ({e}). Attempting non-stratified split.")
                idx_train, idx_val = train_test_split(np.arange(len(y_c)), test_size=Config.test_size, random_state=Config.random_seed)
        
        # Ensure minimum samples for training/validation
        if len(idx_train) == 0:
            logger.critical(f"Client {i+1}: No training samples after split! This client cannot train.")
        if len(idx_val) == 0:
            logger.warning(f"Client {i+1}: No validation samples after split. Evaluation metrics for this client might be NaN.")

        # Graph Creation
        # kneighbors_graph requires n_samples >= n_neighbors + 1
        # If a client has too few samples, adj will be None
        adj = None
        edge_index = None
        if len(X_c) >= Config.n_neighbors_graph + 1:
            try:
                adj = kneighbors_graph(X_c, n_neighbors=Config.n_neighbors_graph, mode='connectivity', include_self=False)
                edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long).to(Config.device)
            except Exception as e:
                logger.error(f"Client {i+1}: Error creating kneighbors_graph: {e}. Edge index set to None.")
        else:
            logger.warning(f"Client {i+1}: Not enough samples ({len(X_c)}) for kneighbors_graph with n_neighbors={Config.n_neighbors_graph}. Edge index set to None.")

        x = torch.tensor(X_c, dtype=torch.float).to(Config.device)
        y_torch = torch.tensor(y_c, dtype=torch.long).to(Config.device)

        train_mask = torch.zeros(len(y_c), dtype=torch.bool).to(Config.device)
        val_mask = torch.zeros(len(y_c), dtype=torch.bool).to(Config.device)
        train_mask[idx_train] = True
        if len(idx_val) > 0:
            val_mask[idx_val] = True

        # Compute class weights for this client's training data
        class_weights = None
        if train_mask.sum().item() > 0 and len(np.unique(y_c[idx_train])) > 1:
            class_weights_np = compute_class_weight('balanced', classes=np.unique(y_c[idx_train]), y=y_c[idx_train])
            class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(Config.device)
        else:
            # Fallback for clients with single class or no training samples
            logger.warning(f"Client {i+1}: Cannot compute balanced class weights for training data (only one class or no samples). Using uniform weights for 2 classes.")
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(Config.device)

        data = DataObj(x=x, edge_index=edge_index, y=y_torch,
                       train_mask=train_mask, val_mask=val_mask,
                       class_weights=class_weights)
        client_datasets.append(data)
        logger.info(f"Client {i+1} data loaded: X shape={x.shape}, Y shape={y_torch.shape}")
        logger.info(f"Client {i+1} masks: Train samples={train_mask.sum().item()}, Val samples={val_mask.sum().item()}")
        logger.info(f"Client {i+1} train label distribution: {np.bincount(y_c[idx_train]).tolist() if train_mask.sum().item() > 0 else 'N/A (empty)'}")
        logger.info(f"Client {i+1} val label distribution: {np.bincount(y_c[idx_val]).tolist() if val_mask.sum().item() > 0 else 'N/A (empty)'}")

    if not client_datasets:
        logger.critical("No valid client datasets were created. Cannot proceed with FL simulation.")
        raise ValueError("No valid client datasets.")

    num_features = client_datasets[0].x.size(1) # Feature count from the first valid client's data
    # Ensure num_classes is consistent across clients, or take it from the global balanced data
    num_classes = len(np.unique(y_balanced))
    logger.info(f"Global: num_features={num_features}, num_classes={num_classes}")

    return client_datasets, num_features, num_classes