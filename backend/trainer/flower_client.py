import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import flwr as fl
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple
from flwr.common import Parameters

from config.settings import Config
from models.gnn_model import SAGENet
from data.loader import DataObj
from utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, data: DataObj, num_features: int, num_classes: int):
        self.client_id = client_id
        self.data = data
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = SAGENet(num_features, Config.hidden_dim, num_classes, Config.dropout).to(Config.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=self.data.class_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        
        self.can_train = self.data.train_mask.sum().item() > 0
        self.can_evaluate = self.data.val_mask.sum().item() > 0
        self.has_graph_data = (self.data.edge_index is not None and self.data.edge_index.numel() > 0)

        if not self.can_train:
            logger.warning(f"Client {self.client_id}: No training samples available. `fit` will be skipped.")
        if not self.can_evaluate:
            logger.warning(f"Client {self.client_id}: No validation samples available. `evaluate` will be skipped or return NaN.")
        if not self.has_graph_data:
            logger.warning(f"Client {self.client_id}: No valid edge_index for GNN. Model will behave more like an MLP.")

    def get_parameters(self, config: Dict) -> Parameters:
        """Returns the current model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets the model parameters from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(Config.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Trains the model locally on the client's data."""
        self.set_parameters(parameters)
        local_epochs = config.get("local_epochs", Config.local_epochs)

        if not self.can_train:
            logger.warning(f"Client {self.client_id}: Skipping fit due to no training data.")
            return self.get_parameters({}), 0, {"train_loss": float("nan")}

        self.model.train()
        train_loss = float("nan") # Default value
        
        if self.data.train_mask.sum().item() > 0:
            for epoch in range(local_epochs):
                self.optimizer.zero_grad()
                out = self.model(self.data.x, self.data.edge_index)
                loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
                loss.backward()
                self.optimizer.step()
                train_loss = loss.item()
            logger.info(f"Client {self.client_id}: Training complete, final train loss: {train_loss:.4f}")
        else:
            logger.warning(f"Client {self.client_id}: No samples in training mask, skipping training loop.")

        num_examples_train = self.data.train_mask.sum().item()
        return self.get_parameters({}), num_examples_train, {"train_loss": train_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluates the model locally on the client's validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        val_loss = float("nan")
        accuracy = float("nan")
        roc_auc = float("nan")
        num_examples_val = self.data.val_mask.sum().item()

        if not self.can_evaluate:
            logger.warning(f"Client {self.client_id}: Skipping evaluation due to no validation data. Returning NaN.")
            return float("nan"), 0, {"accuracy": float("nan"), "roc_auc": float("nan"), "loss": float("nan")}
        
        if num_examples_val == 0:
            logger.warning(f"Client {self.client_id}: Validation set is empty. Returning NaN for metrics.")
            return float("nan"), 0, {"accuracy": float("nan"), "roc_auc": float("nan"), "loss": float("nan")}

        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            
            # Loss calculation
            val_loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask]).item()

            # Accuracy calculation
            preds = out[self.data.val_mask].argmax(dim=1)
            targets = self.data.y[self.data.val_mask]
            accuracy = accuracy_score(targets.cpu(), preds.cpu())

            # ROC AUC calculation
            if self.num_classes > 2:
                if len(np.unique(targets.cpu())) > 1 and len(targets) >= 2: 
                    targets_binarized = label_binarize(targets.cpu(), classes=range(self.num_classes))
                    preds_softmax = F.softmax(out[self.data.val_mask], dim=1).cpu().numpy()
                    try:
                        roc_auc = roc_auc_score(targets_binarized, preds_softmax, multi_class='ovr')
                    except ValueError as e:
                        logger.warning(f"Client {self.client_id}: ROC AUC multi-class error: {e}. Unique targets: {np.unique(targets.cpu())}. Returning NaN.")
                else:
                    logger.warning(f"Client {self.client_id}: Only one class or too few samples ({len(targets)}) present in validation targets for multi-class AUC. Unique targets: {np.unique(targets.cpu())}. Returning NaN.")
            elif self.num_classes == 2:
                if len(np.unique(targets.cpu())) > 1 and len(targets) >= 2:
                    preds_softmax_positive = F.softmax(out[self.data.val_mask], dim=1)[:, 1].cpu().numpy()
                    try:
                        roc_auc = roc_auc_score(targets.cpu(), preds_softmax_positive)
                    except ValueError as e:
                        logger.warning(f"Client {self.client_id}: ROC AUC binary error: {e}. Unique targets: {np.unique(targets.cpu())}. Returning NaN.")
                else:
                    logger.warning(f"Client {self.client_id}: Only one class or too few samples ({len(targets)}) present in validation targets for binary AUC. Unique targets: {np.unique(targets.cpu())}. Returning NaN.")
            else:
                logger.warning(f"Client {self.client_id}: Cannot calculate ROC AUC for single class (num_classes={self.num_classes}). Returning NaN.")

        logger.info(f"Client {self.client_id}: Evaluation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        return val_loss, num_examples_val, {"accuracy": accuracy, "roc_auc": roc_auc, "loss": val_loss}