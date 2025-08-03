import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import logging
from utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class SAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_embeddings=False):
        # Handle cases where edge_index might be None or empty due to small client datasets
        if edge_index is None or edge_index.numel() == 0:
            h = F.relu(self.bn1(self.conv1.lin_l(x)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.bn2(self.conv2.lin_l(h))) + h
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.bn3(self.conv3.lin_l(h))) + h
            h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            h = F.relu(self.bn1(self.conv1(x, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.bn2(self.conv2(h, edge_index))) + h
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.bn3(self.conv3(h, edge_index))) + h
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        if return_embeddings:
            return h

        return self.out(h)