import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class GraphSAGEModel(nn.Module):
    """
    GraphSAGE (Slide 5, 6, 7):
    Neighbor pooling and multi-hop feature aggregation across the transaction network.
    Learns structural fraud patterns from connected accounts.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_layers=2, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr='mean')
        self.fc = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)

    def forward(self, x, edge_index):
        # 1st Hop Message Passing
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 2nd Hop Message Passing
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Classification head
        out = self.fc(h)
        return out

    def get_embeddings(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        return h


class GATModel(nn.Module):
    """
    GAT (Graph Attention Network - Slide 5, 6, 7):
    Applies multi-head self-attention over transfer neighbors.
    Learns to weigh suspicious money-laundering edges higher than innocent high-volume transactions.
    Extracts attention weights for explainability (XAI).
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        self.heads = heads
        # Multi-head attention conv
        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, hidden_channels // 2, heads=1, concat=False, dropout=dropout)
        self.fc = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention_weights=False):
        if return_attention_weights:
            h, (edge_idx_attn, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gat2(h, edge_index)
            h = F.elu(h)
            out = self.fc(h)
            return out, (edge_idx_attn, alpha)
        else:
            h = self.gat1(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gat2(h, edge_index)
            h = F.elu(h)
            out = self.fc(h)
            return out


class FocalLoss(nn.Module):
    """
    Focal loss to handle extreme class imbalance (0.13% fraud rate in PaySim - Slide 4)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
