import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, AGNNConv, global_mean_pool

from models.edge_mlp import EdgeMLP


class HybridTransformerAGNN(nn.Module):
    
    def __init__(self, in_channels, hidden_channels=512, num_classes=2, heads=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.edge_mlp = EdgeMLP(in_dim=2, hidden_dim=32, out_dim=2, dropout=dropout)

        self.transformer1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=2
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.agnn1 = AGNNConv(requires_grad=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels)


        self.lin1 = nn.Linear(hidden_channels, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)

        self.lin2 = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, edge_geom, batch, return_embedding=False):
        edge_attr = self.edge_mlp(edge_geom)

        x = self.transformer1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.agnn1(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embed = global_mean_pool(x, batch)

        if return_embedding:
            return graph_embed

        x = self.lin1(graph_embed)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)

        return x