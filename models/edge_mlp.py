import torch.nn as nn


class EdgeMLP(nn.Module):
    """
    MLP to transform geometric edge features:
    [distance, angle] -> learned edge embedding
    """

    def __init__(self, in_dim=2, hidden_dim=32, out_dim=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, edge_geom):
        return self.net(edge_geom)