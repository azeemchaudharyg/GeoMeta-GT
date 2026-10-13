import os
import yaml
import torch
import wandb
import torch.nn as nn
import pickle

from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.metadata_loader import load_metadata
from src.splits import create_id_splits
from src.graph_builder import create_meta_projection, build_graph_dataset

from models.transformer_agnn import HybridTransformerAGNN


# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------
# Load config
# --------------------------
def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
cfg = load_config()


# --------------------------
# Paths
# --------------------------
dataset = cfg["dataset"]

# pkl_path = f"slic/resnet/{dataset}/{dataset}_slic_region_features_{cfg['segmentation']['regions']}.pkl"
pkl_path = f"datasets/region_features.pkl"
metadata_path = f"datasets/{dataset}_metadata.csv"


# --------------------------
# Load data
# --------------------------
print("Loading region features...")
with open(pkl_path, "rb") as f:
    data_dict = pickle.load(f)

print(f"Loaded {len(data_dict)} samples")


# --------------------------
# Load metadata
# --------------------------
metadata_processed, _, meta_feature_dim = load_metadata(metadata_path, data_dict)

# --------------------------
# Data split
# --------------------------
train_ids, val_ids, test_ids = create_id_splits(metadata_processed, test_size=0.2, val_size=0.1, random_state=42)

# --------------------------
# Feature dims
# --------------------------
sample_key = list(data_dict.keys())[0]
cnn_feature_dim = data_dict[sample_key]["region_features"].shape[1]

# --------------------------
# Meta projection
# --------------------------
meta_proj = create_meta_projection(metadata_feature_dim=meta_feature_dim, cnn_feature_dim=cnn_feature_dim, device=device)


# --------------------------
# Build graphs
# --------------------------
train_graphs, val_graphs, test_graphs = build_graph_dataset(
    data_dict=data_dict,
    metadata_processed=metadata_processed,
    meta_proj=meta_proj,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    k=cfg["graph"]["k"],
    device=device
)


print(
    f"Graphs -> Train: {len(train_graphs)}, "
    f"Val: {len(val_graphs)}, "
    f"Test: {len(test_graphs)}"
)


# --------------------------
# DataLoaders
# --------------------------
train_loader = DataLoader(train_graphs, batch_size=cfg["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=cfg["training"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=cfg["training"]["batch_size"], shuffle=False)

# --------------------------
# Model
# --------------------------
model = HybridTransformerAGNN(in_channels=cnn_feature_dim, hidden_channels=cfg["model"]["hidden_channels"],
    num_classes=2,
    heads=cfg["model"]["heads"],
    dropout=cfg["model"]["dropout"]
).to(device)


# --------------------------
# Loss & Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)


# --------------------------
# Logging
# --------------------------
wandb.init(project="R-GNN", name=f"{dataset}-meta-gnn", config=cfg)


# --------------------------
# Train loop
# --------------------------
best_val_acc = 0.0
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)


def evaluate(loader):
    model.eval()
    loss_all = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.edge_geom, batch.batch)

            loss = criterion(out, batch.y)
            loss_all += loss.item()

            preds = out.argmax(dim=1)

            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)

    return loss_all / len(loader), correct / total


for epoch in range(cfg["training"]["epochs"]):

    # ----------------------
    # Train
    # ----------------------
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_geom, batch.batch)

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = out.argmax(dim=1)
        train_correct += (preds == batch.y).sum().item()
        train_total += batch.y.size(0)

    train_acc = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # ----------------------
    # Validation
    # ----------------------
    val_loss, val_acc = evaluate(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    # ----------------------
    # Save best model
    # ----------------------
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        path = os.path.join(save_dir, f"best_gnn_model_{dataset}.pth")

        torch.save(model.state_dict(), path)

        print("Saved best model:", path)


# --------------------------
# Test evaluation
# --------------------------
print("\nLoading best model...")

model.load_state_dict(torch.load(os.path.join(save_dir, f"best_gnn_model.pth"), map_location=device))

model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.edge_geom, batch.batch)

        preds = out.argmax(dim=1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


wandb.finish()