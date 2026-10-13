import torch
import torch.nn as nn

from torch_geometric.data import Data
from graphs_utils import remap_segments_to_contiguous, superpixel_centroids_from_segments, normalize_pos_by_image
from graphs_utils import knn_graph_manual, edge_geom_from_pos, normalize_edge_geom
from metadata_loader import get_metadata_row

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def build_graph(fname, f_data, meta_proj, metadata_processed, k=6, device=device):
    """
    Build graph consisting of:

    Nodes:
        - Superpixel region nodes
        - One metadata node

    Edges:
        - Region ↔ Region (kNN)
        - Metadata ↔ Region

    Returns
    -------
    torch_geometric.data.Data
    """

    fname_norm = fname.lower().replace(".jpg", "").replace(".png", "").strip()

    row = get_metadata_row(metadata_processed, fname_norm)

    if row is None:
        return None

    # --------------------------------------------------
    # Region features
    # --------------------------------------------------
    features = f_data["region_features"]

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    else:
        features = features.float()

    # --------------------------------------------------
    # Segments
    # --------------------------------------------------
    segments = f_data["segments"]

    if not isinstance(segments, torch.Tensor):
        segments = torch.tensor(segments)

    segments = segments.long()
    label = int(f_data["label"])

    # --------------------------------------------------
    # Remap segment labels
    # --------------------------------------------------
    segments_remap, num_nodes_from_segments = (remap_segments_to_contiguous(segments))

    num_region_nodes = features.shape[0]

    if num_nodes_from_segments != num_region_nodes:

        print(
            f"Warning: "
            f"{fname_norm} "
            f"features={num_region_nodes} "
            f"segments={num_nodes_from_segments}"
        )

        return None

    H, W = segments_remap.shape

    # --------------------------------------------------
    # Region positions
    # --------------------------------------------------
    pos_region = superpixel_centroids_from_segments(segments_remap, num_nodes=num_region_nodes)
    pos_region = normalize_pos_by_image(pos_region, H, W)

    # --------------------------------------------------
    # Region-region edges
    # --------------------------------------------------
    k_eff = min(k, max(num_region_nodes - 1, 1))

    edge_index_rr = knn_graph_manual(pos_region, k=k_eff)

    if edge_index_rr.numel() == 0:
        return None

    edge_geom_rr = edge_geom_from_pos(pos_region, edge_index_rr)
    edge_geom_rr = normalize_edge_geom(edge_geom_rr)

    # --------------------------------------------------
    # Metadata node
    # --------------------------------------------------
    meta_values = row.drop(columns=["isic_id"]).values

    meta_node = torch.tensor(meta_values, dtype=torch.float32).to(device)

    with torch.no_grad():
        meta_node = meta_proj(meta_node)

    # --------------------------------------------------
    # Combine node features
    # --------------------------------------------------
    x_all = torch.cat([features, meta_node.cpu()], dim=0)
    meta_idx = x_all.size(0) - 1

    # --------------------------------------------------
    # Metadata node position
    # --------------------------------------------------
    center_xy = pos_region.mean(dim=0)

    pos_meta = center_xy.view(1, 2)

    pos_all = torch.cat([pos_region, pos_meta], dim=0)

    # --------------------------------------------------
    # Metadata-region edges
    # --------------------------------------------------
    region_idx = torch.arange(num_region_nodes, dtype=torch.long)

    meta_to_region = torch.stack([torch.full((num_region_nodes,), meta_idx, dtype=torch.long),region_idx], dim=0)

    region_to_meta = meta_to_region.flip(0)

    edge_index_meta = torch.cat([meta_to_region, region_to_meta], dim=1)

    edge_geom_meta = edge_geom_from_pos(pos_all, edge_index_meta)

    edge_geom_meta = normalize_edge_geom(edge_geom_meta)

    # --------------------------------------------------
    # Final graph
    # --------------------------------------------------
    edge_index = torch.cat([edge_index_rr, edge_index_meta], dim=1)
    edge_geom = torch.cat([edge_geom_rr, edge_geom_meta], dim=0)

    y = torch.tensor([label], dtype=torch.long)

    graph = Data(x=x_all, pos=pos_all, edge_index=edge_index, edge_geom=edge_geom, y=y)

    graph.isic_id = fname_norm

    return graph


def build_graph_dataset(data_dict, metadata_processed, meta_proj, train_ids, val_ids, test_ids, k=6, device=device):
    """
    Build all graphs and distribute
    into train/val/test sets.
    """

    train_graphs = []
    val_graphs = []
    test_graphs = []

    for fname, f_data in data_dict.items():
        graph = build_graph(
            fname=fname,
            f_data=f_data,
            meta_proj=meta_proj,
            metadata_processed=metadata_processed,
            k=k,
            device=device
        )

        if graph is None:
            continue

        if graph.isic_id in train_ids:
            train_graphs.append(graph)

        elif graph.isic_id in val_ids:
            val_graphs.append(graph)

        elif graph.isic_id in test_ids:
            test_graphs.append(graph)

    print(
        f"Built "
        f"{len(train_graphs)} train, "
        f"{len(val_graphs)} val, "
        f"{len(test_graphs)} test graphs."
    )

    return train_graphs, val_graphs, test_graphs


def create_meta_projection(metadata_feature_dim, cnn_feature_dim, device=device):
    return nn.Linear(metadata_feature_dim, cnn_feature_dim).to(device)