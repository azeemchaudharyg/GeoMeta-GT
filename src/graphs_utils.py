import math
import torch


def remap_segments_to_contiguous(segments: torch.Tensor):
    """
    Remap arbitrary superpixel labels to contiguous labels:
    e.g. [3, 5, 10] -> [0, 1, 2]
    """

    flat = segments.view(-1).long()
    uniq = torch.unique(flat)

    new_ids = torch.arange(uniq.numel(), device=segments.device, dtype=torch.long)
    remapped = torch.empty_like(flat)

    for old, new in zip(uniq.tolist(), new_ids.tolist()):
        remapped[flat == old] = new

    remapped = remapped.view_as(segments)

    return remapped, int(uniq.numel())


def knn_graph_manual(pos: torch.Tensor, k: int):
    """
    Build undirected k-NN graph manually.

    Parameters
    ----------
    pos : Tensor [N,2]
        node coordinates

    k : int
        number of neighbours

    Returns
    -------
    edge_index : Tensor [2,E]
    """

    N = pos.size(0)

    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))

    k_eff = min(k, N - 1)

    nn_idx = dist.topk(k=k_eff, largest=False).indices

    src = (torch.arange(N, device=pos.device).view(-1, 1).repeat(1, k_eff))

    dst = nn_idx

    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)

    reverse_edges = edge_index.flip(0)

    edge_index = torch.cat([edge_index, reverse_edges], dim=1)

    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def superpixel_centroids_from_segments(segments: torch.Tensor, num_nodes: int):
    """
    Compute bounding-box centroid
    for each superpixel region.

    Returns
    -------
    pos : Tensor [num_nodes,2]
        [cx, cy]
    """

    segments = segments.long()

    H, W = segments.shape

    device = segments.device

    pos = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)

    for region_id in range(num_nodes):
        ys, xs = torch.where(segments == region_id)

        if ys.numel() == 0:
            pos[region_id] = torch.tensor([0.0, 0.0], device=device)
            continue

        xmin = xs.min().float()
        xmax = xs.max().float()

        ymin = ys.min().float()
        ymax = ys.max().float()

        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        pos[region_id, 0] = cx
        pos[region_id, 1] = cy

    return pos


def normalize_pos_by_image(pos: torch.Tensor, H: int, W: int):
    """
    Normalize coordinates
    using image diagonal length.
    """

    diag = math.sqrt((H - 1) ** 2 + (W - 1) ** 2)

    return pos / max(diag, 1e-8)


def edge_geom_from_pos(pos: torch.Tensor, edge_index: torch.Tensor, eps: float = 1e-8):
    """
    Compute geometric edge features.

    Returns
    -------
    edge_geom : [E,2]

    column 0 = distance
    column 1 = angle
    """

    src = edge_index[0]
    dst = edge_index[1]

    delta = pos[dst] - pos[src]

    dx = delta[:, 0]
    dy = delta[:, 1]

    dist = torch.sqrt(dx * dx + dy * dy + eps)

    angle = torch.atan2(dy, dx)

    edge_geom = torch.stack([dist, angle], dim=1)

    return edge_geom


def normalize_edge_geom(edge_geom: torch.Tensor):
    """
    Normalize edge geometry.

    Distance already normalized if
    node coordinates were normalized.

    Angle:
    [-pi,+pi] -> [-1,+1]
    """

    out = edge_geom.clone()

    out[:, 1] = out[:, 1] / math.pi

    return out