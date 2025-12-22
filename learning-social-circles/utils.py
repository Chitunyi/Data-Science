# util.py
import os
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# t-SNE visualization
def plot_embedding(model, data, ego_id, tag="test"):
    if not hasattr(model, "get_embedding"):
        return

    model.eval()
    with torch.no_grad():
        h = model.get_embedding(data.x, data.edge_index).cpu().numpy()

    if h.shape[0] < 3:
        return

    perp = min(30, h.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )
    z = tsne.fit_transform(h)

    y_true = data.y.cpu().numpy()
    cmap = plt.get_cmap("tab10")

    node_colors = []
    for i in range(len(y_true)):
        circles = np.where(y_true[i] == 1)[0]
        if len(circles) > 0:
            node_colors.append(cmap(circles[0] % 10))
        else:
            node_colors.append("lightgray")

    plt.figure(figsize=(8, 8))
    for i, c in enumerate(node_colors):
        plt.scatter(
            z[i, 0], z[i, 1],
            s=30 if c != "lightgray" else 20,
            c=[c] if c != "lightgray" else c,
            alpha=0.8 if c != "lightgray" else 0.5,
            edgecolors="k" if c != "lightgray" else None,
            linewidth=0.5 if c != "lightgray" else 0
        )

    plt.title(f"t-SNE Embedding (Ego {ego_id}, {tag})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"tsne_ego_{ego_id}_{tag}.png", dpi=150)
    plt.close()

# Data loading
def load_kaggle_ego_graph_global_features(root_dir, ego_id):
    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    if not (os.path.exists(edge_path) and os.path.exists(circle_path)):
        raise FileNotFoundError()

    nodes = set([ego_id])
    raw_edges = []

    with open(edge_path) as f:
        for line in f:
            u, *rest = line.strip().split(":")
            u = int(u)
            nodes.add(u)
            if rest and rest[0]:
                for v in map(int, rest[0].split()):
                    nodes.add(v)
                    raw_edges.append((u, v))

    sorted_nodes = sorted(nodes)
    node_mapper = {gid: i for i, gid in enumerate(sorted_nodes)}

    edge_index = [[], []]
    for u, v in raw_edges:
        ui, vi = node_mapper[u], node_mapper[v]
        edge_index[0] += [ui, vi]
        edge_index[1] += [vi, ui]

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    node_raw_features = {nid: [] for nid in sorted_nodes}
    all_features = set()

    with open(feat_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) in node_mapper:
                feats = parts[1:]
                node_raw_features[int(parts[0])] = feats
                all_features.update(feats)

    sorted_features = sorted(all_features)
    feat_mapper = {f: i for i, f in enumerate(sorted_features)}

    x = torch.zeros((len(sorted_nodes), len(sorted_features)))
    for gid, feats in node_raw_features.items():
        for f in feats:
            x[node_mapper[gid], feat_mapper[f]] = 1.0

    circles = []
    with open(circle_path) as f:
        for line in f:
            _, members = line.strip().split(":")
            circles.append(list(map(int, members.split())))

    y = torch.zeros((len(sorted_nodes), len(circles)))
    for cid, members in enumerate(circles):
        for m in members:
            if m in node_mapper:
                y[node_mapper[m], cid] = 1.0

    idx = torch.randperm(len(sorted_nodes))
    split = int(0.8 * len(sorted_nodes))
    train_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
    train_mask[idx[:split]] = True
    test_mask[idx[split:]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        num_nodes=len(sorted_nodes),
        num_features=x.size(1),
        num_circles=y.size(1),
        sorted_features=sorted_features
    )

def induced_subgraph_data(full_data, node_mask):
    edge_index, _ = subgraph(
        node_mask,
        full_data.edge_index,
        relabel_nodes=True,
        num_nodes=full_data.num_nodes
    )

    return Data(
        x=full_data.x[node_mask],
        edge_index=edge_index,
        y=full_data.y[node_mask],
        num_nodes=int(node_mask.sum()),
        num_features=full_data.num_features,
        num_circles=full_data.num_circles,
        sorted_features=full_data.sorted_features
    )


def filter_features_by_support_train_only(train_data, min_count=2, max_frac=1.0):
    counts = train_data.x.sum(dim=0)
    N = train_data.x.size(0)
    keep = (counts >= min_count) & (counts <= max_frac * N)
    idx = keep.nonzero(as_tuple=False).view(-1)

    if len(idx) == 0:
        idx = torch.arange(train_data.x.size(1))

    names = [train_data.sorted_features[i] for i in idx.tolist()]
    return idx, names


def apply_feature_filter(data, keep_idx, new_names):
    return Data(
        x=data.x[:, keep_idx],
        edge_index=data.edge_index,
        y=data.y,
        num_nodes=data.num_nodes,
        num_features=len(keep_idx),
        num_circles=data.num_circles,
        sorted_features=new_names
    )

# Metrics
def compute_auc_scores(y_true, y_score):
    try:
        micro = roc_auc_score(y_true, y_score, average="micro")
        macro = roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        micro = macro = 0.5

    per_class = []
    for k in range(y_true.shape[1]):
        if len(np.unique(y_true[:, k])) == 2:
            per_class.append(roc_auc_score(y_true[:, k], y_score[:, k]))
        else:
            per_class.append(0.5)

    return micro, macro, per_class


def compute_ber_score(y_true, y_pred):
    bers = []
    for k in range(y_true.shape[1]):
        if len(np.unique(y_true[:, k])) < 2:
            bers.append(0.5)
            continue
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, k], y_pred[:, k], labels=[0, 1]
        ).ravel()
        bers.append(0.5 * (
            fn / (tp + fn + 1e-9) +
            fp / (tn + fp + 1e-9)
        ))
    return float(np.mean(bers))

# Forward wrapper
def forward_logits(model, data, need_edge):
    out = model(data.x, data.edge_index) if need_edge else model(data.x)
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None
    return out, None


import numpy as np

def extract_top1_importance_per_circle(explanations):
    """
    return:
        List[float] : top-1 |importance| for each circle
    """
    vals = []

    for circle, feats in explanations.items():
        if len(feats) == 0:
            continue
        vals.append(abs(feats[0][1]))

    return vals