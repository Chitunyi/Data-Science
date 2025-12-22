import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import glob

# ============================================================
#  Utility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def filter_features_by_support(data, min_count=2, max_frac=1.0):
    """Filter features based on occurrence frequency across nodes."""
    x = data.x
    counts = x.sum(dim=0)
    N = x.size(0)
    max_count = max_frac * N

    keep_mask = (counts >= min_count) & (counts <= max_count)
    keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)

    if keep_indices.numel() == 0:
        return data

    x_new = x[:, keep_indices]
    new_feat_names = [data.sorted_features[i] for i in keep_indices.tolist()] if hasattr(data, 'sorted_features') else []

    return Data(
        x=x_new, edge_index=data.edge_index, y=data.y,
        num_nodes=data.num_nodes, num_features=x_new.size(1),
        num_circles=data.num_circles, train_mask=data.train_mask,
        test_mask=data.test_mask, sorted_features=new_feat_names
    )

# ============================================================
#  Data Loader
# ============================================================

def load_kaggle_ego_graph_global_features(root_dir: str, ego_id: int):
    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    if not os.path.exists(edge_path) or not os.path.exists(circle_path):
        raise FileNotFoundError(f"Missing files for ego {ego_id}")

    # Build graph structure
    nodes = {ego_id}
    raw_edges = []
    with open(edge_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            u = int(parts[0])
            neighbors = [int(x) for x in parts[1].strip().split()] if len(parts) > 1 else []
            nodes.add(u)
            for v in neighbors:
                nodes.add(v)
                raw_edges.append((u, v))

    sorted_nodes = sorted(list(nodes))
    node_mapper = {gid: idx for idx, gid in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    edge_index = [[], []]
    for u, v in raw_edges:
        if u in node_mapper and v in node_mapper:
            edge_index[0] += [node_mapper[u], node_mapper[v]]
            edge_index[1] += [node_mapper[v], node_mapper[u]]
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Process node features
    node_raw_features = {node: [] for node in sorted_nodes}
    all_features = set()
    with open(feat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            nid = int(parts[0])
            if nid in node_mapper:
                feats = parts[1:]
                node_raw_features[nid] = feats
                all_features.update(feats)

    sorted_features = sorted(list(all_features))
    feat_mapper = {ft: idx for idx, ft in enumerate(sorted_features)}
    x = torch.zeros((num_nodes, len(sorted_features)), dtype=torch.float)
    for gid, feats in node_raw_features.items():
        for ft in feats:
            x[node_mapper[gid], feat_mapper[ft]] = 1.0

    # Load ground truth circles
    circles = []
    with open(circle_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            circles.append([int(n) for n in parts[1].split()])

    y = torch.zeros((num_nodes, len(circles)), dtype=torch.float)
    for cid, members in enumerate(circles):
        for mid in members:
            if mid in node_mapper:
                y[node_mapper[mid], cid] = 1.0

    # Train/Test masks
    idx = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    train_mask, test_mask = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:split]], test_mask[idx[split:]] = True, True

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, 
                num_features=x.size(1), num_circles=len(circles),
                train_mask=train_mask, test_mask=test_mask, sorted_features=sorted_features)

# ============================================================
#  GAE Model
# ============================================================

class GAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, emb_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def gae_loss(z, edge_index, num_nodes):
    src, dst = edge_index
    pos_score = (z[src] * z[dst]).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

    neg_src = torch.randint(0, num_nodes, (edge_index.size(1),), device=z.device)
    neg_dst = torch.randint(0, num_nodes, (edge_index.size(1),), device=z.device)
    neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

    return pos_loss + neg_loss

def train_gae_encoder(encoder, data, epochs=200, lr=0.01):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = gae_loss(encoder(data.x, data.edge_index), data.edge_index, data.num_nodes)
        loss.backward()
        opt.step()
    return encoder

# ============================================================
#  Evaluation
# ============================================================

def reduce_multilabel_to_single(y_multi):
    """Maps multi-label nodes to the first active class, or -1 if no class."""
    y_multi = np.asarray(y_multi)
    return np.where(y_multi.sum(axis=1) > 0, y_multi.argmax(axis=1), -1)

def hungarian_match_and_micro_f1(y_true_single, cluster_labels):
    """Calculates Micro-F1 using Hungarian matching for cluster-to-class alignment."""
    mask = y_true_single != -1
    y_t, y_p = y_true_single[mask], cluster_labels[mask]

    if len(np.unique(y_t)) < 2 or len(np.unique(y_p)) < 2: return 0.0

    # Map -1 noise to a high integer for contingency matrix compatibility
    y_p_safe = np.where(y_p == -1, 999999, y_p)
    cont = contingency_matrix(y_t, y_p_safe)
    row_ind, col_ind = linear_sum_assignment(-cont.T)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    
    mapped_preds = np.array([mapping.get(val, -1) for val in y_p_safe])
    return f1_score(y_t, mapped_preds, average="micro", labels=np.unique(y_t))

def evaluate_clustering(cluster_labels, y_multi):
    y_single = reduce_multilabel_to_single(y_multi)
    mask = y_single != -1
    y_t, y_p = y_single[mask], cluster_labels[mask]

    if len(y_t) == 0: return {"micro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    return {
        "micro_f1": hungarian_match_and_micro_f1(y_single, cluster_labels),
        "nmi": normalized_mutual_info_score(y_t, y_p),
        "ari": adjusted_rand_score(y_t, y_p)
    }

def run_single_ego(data, device):
    data = data.to(device)
    encoder = train_gae_encoder(GAEEncoder(data.num_features, 64, 64).to(device), data)
    
    encoder.eval()
    z = encoder(data.x, data.edge_index).cpu().detach().numpy()

    # DBSCAN: eps=0.3, cosine distance allows for noise detection (-1)
    pred = DBSCAN(eps=0.1, min_samples=3, metric='cosine').fit_predict(z)
    return evaluate_clustering(pred, data.y.cpu().numpy())

# ============================================================
#  Main Loop
# ============================================================

def run_experiment():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.dirname(os.path.abspath(__file__))

    ego_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))])
    print(f"Found {len(ego_ids)} ego-nets. Starting GAE + DBSCAN...")

    results = {"f1": [], "nmi": [], "ari": []}
    for ego_id in ego_ids:
        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
            data = filter_features_by_support(data, min_count=2, max_frac=1.0) # Remove rare features
            metrics = run_single_ego(data, device)
            for k, v in zip(results.keys(), metrics.values()): results[k].append(v)
        except Exception: continue

    if results["f1"]:
        print(f"\nAvg Micro-F1: {np.mean(results['f1']):.4f}")
        print(f"Avg NMI:      {np.mean(results['nmi']):.4f}")
        print(f"Avg ARI:      {np.mean(results['ari']):.4f}")

if __name__ == "__main__":
    run_experiment()