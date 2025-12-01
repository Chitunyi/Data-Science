import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score, normalized_mutual_info_score, adjusted_rand_score
)
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


# ============================================================
#  Data Loader
# ============================================================

def load_kaggle_ego_graph_global_features(root_dir: str, ego_id: int):
    # print(f"Loading Ego-net: {ego_id}...")

    # Paths
    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    if not os.path.exists(edge_path) or not os.path.exists(circle_path):
        raise FileNotFoundError(f"Missing file for ego {ego_id}")

    # Read edges
    nodes = set([ego_id])
    raw_edges = []
    with open(edge_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            u = int(parts[0])
            neighbors = parts[1].strip().split() if len(parts) > 1 else []
            neighbors = [int(x) for x in neighbors]

            nodes.add(u)
            for v in neighbors:
                nodes.add(v)
                raw_edges.append((u, v))

    sorted_nodes = sorted(list(nodes))
    node_mapper = {gid: idx for idx, gid in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    # Build edge index
    edge_index = [[], []]
    for u, v in raw_edges:
        if u in node_mapper and v in node_mapper:
            edge_index[0] += [node_mapper[u], node_mapper[v]]
            edge_index[1] += [node_mapper[v], node_mapper[u]]

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Read features
    node_raw_features = {node: [] for node in sorted_nodes}
    all_features = set()

    with open(feat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            nid = int(parts[0])
            feats = parts[1:]

            if nid in node_mapper:
                node_raw_features[nid] = feats
                for ft in feats:
                    all_features.add(ft)

    sorted_features = sorted(list(all_features))
    feat_mapper = {ft: idx for idx, ft in enumerate(sorted_features)}
    num_features = len(sorted_features)

    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for gid, feats in node_raw_features.items():
        for ft in feats:
            x[node_mapper[gid], feat_mapper[ft]] = 1.0

    # Read circles
    circles = []
    with open(circle_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 2:
                continue
            members = [int(n) for n in parts[1].split()]
            circles.append(members)

    num_circles = len(circles)
    y = torch.zeros((num_nodes, num_circles), dtype=torch.float)

    for cid, members in enumerate(circles):
        for mid in members:
            if mid in node_mapper:
                y[node_mapper[mid], cid] = 1.0

    # Train/Test split
    idx = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:split]] = True
    test_mask[idx[split:]] = True

    # print(f"Ego {ego_id}: Nodes={num_nodes}, Edges={edge_index.size(1)}, "
    #       f"Features={num_features}, Circles={num_circles}")

    return Data(
        x=x, edge_index=edge_index, y=y,
        num_nodes=num_nodes, num_features=num_features, num_circles=num_circles,
        train_mask=train_mask, test_mask=test_mask
    )


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
        x = self.conv2(x, edge_index)
        return x


def gae_loss(z, edge_index, num_nodes, num_neg_samples=None):
    """_summary_
    calculate the dot product of the positive pair and negative pair.
    
    Args:
        z (_type_): _description_
        edge_index (_type_): _description_
        num_nodes (_type_): _description_
        num_neg_samples (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1) # choose the same numbers as the positive pair

    # edge_index.shape = [2, edge_pair_counts * 2]
    src, dst = edge_index
    pos_score = (z[src] * z[dst]).sum(dim=1) ## dot product
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

    # negative pair : torch.randint (low, high, tensor_size)
    neg_src = torch.randint(0, num_nodes, (num_neg_samples,), device=z.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,), device=z.device)
    
    neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

    return pos_loss + neg_loss


def train_gae_encoder(encoder, data, epochs=200, lr=0.01, verbose=False):
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    for epoch in range(epochs):
        opt.zero_grad()
        z = encoder(data.x, data.edge_index)
        loss = gae_loss(z, data.edge_index, data.num_nodes)
        loss.backward()
        opt.step()

    return encoder

# ============================================================
#  Evaluation / Clustering
# ============================================================

def reduce_multilabel_to_single(y_multi):
    y_multi = np.asarray(y_multi)
    count = y_multi.sum(axis=1)
    return np.where(count > 0, y_multi.argmax(axis=1), -1)


def hungarian_match_and_micro_f1(y_true_single, cluster_labels):
    mask = y_true_single != -1
    y_true = y_true_single[mask]
    y_pred = cluster_labels[mask]

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0

    cont = contingency_matrix(y_true, y_pred)
    cost = -cont.T
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}

    mapped_pred = np.array([mapping.get(c, -1) for c in y_pred])
    mask2 = mapped_pred != -1

    if np.unique(mapped_pred[mask2]).size < 2:
        return 0.0

    return f1_score(y_true[mask2], mapped_pred[mask2], average="micro")


def evaluate_clustering_against_circles(cluster_labels, y_multi):
    y_single = reduce_multilabel_to_single(y_multi)
    mask = y_single != -1

    y_t = y_single[mask]
    y_p = cluster_labels[mask]

    if len(np.unique(y_t)) < 2 or len(np.unique(y_p)) < 2:
        return {"micro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    nmi = normalized_mutual_info_score(y_t, y_p)
    ari = adjusted_rand_score(y_t, y_p)
    micro_f1 = hungarian_match_and_micro_f1(y_single, cluster_labels)

    return {"micro_f1": micro_f1, "nmi": nmi, "ari": ari}


def run_gae_clustering_single_ego(data, device, emb_dim=64, hidden_channels=64,
                                  epochs=200, lr=0.01, n_clusters=None):

    data = data.to(device)

    encoder = GAEEncoder(data.num_features, hidden_channels, emb_dim).to(device)
    train_gae_encoder(encoder, data, epochs=epochs, lr=lr)

    encoder.eval()
    z = encoder(data.x, data.edge_index).cpu().detach().numpy()

    if n_clusters is None:
        n_clusters = data.num_circles

    if n_clusters <= 1:
        return {"micro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(z)
    y_multi = data.y.cpu().numpy()

    return evaluate_clustering_against_circles(pred, y_multi)


# ============================================================
#  Full Experiment Loop
# ============================================================

def run_gae_clustering_experiment():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)

    egonet_files = glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))
    ego_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in egonet_files])

    print(f"Found {len(ego_ids)} ego-nets.")
    print("====== Starting GAE + Clustering Evaluation ======")

    results = {"ego_id": [], "micro_f1": [], "nmi": [], "ari": []}

    for ego_id in ego_ids:
        # print(f"\n--- Processing Ego {ego_id} ---")

        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except:
            # print(f"Skipping ID {ego_id}")
            continue

        metrics = run_gae_clustering_single_ego(data, device)
        # print(f"Ego {ego_id} -> "
        #       f"Micro-F1: {metrics['micro_f1']:.4f}, "
        #       f"NMI: {metrics['nmi']:.4f}, "
        #       f"ARI: {metrics['ari']:.4f}")

        results["ego_id"].append(ego_id)
        results["micro_f1"].append(metrics["micro_f1"])
        results["nmi"].append(metrics["nmi"])
        results["ari"].append(metrics["ari"])

    # # Final Summary
    # print("\n============= FINAL REPORT =============")
    # print("Ego ID | Micro-F1 |   NMI   |   ARI")
    # print("--------------------------------------")

    # for eid, f1, nmi, ari in zip(
    #         results["ego_id"], results["micro_f1"], results["nmi"], results["ari"]):
    #     print(f"{eid:<6} | {f1:<9.4f} | {nmi:<7.4f} | {ari:<7.4f}")

    if len(results["micro_f1"]) > 0:
        print("\n------ Averages ------")
        print(f"Avg Micro-F1: {np.mean(results['micro_f1']):.4f}")
        print(f"Avg NMI:      {np.mean(results['nmi']):.4f}")
        print(f"Avg ARI:      {np.mean(results['ari']):.4f}")

    print("=========================================")


# ============================================================
#  Run
# ============================================================

if __name__ == "__main__":
    run_gae_clustering_experiment()
