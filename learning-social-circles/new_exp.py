import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import f1_score
from torch_geometric.data import Data
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

class CESNAOverlappingCommunities(nn.Module):
    """
    Simplified CESNA-style model:
      - 每個 circle k 有一組 feature weight theta_k (size = num_features)
      - 每個 node u 對 circle k 的 membership:
            m[u,k] = sigmoid( x[u] · theta_k + bias_k )
      - edge (u,v) 的 logit:
            s(u,v) = edge_scale * sum_k m[u,k] * m[v,k] + edge_bias

    用 BCE loss 讓 s(u,v) 對正樣本邊接近 1、對負樣本邊接近 0。
    """
    def __init__(self, num_features, num_communities, init_std=0.01):
        super().__init__()
        # theta: [K, F]
        self.theta = nn.Parameter(
            torch.randn(num_communities, num_features) * init_std
        )
        # bias per community: [K]
        self.bias = nn.Parameter(torch.zeros(num_communities))

        # global scale & bias for edge logits
        self.edge_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_bias = nn.Parameter(torch.tensor(0.0))

    def memberships(self, x):
        """
        x: [N, F]
        return m: [N, K] in (0,1)
        """
        # [N, K] = [N, F] @ [F, K]
        logits = x @ self.theta.t() + self.bias  # broadcast bias: [K]
        return torch.sigmoid(logits)

    def edge_logits(self, memberships, edge_pairs):
        """
        memberships: [N, K]
        edge_pairs: [B, 2] (node indices)
        return logits: [B]
        """
        u = edge_pairs[:, 0]
        v = edge_pairs[:, 1]
        m_u = memberships[u]     # [B, K]
        m_v = memberships[v]     # [B, K]
        scores = (m_u * m_v).sum(dim=1)  # [B]
        return self.edge_scale * scores + self.edge_bias

    def forward(self, x, edge_pairs=None):
        m = self.memberships(x)
        if edge_pairs is None:
            return m
        logits = self.edge_logits(m, edge_pairs)
        return logits, m


# ============================================================
#  Utils for CESNA: edges & negative sampling
# ============================================================

def build_undirected_edge_list(edge_index: torch.Tensor) -> np.ndarray:
    """
    將 PyG edge_index [2, E] 轉成 undirected edge list [E_u, 2] 並去重。
    你的 loader 已經把邊補成雙向，這裡再 sort+unique 一次。
    """
    edge_index = edge_index.detach().cpu().numpy()
    src, dst = edge_index
    edges = np.stack([src, dst], axis=1)      # [E, 2]
    edges = np.sort(edges, axis=1)           # 保證 (min,max)
    edges = np.unique(edges, axis=0)         # 去重
    return edges


def sample_negative_edges(num_nodes: int,
                          num_samples: int,
                          existing_edges_set: set) -> np.ndarray:
    """
    從所有 node pair 中 uniform 抽取 non-edges 當負樣本。
    避開 existing_edges_set (undirected, (u,v) with u<v)。
    """
    neg_edges = set()
    tries = 0
    max_tries = num_samples * 20

    while len(neg_edges) < num_samples and tries < max_tries:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            tries += 1
            continue
        if u > v:
            u, v = v, u
        if (u, v) in existing_edges_set:
            tries += 1
            continue
        neg_edges.add((u, v))
        tries += 1

    # fallback: 如果還沒抽滿就不再避 existing_edges_set（影響很小）
    while len(neg_edges) < num_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        neg_edges.add((u, v))

    return np.array(list(neg_edges), dtype=np.int64)


# ============================================================
#  CESNA Training
# ============================================================

def train_cesna_model(data,
                      device,
                      num_communities=None,
                      epochs=200,
                      lr=0.01,
                      batch_size_edges=2048,
                      neg_pos_ratio=1.0,
                      verbose=False):
    """
    Unsupervised training:
      - 用 node features x + edge_index 來訓練 CESNAOverlappingCommunities
      - y 只在 evaluation 時使用，不參與訓練
    """
    if num_communities is None:
        # 最簡單：先用 ground-truth 圈數當 K（只當 hyperparam，用不到 labels）
        num_communities = data.num_circles

    x = data.x.to(device)  # [N, F]
    num_nodes = data.num_nodes

    model = CESNAOverlappingCommunities(
        num_features=data.num_features,
        num_communities=num_communities
    ).to(device)

    edges = build_undirected_edge_list(data.edge_index)  # [E_u, 2]
    num_edges = edges.shape[0]
    existing_edges_set = {(int(u), int(v)) for u, v in edges}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # shuffle positive edges
        perm = np.random.permutation(num_edges)
        edges_shuffled = edges[perm]

        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_edges, batch_size_edges):
            end = min(start + batch_size_edges, num_edges)
            pos_batch_np = edges_shuffled[start:end]
            if pos_batch_np.shape[0] == 0:
                continue

            # 負樣本數量 = neg_pos_ratio * 正樣本數，至少 1
            num_neg = max(int(pos_batch_np.shape[0] * neg_pos_ratio), 1)
            neg_batch_np = sample_negative_edges(
                num_nodes=num_nodes,
                num_samples=num_neg,
                existing_edges_set=existing_edges_set
            )

            pos_batch = torch.from_numpy(pos_batch_np).long().to(device)
            neg_batch = torch.from_numpy(neg_batch_np).long().to(device)

            optimizer.zero_grad()

            # memberships: [N, K]
            memberships = model.memberships(x)

            # positive edge logits
            pos_logits = model.edge_logits(memberships, pos_batch)  # [B_pos]
            pos_target = torch.ones_like(pos_logits)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_target)

            # negative edge logits
            neg_logits = model.edge_logits(memberships, neg_batch)  # [B_neg]
            neg_target = torch.zeros_like(neg_logits)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_target)

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if verbose and num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"[CESNA] Epoch {epoch:03d}/{epochs}  Loss = {avg_loss:.4f}")

    return model


# ============================================================
#  Membership → Multi-label prediction & evaluation
# ============================================================

def infer_cesna_memberships(model, data, device, threshold=0.5):
    """
    把連續 membership m[u,k] ∈ (0,1) threshold 成 multi-label {0,1}。
    """
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        m = model.memberships(x)  # [N, K]
    m_np = m.detach().cpu().numpy()
    y_pred = (m_np >= threshold).astype(np.int64)
    return y_pred


def evaluate_multilabel_circles(y_true_multi, y_pred_multi):
    """
    Multi-label F1 (micro / macro).
    y_*: numpy array, shape [N, K]
    """
    y_true_multi = np.asarray(y_true_multi)
    y_pred_multi = np.asarray(y_pred_multi)
    assert y_true_multi.shape == y_pred_multi.shape

    micro = f1_score(y_true_multi.ravel(), y_pred_multi.ravel(), average="micro")
    macro = f1_score(y_true_multi.ravel(), y_pred_multi.ravel(), average="macro")

    return {"micro_f1": float(micro), "macro_f1": float(macro)}


def run_cesna_single_ego(data,
                         device,
                         num_communities=None,
                         epochs=200,
                         lr=0.01,
                         batch_size_edges=2048,
                         neg_pos_ratio=1.0,
                         threshold=0.5,
                         verbose=False):
    """
    在單一 ego graph 上跑 CESNA-style overlapping community detection，
    回傳 multi-label F1。
    """
    data = data.to(device)

    model = train_cesna_model(
        data=data,
        device=device,
        num_communities=num_communities,
        epochs=epochs,
        lr=lr,
        batch_size_edges=batch_size_edges,
        neg_pos_ratio=neg_pos_ratio,
        verbose=verbose,
    )

    y_pred = infer_cesna_memberships(model, data, device, threshold=threshold)
    y_true = data.y.cpu().numpy()

    metrics = evaluate_multilabel_circles(y_true, y_pred)
    return metrics

def run_cesna_experiment():
    """
    全部 ego-nets 上跑 CESNA-style overlapping community detection。
    """
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)

    egonet_files = glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))
    ego_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in egonet_files])

    print(f"Found {len(ego_ids)} ego-nets.")
    print("====== Starting CESNA-style Overlapping Community Evaluation ======")

    results = {"ego_id": [], "micro_f1": [], "macro_f1": []}

    for ego_id in ego_ids:
        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except Exception as e:
            # print(f"Skipping ID {ego_id} due to error: {e}")
            continue

        metrics = run_cesna_single_ego(
            data,
            device,
            num_communities=None,   # 預設用 data.num_circles
            epochs=200,
            lr=0.01,
            batch_size_edges=2048,
            neg_pos_ratio=1.0,
            threshold=0.5,
            verbose=False,
        )

        print(f"Ego {ego_id} -> "
              f"Micro-F1: {metrics['micro_f1']:.4f}, "
              f"Macro-F1: {metrics['macro_f1']:.4f}")

        results["ego_id"].append(ego_id)
        results["micro_f1"].append(metrics["micro_f1"])
        results["macro_f1"].append(metrics["macro_f1"])

    if len(results["micro_f1"]) > 0:
        print("\n------ CESNA Averages ------")
        print(f"Avg Micro-F1: {np.mean(results['micro_f1']):.4f}")
        print(f"Avg Macro-F1: {np.mean(results['macro_f1']):.4f}")

    print("=========================================")

if __name__ == "__main__":
    run_cesna_experiment()