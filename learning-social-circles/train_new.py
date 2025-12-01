import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, roc_auc_score
import glob
import ipdb

# -----------------------------
# Utility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data Loader
# -----------------------------
def load_kaggle_ego_graph_global_features(root_dir, ego_id):
    print(f"Loading Ego-net: {ego_id}...")

    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    if not (os.path.exists(edge_path) and os.path.exists(circle_path)):
        raise FileNotFoundError(f"Missing egonet or circle file for ego {ego_id}")

    # -----------------------------
    # Read Edges
    # -----------------------------
    nodes = set([ego_id])
    raw_edges = []

    with open(edge_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            u = int(parts[0])
            neighbors = []
            if len(parts) > 1 and parts[1].strip() != "":
                neighbors = [int(x) for x in parts[1].strip().split()] ## [846, 730, 739...]

            nodes.add(u)
            for v in neighbors:
                nodes.add(v)
                raw_edges.append((u, v))
                
                
    # --- ID Mapping (Global ID -> Local Index 0..N-1) ---
    sorted_nodes = sorted(list(nodes))
    node_mapper = {gid: idx for idx, gid in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    edge_index = [[], []]
    for u, v in raw_edges:
        if u in node_mapper and v in node_mapper:
            # Add both directions for undirected graph
            ui, vi = node_mapper[u], node_mapper[v]
            edge_index[0].append(ui)
            edge_index[1].append(vi)
            edge_index[0].append(vi)
            edge_index[1].append(ui)

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # --- Features ---
    # Turn feature into digits, build a "Feature Vocabulary"
    # Each node has a set of feature strings
    node_raw_features = {nid: [] for nid in sorted_nodes}
    all_features = set()

    with open(feat_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            nid = int(parts[0])
            feats = parts[1:]
            if nid in node_mapper:
                node_raw_features[nid] = feats
                all_features.update(feats)

    sorted_features = sorted(list(all_features))
    feat_mapper = {ft: i for i, ft in enumerate(sorted_features)}
    num_features = len(sorted_features)

    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for gid, feats in node_raw_features.items():
        li = node_mapper[gid]
        for ft in feats:
            x[li, feat_mapper[ft]] = 1.0

    # -----------------------------
    # Read Circle Labels (multi-label)
    # -----------------------------
    circles = []
    with open(circle_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 2: continue
            members = [int(n) for n in parts[1].split()]
            circles.append(members)

    num_circles = len(circles)
    y = torch.zeros((num_nodes, num_circles), dtype=torch.float)
    for cid, members in enumerate(circles):
        for mid in members:
            if mid in node_mapper:
                y[node_mapper[mid], cid] = 1.0

    # -----------------------------
    # Train/Test masks
    # -----------------------------
    idx = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:split]] = True
    test_mask[idx[split:]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        num_features=num_features,
        num_circles=num_circles,
        train_mask=train_mask,
        test_mask=test_mask,
        sorted_features=sorted_features,
    )

    print(f"Ego {ego_id}: Nodes={num_nodes}, Edges={edge_index.shape[1]}, Feats={num_features}, Circles={num_circles}")

    return data


# -----------------------------
# GNN + CESNA-style per-circle θ_k
# -----------------------------
# class GNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_circles):
#         super().__init__()

#         # Graph signal
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)

#         # Circle-specific structural weight
#         self.weight = nn.Parameter(torch.randn(num_circles, hidden_channels) * 0.01)

#         # Circle-specific feature weight (CESNA θ_k)
#         self.theta = nn.Parameter(torch.randn(num_circles, in_channels) * 0.01)

#         self.bias = nn.Parameter(torch.zeros(num_circles))

#     def forward(self, x, edge_index): # x : [num_node, num_features]
        
#         # GNN structural embedding
        
#         h = F.relu(self.conv1(x, edge_index))
#         h = F.dropout(h, p=0.5, training=self.training)
#         h = self.conv2(h, edge_index)

#         struct_logits = h @ self.weight.T
#         feat_logits = x @ self.theta.T

#         # logit[u,k] = (graph info) + (feature info) + (bias)
#         return struct_logits + feat_logits + self.bias

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_circles):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_circles)

        # Explainable Prototypes
        # [num_circles, in_channels]
        self.prototypes = nn.Parameter(torch.randn(num_circles, in_channels))

    def forward(self, x, edge_index):
        # Prediction
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        
        logits = self.classifier(h)
        # Multi-label
        probs = torch.sigmoid(logits) # shape: [N, num_circles]

        # Reconstruction
        # 邏輯：如果這個人屬於圈子 A 和 B，那他的特徵應該像 (Template A + Template B)
        # probs: [N, K], prototypes: [K, F] -> recon_x: [N, F]
        recon_x = probs @ self.prototypes
        
        return logits, recon_x

# -----------------------------
# Feature Explanation
# -----------------------------
# def print_top_features(model, feature_names, top_k=10):
#     theta = model.theta.detach().cpu().numpy() # [num_circles, num_features]

#     for cid in range(theta.shape[0]):
#         print(f"\n===== Circle {cid} Top {top_k} Features =====")
        
#         weights = theta[cid] #
#         idx = np.argsort(-np.abs(weights))[:top_k] ## return the index by ascending power
#         for rank, feat_idx in enumerate(idx, start=1):
#             print(f"{rank:2d}. {feature_names[feat_idx]:30s}  weight={weights[feat_idx]:.4f}")

def get_circle_explanation(self, feature_names):
    
    """return  the feature

    Returns:
        _type_: _description_
    """
    
    explanations = {}
    with torch.no_grad():
        # 這裡的 weights 直接對應 input features
        weights = self.prototypes.cpu().numpy()
        
        for k in range(weights.shape[0]):
            # 找出權重最大的特徵 (正值代表正相關，負值代表負相關)
            # 配合 L1 Loss，這裡很多值會接近 0
            top_indices = abs(weights[k]).argsort()[::-1][:5] # 取前5重要
            
            circle_features = []
            for idx in top_indices:
                importance = weights[k][idx]
                feat_name = feature_names[idx]
                circle_features.append((feat_name, importance))
            
            explanations[f"Circle_{k}"] = circle_features
    return explanations



def compute_auc_scores(y_true, y_score):
    """
    y_true: numpy array (N, K)
    y_score: numpy array (N, K) ← logits or sigmoid(logits)

    return:
        micro_auc, macro_auc, per_class_auc(list)
    """
    # macro / micro AUC
    macro_auc = roc_auc_score(y_true, y_score, average='macro')
    micro_auc = roc_auc_score(y_true, y_score, average='micro')

    # per-class AUC (K個圈子)
    K = y_true.shape[1]
    per_class_auc = []
    for k in range(K):
        auc_k = roc_auc_score(y_true[:, k], y_score[:, k])
        per_class_auc.append(auc_k)

    return micro_auc, macro_auc, per_class_auc
# -----------------------------
# Experiment Runner
# -----------------------------
def run_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)

    egonet_files = glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))
    ego_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in egonet_files])

    print(f"Found {len(ego_ids)} ego-nets.")

    f1_scores = []

    for ego_id in ego_ids:
        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except Exception as e:
            print(f"Skip {ego_id}: {e}")
            continue
        ipdb.set_trace()
        data = data.to(device)

        model = GNN(
            in_channels=data.num_features,
            hidden_channels=64,
            num_circles=data.num_circles
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        alpha = 1.0 
        beta = 1e-3
        # Training
        for epoch in range(400):
            model.train()
            optimizer.zero_grad()
            # logits = model(data.x, data.edge_index)
            # loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            # loss.backward()
            # optimizer.step()
            
            logits, recon_x = model(data.x, data.edge_index)
            
            loss_cls = F.binary_cross_entropy_with_logits(logits[data.train_mask], data.y[data.train_mask])
            loss_recon = F.mse_loss(recon_x[data.train_mask], data.x[data.train_mask])
            loss_l1 = torch.norm(model.prototypes, p=1)

            total_loss = loss_cls + alpha * loss_recon + beta * loss_l1
            total_loss.backward()
            optimizer.step()
            
            
        # Eval
        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index)
            prob = torch.sigmoid(logits)              # (N,K)
            pred = (prob > 0.5).float()               # (N,K)

            y_true = data.y[data.test_mask].cpu().numpy()
            y_prob = prob[data.test_mask].cpu().numpy()   # 用 score 算 AUC
            y_pred = pred[data.test_mask].cpu().numpy()

            # ===== AUC =====
            micro_auc, macro_auc, per_class_auc = compute_auc_scores(y_true, y_prob)

            # ===== F1 =====
            f1_micro = f1_score(y_true, y_pred, average="micro")

            print(f"[Ego {ego_id}]  Micro-F1 = {f1_micro:.4f} | Micro-AUC = {micro_auc:.4f} | Macro-AUC = {macro_auc:.4f}")
            
            print("Per-circle AUC:")
            for i, auc_k in enumerate(per_class_auc):
                print(f"  Circle {i}: AUC = {auc_k:.4f}")

            f1_scores.append(f1_micro)


        # Feature explanation
        ### needd implement
    print("\n===== Final Report =====")
    print(f"Average F1 = {np.mean(f1_scores):.4f}")


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    run_experiment()
