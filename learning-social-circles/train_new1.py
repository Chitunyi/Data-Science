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

    # --- Read Edges ---
    nodes = set([ego_id])
    raw_edges = []

    with open(edge_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            u = int(parts[0])
            neighbors = []
            if len(parts) > 1 and parts[1].strip() != "":
                neighbors = [int(x) for x in parts[1].strip().split()]

            nodes.add(u)
            for v in neighbors:
                nodes.add(v)
                raw_edges.append((u, v))
                
    # --- ID Mapping ---
    sorted_nodes = sorted(list(nodes))
    node_mapper = {gid: idx for idx, gid in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    edge_index = [[], []]
    for u, v in raw_edges:
        if u in node_mapper and v in node_mapper:
            ui, vi = node_mapper[u], node_mapper[v]
            edge_index[0].append(ui)
            edge_index[1].append(vi)
            edge_index[0].append(vi)
            edge_index[1].append(ui)

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # --- Features ---
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

    # --- Circle Labels ---
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

    # --- Masks ---
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
    return data

# -----------------------------
# Metric Calculation
# -----------------------------
def compute_auc_scores(y_true, y_score):
    """
    y_true: (N, K)
    y_score: (N, K)
    """
    try:
        macro_auc = roc_auc_score(y_true, y_score, average='macro')
        micro_auc = roc_auc_score(y_true, y_score, average='micro')
    except ValueError:
        # 當測試集中某個類別完全沒有正樣本或負樣本時，AUC 無法計算
        macro_auc = 0.5
        micro_auc = 0.5

    K = y_true.shape[1]
    per_class_auc = []
    for k in range(K):
        # 檢查該類別在測試集中是否同時存在 0 和 1
        if len(np.unique(y_true[:, k])) == 2:
            auc_k = roc_auc_score(y_true[:, k], y_score[:, k])
        else:
            auc_k = 0.5 # 無法計算時給預設值
        per_class_auc.append(auc_k)

    return micro_auc, macro_auc, per_class_auc

# -----------------------------
# Model
# -----------------------------
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_circles):
        super().__init__()
        
        # 1. GNN Backbone (Structure Encoding)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_circles)

        # 2. Explainable Prototypes (Feature Templates)
        # [num_circles, in_channels] - 與 Input Feature 維度一致
        self.prototypes = nn.Parameter(torch.randn(num_circles, in_channels))

    def forward(self, x, edge_index):
        # --- Prediction Path ---
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        
        logits = self.classifier(h)
        probs = torch.sigmoid(logits) # Multi-label probability [N, K]

        # --- Reconstruction Path ---
        # 重建邏輯：用預測出的圈子機率，混合對應的 Prototype
        # [N, K] @ [K, F] -> [N, F]
        recon_x = probs @ self.prototypes
        
        return logits, recon_x

    def get_circle_explanation(self, feature_names, top_k=5):
        """
        回傳每個圈子最重要的 input feature。
        """
        explanations = {}
        with torch.no_grad():
            # 轉到 CPU 讀取數值
            weights = self.prototypes.cpu().numpy()
            
            for k in range(weights.shape[0]):
                # 排序：取絕對值最大的 top_k (無論正負都代表該特徵對該圈子定義很重要)
                # 使用 [::-1] 反轉，變成從大到小
                top_indices = np.argsort(np.abs(weights[k]))[::-1][:top_k]
                
                circle_features = []
                for idx in top_indices:
                    importance = weights[k][idx]
                    feat_name = feature_names[idx] if idx < len(feature_names) else f"Feat_{idx}"
                    circle_features.append((feat_name, importance))
                
                explanations[f"Circle_{k}"] = circle_features
        return explanations

# -----------------------------
# Experiment Runner
# -----------------------------
def run_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)

    egonet_files = glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))
    # 取前幾個測試即可，全部跑完可能太久
    ego_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in egonet_files])[:5] 

    print(f"Found {len(ego_ids)} ego-nets (processing subset).")

    f1_scores = []

    for ego_id in ego_ids:
        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except Exception as e:
            print(f"Skip {ego_id}: {e}")
            continue
        
        data = data.to(device)

        model = GNN(
            in_channels=data.num_features,
            hidden_channels=64,
            num_circles=data.num_circles
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # LR稍微調大一點加速收斂
        
        # Hyperparameters for explanation
        alpha = 1.0     # Reconstruction weight
        beta = 1e-3     # L1 Sparsity weight

        # Training
        print(f"Training Ego {ego_id}...")
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            
            logits, recon_x = model(data.x, data.edge_index)
            
            # 1. Classification Loss (Multi-label)
            loss_cls = F.binary_cross_entropy_with_logits(logits[data.train_mask], data.y[data.train_mask])
            
            # 2. Reconstruction Loss (只對 Training set 做特徵對齊)
            loss_recon = F.mse_loss(recon_x[data.train_mask], data.x[data.train_mask])
            
            # 3. L1 Sparsity (針對 Prototypes)
            loss_l1 = torch.norm(model.prototypes, p=1)

            total_loss = loss_cls + alpha * loss_recon + beta * loss_l1
            total_loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index)
            prob = torch.sigmoid(logits)              
            pred = (prob > 0.5).float()               

            y_true = data.y[data.test_mask].cpu().numpy()
            y_prob = prob[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()

            # Metric Calculation
            if y_true.shape[0] > 0: # 確保有測試數據
                micro_auc, macro_auc, per_class_auc = compute_auc_scores(y_true, y_prob)
                f1_micro = f1_score(y_true, y_pred, average="micro")

                print(f"[Ego {ego_id}] Micro-F1: {f1_micro:.4f} | Micro-AUC: {micro_auc:.4f}")
                f1_scores.append(f1_micro)
            else:
                print(f"[Ego {ego_id}] No test data available.")

        # -----------------------------
        # Feature Explanation Output
        # -----------------------------
        print(f"\n[Explanation for Ego {ego_id}]")
        explanations = model.get_circle_explanation(data.sorted_features, top_k=5)
        
        for circle_name, features in explanations.items():
            # 組合字串： "FeatureName (Weight)"
            feat_str = ", ".join([f"{name} ({val:.2f})" for name, val in features])
            print(f"  {circle_name}: {feat_str}")
        print("-" * 50)

    print("\n===== Final Report =====")
    if f1_scores:
        print(f"Average F1 = {np.mean(f1_scores):.4f}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    run_experiment()