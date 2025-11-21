import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

# -----------------------------
#  Utility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
#  Data Loader
# -----------------------------
def load_kaggle_ego_graph_global_features(root_dir: str, ego_id: int):
    print(f"Loading Ego-net: {ego_id}...")
    
    # path
    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    # print(f"Debug Checking Paths:")
    # print(f"1. Edge Path:   {edge_path} -> {'[EXIST]' if os.path.exists(edge_path) else '[MISSING]'}")
    # print(f"2. Circle Path: {circle_path} -> {'[EXIST]' if os.path.exists(circle_path) else '[MISSING]'}")
    # print(f"3. Feat Path:   {feat_path} -> {'[EXIST]' if os.path.exists(feat_path) else '[MISSING]'}")
    
    if not os.path.exists(edge_path) or not os.path.exists(circle_path):
        missing_file = edge_path if not os.path.exists(edge_path) else circle_path
        raise FileNotFoundError(f"Critical file missing: {missing_file}")

    if not os.path.exists(edge_path) or not os.path.exists(circle_path):
        raise FileNotFoundError(f"Files not found for ego_id {ego_id}")

    # --- Find all Nodes and Edges in Ego-net  ---
    nodes = set([ego_id])
    raw_edges = []
    
    with open(edge_path, 'r') as f:
        for line in f:
            raw_parts = line.strip().split(':') 
            u = int(raw_parts[0])
            neighbors = []
            if len(raw_parts) > 1 and raw_parts[1].strip() != "":
                neighbors = [int(x) for x in raw_parts[1].strip().split()]
            
            nodes.add(u)
            for v in neighbors:
                nodes.add(v)
                raw_edges.append((u, v))

    # --- ID Mapping (Global ID -> Local Index 0..N-1) ---
    sorted_nodes = sorted(list(nodes))
    node_mapper = {global_id: i for i, global_id in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    
    edge_index = [[], []]
    for u, v in raw_edges:
        if u in node_mapper and v in node_mapper:
            # Add both directions for undirected graph
            edge_index[0].append(node_mapper[u])
            edge_index[1].append(node_mapper[v])
            edge_index[0].append(node_mapper[v])
            edge_index[1].append(node_mapper[u])
            
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # --- Features ---
    # Turn feature into digits, build a "Feature Vocabulary"
    # Each node has a set of feature strings
    node_raw_features = {node: [] for node in sorted_nodes}
    all_unique_features = set()

    print("Parsing features.txt (this might take a moment)...")
    with open(feat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            node_id = int(parts[0])
            
            if node_id in node_mapper:
                feats = parts[1:]
                node_raw_features[node_id] = feats
                for ft in feats:
                    all_unique_features.add(ft)

            # Exp B:
            # if node_id in node_mapper:
            #     raw_feats = parts[1:]
                
            #     filtered_feats = []
            #     for ft in raw_feats:
            #         #  or "work" in ft
            #         if "school" in ft:
            #             continue  # 跳過（模擬隱私保護）
            #         filtered_feats.append(ft)
                
            #     node_raw_features[node_id] = filtered_feats
            #     for ft in filtered_feats:
            #         all_unique_features.add(ft)
    
    # Feature ID Map
    sorted_features = sorted(list(all_unique_features))
    feat_mapper = {ft: i for i, ft in enumerate(sorted_features)}
    num_features = len(sorted_features)
    
    # Feature Matrix (Multi-hot encoding)
    # Shape: [num_nodes, num_features]
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)

    # Exp.A 改成設置為單位矩陣，註解掉下面處理 feature 的 for loop
    # x = torch.eye(num_nodes, dtype=torch.float)
    # num_features = num_nodes
    
    for global_id, feats in node_raw_features.items():
        local_idx = node_mapper[global_id]
        for ft in feats:
            if ft in feat_mapper:
                feat_idx = feat_mapper[ft]
                x[local_idx, feat_idx] = 1.0

    # --- Read Circles ---
    # Build Multi-label y
    # Shape: circleID: node1 node2 ...
    
    circles = []
    with open(circle_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            circle_members = [int(n) for n in parts[1].strip().split()]
            circles.append(circle_members)
            
    num_circles = len(circles)
    
    # Matrix y [num_nodes, num_circles]
    y = torch.zeros((num_nodes, num_circles), dtype=torch.float)
    
    for i, members in enumerate(circles):
        for member_id in members:
            if member_id in node_mapper:
                local_idx = node_mapper[member_id]
                y[local_idx, i] = 1.0

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes
    data.num_features = num_features
    data.num_circles = num_circles
    
    # Train/Test (Random Split 80/20)
    indices = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[:split]] = True
    data.test_mask[indices[split:]] = True

    print(f"Ego {ego_id} Loaded: Nodes={num_nodes}, Edges={edge_index.shape[1]}, Feats={num_features}, Circles={num_circles}")
    return data

# -----------------------------
#  Utilities 
# -----------------------------
def apply_feature_mask(x, rate):
    if rate <= 0:
        return x
    if rate >= 1:
        return torch.zeros_like(x)
    
    mask = torch.rand_like(x) > rate 
    return x * mask.float()

def apply_structure_perturbation(edge_index, num_nodes, rate):
    """
    1. Randomly DropEdge
    2. Randomly Add Noise
    """
    if rate <= 0:
        return edge_index
    
    num_edges = edge_index.shape[1]
    
    # --- Drop ---
    num_keep = int(num_edges * (1 - rate))
    perm = torch.randperm(num_edges, device=edge_index.device)
    keep_indices = perm[:num_keep]
    new_edge_index = edge_index[:, keep_indices]
    
    # --- Add ---
    # 加入與刪除數量相同的雜訊 edge
    num_add = int(num_edges * rate)
    
    # 隨機起點與終點
    row = torch.randint(0, num_nodes, (num_add,), device=edge_index.device)
    col = torch.randint(0, num_nodes, (num_add,), device=edge_index.device)
    added_edges = torch.stack([row, col], dim=0)
    
    final_edge_index = torch.cat([new_edge_index, added_edges], dim=1)
    return final_edge_index

# -----------------------------
#  Model
# -----------------------------
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x  

# -----------------------------
#  4. Exp
# -----------------------------
import glob
from sklearn.metrics import f1_score

# def run_experiment():
#     set_seed(42)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     current_file_path = os.path.abspath(__file__)
#     script_dir = os.path.dirname(current_file_path)
#     root_dir = script_dir
#     print(f"Script is running from: {root_dir}")

#     egonet_files = glob.glob(os.path.join(root_dir, 'egonets', '*.egonet'))
#     all_ego_ids = [int(os.path.basename(f).split('.')[0]) for f in egonet_files]
#     all_ego_ids.sort()
    
#     print(f"Found {len(all_ego_ids)} egonets. Starting training loop...")

#     f1_scores = []
    

#     for ego_id in all_ego_ids:
#         print(f"\n--- Processing Ego ID: {ego_id} ---")
        

#         try:
#             data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
#         except Exception as e:
#             print(f"Skipping {ego_id}: {e}")
#             continue

#         data = data.to(device)

#         model = GNNClassifier(data.num_features, 64, data.num_circles).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#         criterion = nn.BCEWithLogitsLoss()

#         model.train()
#         for epoch in range(200):
#             optimizer.zero_grad()
#             out = model(data.x, data.edge_index)
#             loss = criterion(out[data.train_mask], data.y[data.train_mask])
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             out = model(data.x, data.edge_index)
#             pred = (torch.sigmoid(out) > 0.5).float()
            
#             from sklearn.metrics import f1_score
#             if data.test_mask.sum() > 0:
#                 y_true = data.y[data.test_mask].cpu().numpy()
#                 y_pred = pred[data.test_mask].cpu().numpy()
#                 score = f1_score(y_true, y_pred, average='micro')
#                 print(f"ID {ego_id} Test F1: {score:.4f}")
#                 f1_scores.append(score)
#             else:
#                 print(f"ID {ego_id} has no test nodes.")

#     if len(f1_scores) > 0:
#         avg_f1 = sum(f1_scores) / len(f1_scores)
#         print(f"\n===========================================")
#         print(f"Successfully processed {len(f1_scores)} ego-nets.")
#         print(f"Average Micro-F1 Score: {avg_f1:.4f}")
#         print(f"===========================================")
#     else:
#         print("No valid results obtained.")

def run_experiment():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)
    
    egonet_files = glob.glob(os.path.join(root_dir, 'egonets', '*.egonet'))
    all_ego_ids = [int(os.path.basename(f).split('.')[0]) for f in egonet_files]
    all_ego_ids.sort()
    
    # target_ego_ids = all_ego_ids[:30] 
    target_ego_ids = all_ego_ids 
    
    print(f"Found {len(all_ego_ids)} egonets. Using {len(target_ego_ids)} for analysis.")

    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # --- results[experiment_type][noise_level] = average_f1 ---
    results = {
        'feature_mask': {lvl: [] for lvl in noise_levels},
        'structure_noise': {lvl: [] for lvl in noise_levels}
    }

    print("\n====== Starting Sensitivity Analysis ======")

    for ego_id in target_ego_ids:
        try:
            raw_data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except Exception:
            continue # Skip if file missing

        raw_data = raw_data.to(device)
        
        for rate in noise_levels:
            
            # --- Exp C: Feature Masking ---
            # Data Copy
            data_feat = raw_data.clone()
            data_feat.x = apply_feature_mask(data_feat.x, rate)
            
            model = GNNClassifier(data_feat.num_features, 64, data_feat.num_circles).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCEWithLogitsLoss()
            
            model.train()
            for _ in range(100): 
                optimizer.zero_grad()
                out = model(data_feat.x, data_feat.edge_index)
                loss = criterion(out[data_feat.train_mask], data_feat.y[data_feat.train_mask])
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                out = model(data_feat.x, data_feat.edge_index)
                pred = (torch.sigmoid(out) > 0.5).float()
                if data_feat.test_mask.sum() > 0:
                    score = f1_score(data_feat.y[data_feat.test_mask].cpu(), pred[data_feat.test_mask].cpu(), average='micro')
                    results['feature_mask'][rate].append(score)

            # --- Exp D: Structure Perturbation ---
            # Use raw features, perturb structure
            data_struc = raw_data.clone()
            data_struc.edge_index = apply_structure_perturbation(data_struc.edge_index, data_struc.num_nodes, rate)
            
            model = GNNClassifier(data_struc.num_features, 64, data_struc.num_circles).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            model.train()
            for _ in range(100):
                optimizer.zero_grad()
                out = model(data_struc.x, data_struc.edge_index)
                loss = criterion(out[data_struc.train_mask], data_struc.y[data_struc.train_mask])
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                out = model(data_struc.x, data_struc.edge_index)
                pred = (torch.sigmoid(out) > 0.5).float()
                if data_struc.test_mask.sum() > 0:
                    score = f1_score(data_struc.y[data_struc.test_mask].cpu(), pred[data_struc.test_mask].cpu(), average='micro')
                    results['structure_noise'][rate].append(score)

        print(f"Processed Ego {ego_id}")

    print("\n\n====== Final Report ======")
    print(f"{'Noise Rate':<12} | {'Feat Mask F1':<15} | {'Struct Noise F1':<15}")
    print("-" * 50)
    
    for rate in noise_levels:
        avg_feat = sum(results['feature_mask'][rate]) / len(results['feature_mask'][rate]) if results['feature_mask'][rate] else 0
        avg_struc = sum(results['structure_noise'][rate]) / len(results['structure_noise'][rate]) if results['structure_noise'][rate] else 0
        print(f"{rate:<12.1f} | {avg_feat:<15.4f} | {avg_struc:<15.4f}")

if __name__ == "__main__":
    run_experiment()