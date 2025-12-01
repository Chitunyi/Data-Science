import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import glob
from torch_geometric.utils import negative_sampling
import pandas as pd
import warnings
from tqdm import tqdm

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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
    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    if not (os.path.exists(edge_path) and os.path.exists(circle_path)):
        raise FileNotFoundError()

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
    try:
        macro_auc = roc_auc_score(y_true, y_score, average='macro')
        micro_auc = roc_auc_score(y_true, y_score, average='micro')
    except ValueError:
        macro_auc = 0.5
        micro_auc = 0.5

    K = y_true.shape[1]
    per_class_auc = []
    for k in range(K):
        if len(np.unique(y_true[:, k])) == 2:
            auc_k = roc_auc_score(y_true[:, k], y_score[:, k])
        else:
            auc_k = 0.5 
        per_class_auc.append(auc_k)

    return micro_auc, macro_auc, per_class_auc


def compute_ber_score(y_true, y_pred):
    """
    calculate Balanced Error Rate (BER)
    BER = 0.5 * (False Negative Rate + False Positive Rate)
    Lower is better.
    """
    K = y_true.shape[1]
    ber_list = []
    
    for k in range(K):
        if len(np.unique(y_true[:, k])) < 2:
            # edge case
            ber_list.append(0.5) 
            continue

        # confusion_matrix requires binary inputs (0 or 1)
        tn, fp, fn, tp = confusion_matrix(y_true[:, k], y_pred[:, k], labels=[0, 1]).ravel()
        
        # FN Rate = FN / (TP + FN)  
        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        # FP Rate = FP / (TN + FP)  
        fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        ber = 0.5 * (fn_rate + fp_rate)
        ber_list.append(ber)

    return np.mean(ber_list)

# -----------------------------
# Model
# -----------------------------
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_circles):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_circles)
        self.prototypes = nn.Parameter(torch.randn(num_circles, in_channels))

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        
        logits = self.classifier(h)
        probs = torch.sigmoid(logits) 

        recon_x = probs @ self.prototypes
        
        return logits, recon_x

    def get_circle_explanation(self, feature_names, top_k=5):
        explanations = {}
        with torch.no_grad():
            weights = self.prototypes.cpu().numpy()
            
            for k in range(weights.shape[0]):
                top_indices = np.argsort(np.abs(weights[k]))[::-1][:top_k]
                
                circle_features = []
                for idx in top_indices:
                    importance = weights[k][idx]
                    feat_name = feature_names[idx] if idx < len(feature_names) else f"Feat_{idx}"
                    circle_features.append((feat_name, importance))
                
                explanations[f"Circle_{k}"] = circle_features
        return explanations

# -----------------------------
# Link Inference Attack (Return string instead of print)
# -----------------------------
def link_inference_attack(model, data, ego_id):
    """
    Returns (attack_auc, log_string)
    """
    log_buf = []
    log_buf.append(f"\n[Link Inference Attack - Ego {ego_id}]")
    
    model.eval()
    with torch.no_grad():
        h = F.relu(model.conv1(data.x, data.edge_index))
        h = model.conv2(h, data.edge_index) 
        
        pos_edge_index = data.edge_index
        num_pos = pos_edge_index.shape[1]
        perm = torch.randperm(num_pos)[:1000]
        pos_edges = pos_edge_index[:, perm]
        
        neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=4000)
        
        def compute_similarity(edges, embeddings):
            u, v = edges[0], edges[1]
            emb_u = embeddings[u]
            emb_v = embeddings[v]
            return F.cosine_similarity(emb_u, emb_v).cpu().numpy()
            
        pos_scores = compute_similarity(pos_edges, h)
        neg_scores = compute_similarity(neg_edge_index, h)
        
        log_buf.append(f"  Avg Similarity (Connected)   : {np.mean(pos_scores):.4f}")
        log_buf.append(f"  Avg Similarity (Unconnected) : {np.mean(neg_scores):.4f}")
        
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        
        if len(np.unique(y_true)) < 2:
            attack_auc = 0.5
        else:
            attack_auc = roc_auc_score(y_true, y_scores)
        
        log_buf.append(f"  => Attack AUC: {attack_auc:.4f}")
        
    return attack_auc, "\n".join(log_buf)

# -----------------------------
# Experiment Runner
# -----------------------------
def run_experiment():
    set_seed(40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)

    egonet_files = glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))
    ego_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in egonet_files])
   
    print(f"Found {len(ego_ids)} ego-nets.")
    
    # Initialize Log File
    with open("log.txt", "w") as f:
        f.write("Experiment Log\n================\n")

    results_summary = {
        'ego_id': [],
        'f1_micro': [],
        'auc_micro': [],
        'balanced_Error_Rate':[],
        'link_attack_auc': []
    }

    for ego_id in tqdm(ego_ids):
        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except Exception as e:
            continue
        
        data = data.to(device)

        model = GNN(
            in_channels=data.num_features,
            hidden_channels=128,
            num_circles=data.num_circles
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=5e-4)
        
        # Training
        for epoch in range(1300):
            model.train()
            optimizer.zero_grad()
            logits, recon_x = model(data.x, data.edge_index)
            loss_cls = F.binary_cross_entropy_with_logits(logits[data.train_mask], data.y[data.train_mask])
            loss_recon = F.mse_loss(recon_x[data.train_mask], data.x[data.train_mask])
            loss_l1 = torch.norm(model.prototypes, p=1)
            total_loss = loss_cls + 1.0 * loss_recon + 1e-5 * loss_l1
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

            if y_true.shape[0] > 0:
                micro_auc, macro_auc, _ = compute_auc_scores(y_true, y_prob)
                
                # [FIXED] Use y_pred (binary) instead of y_prob (continuous) for BER
                ber_score = compute_ber_score(y_true, y_pred)
                
                f1_micro = f1_score(y_true, y_pred, average="micro")

                # Attack
                attack_auc, attack_log = link_inference_attack(model, data, ego_id)
                
                # Collect Results
                results_summary['ego_id'].append(ego_id)
                results_summary['f1_micro'].append(f1_micro)
                results_summary['auc_micro'].append(micro_auc)
                results_summary['balanced_Error_Rate'].append(ber_score)
                results_summary['link_attack_auc'].append(attack_auc)
                
                # --- Write to log.txt ---
                with open("log.txt", "a") as f:
                    f.write(f"\n{'='*30}\n")
                    f.write(f"Ego ID: {ego_id}\n")
                    f.write(f"Micro-F1: {f1_micro:.4f} | Micro-AUC: {micro_auc:.4f}\n")
                    f.write(attack_log + "\n")
                    f.write(f"\n[Explanation for Ego {ego_id}]\n")
                    
                    explanations = model.get_circle_explanation(data.sorted_features, top_k=5)
                    for circle_name, features in explanations.items():
                        feat_str = ", ".join([f"{name} ({val:.2f})" for name, val in features])
                        f.write(f"  {circle_name}: {feat_str}\n")

    # -----------------------------
    # Final Terminal Output
    # -----------------------------
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*60)
    
    if results_summary['ego_id']:
        df = pd.DataFrame(results_summary)
        print(df.to_string(index=False))
        print("-" * 60)
        print(f"Average Micro-F1      : {df['f1_micro'].mean():.4f}")
        print(f"Average Micro-AUC     : {df['auc_micro'].mean():.4f}")
        # [FIXED] Use correct key 'balanced_Error_Rate'
        print(f"Average BER           : {df['balanced_Error_Rate'].mean():.4f} (Lower is better)")
        print(f"Average Link Attack AUC: {df['link_attack_auc'].mean():.4f}")
        print("\nNote: Detailed explanations and attack logs are saved in 'log.txt'.")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    run_experiment()