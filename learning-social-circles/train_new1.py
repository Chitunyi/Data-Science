import os
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from model import GNN, LogisticRegressionBaseline, MLPBaseline, GNN_origin
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
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

def filter_features_by_support(data, min_count=2, max_frac=1.0):
    """
    根據 feature 在節點上的出現次數做篩選：
      - min_count: 至少要在多少個節點上出現，才保留
      - max_frac:  最多允許在多少比例的節點上出現（例如 0.95），> 這個比例就太常見，可以選擇砍掉

    data.x: [num_nodes, num_features] 的 0/1 tensor
    data.sorted_features: 長度為 num_features 的 feature name list
    """
    x = data.x  # [N, D]
    counts = x.sum(dim=0)              # [D] 每個 feature 在幾個 node 上是 1
    N = x.size(0)
    max_count = max_frac * N

    keep_mask = (counts >= min_count) & (counts <= max_count)

    keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)

    # 如果全部被砍光，避免爆炸，直接回原 data
    if keep_indices.numel() == 0:
        return data

    x_new = x[:, keep_indices]
    new_feature_names = [data.sorted_features[i] for i in keep_indices.tolist()]

    data_new = Data(
        x=x_new,
        edge_index=data.edge_index,
        y=data.y,
        num_nodes=data.num_nodes,
        num_features=x_new.size(1),
        num_circles=data.num_circles,
        train_mask=data.train_mask,
        test_mask=data.test_mask,
        sorted_features=new_feature_names,
    )
    return data_new

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
    # [NEW] Add Average Precision (AP) for each ego
    'ap_micro': [] 
    }
    total_orig_dim = 0
    total_filt_dim  = 0
    all_y_true = []
    all_y_prob = []
    
    # ---------------------------------------------
    # Main Loop
    # ---------------------------------------------
    for ego_id in tqdm(ego_ids):
        try:
            data = load_kaggle_ego_graph_global_features(root_dir, ego_id)
        except Exception as e:
            continue
        
        data_cpu = filter_features_by_support(data, min_count=2, max_frac=1.0)
        orig_dim = data.num_features
        filt_dim = data_cpu.num_features
        
        total_orig_dim += orig_dim
        total_filt_dim += filt_dim
        
        data = data.to(device)

        model = GNN(
            in_channels=data.num_features,
            hidden_channels=32,
            num_circles=data.num_circles
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=5e-4)
        
        # Training
        for epoch in range(1000):
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
            pred = (prob > 0.2).float()               

            y_true = data.y[data.test_mask].cpu().numpy()
            y_prob = prob[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()

            if y_true.shape[0] > 0:
                micro_auc, macro_auc, _ = compute_auc_scores(y_true, y_prob)
                
                # [NEW] Calculate Micro-Average Precision Score (AP)
                try:
                    micro_ap = average_precision_score(y_true, y_prob, average='micro')
                except ValueError:
                    micro_ap = 0.0 # Handle case where only one class is present
                
                all_y_true.append(y_true)
                all_y_prob.append(y_prob)
                ber_score = compute_ber_score(y_true, y_pred)
                
                f1_micro = f1_score(y_true, y_pred, average="micro")

          
                
                # Collect Results
                results_summary['ego_id'].append(ego_id)
                results_summary['f1_micro'].append(f1_micro)
                results_summary['auc_micro'].append(micro_auc)
                results_summary['balanced_Error_Rate'].append(ber_score)
                # [NEW] Collect Micro-AP
                results_summary['ap_micro'].append(micro_ap)

                
                # --- Write to log.txt ---
                with open("log.txt", "a") as f:
                    f.write(f"\n{'='*30}\n")
                    f.write(f"Ego ID: {ego_id}\n")
                    f.write(f"Micro-F1: {f1_micro:.4f} | Micro-AUC: {micro_auc:.4f} | Micro-AP: {micro_ap:.4f}\n")
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
    

    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))
    print("-" * 60)
    print(f"Average Micro-F1      : {df['f1_micro'].mean():.4f}")
    print(f"Average Micro-AUC     : {df['auc_micro'].mean():.4f}")
    # [NEW] Print Average Micro-AP
    print(f"Average Micro-AP      : {df['ap_micro'].mean():.4f}") 
    print(f"Average BER           : {df['balanced_Error_Rate'].mean():.4f} (Lower is better)")
    overall_removed = 1.0 - (total_filt_dim / total_orig_dim)
    print(f"Weighted feature removal ratio : {overall_removed:.2%}")

    # -----------------------------
    # Threshold Analysis & Plotting
    # -----------------------------
    
    # 要掃的 threshold 範圍
    thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9
    f1_curve = []
    ber_curve = []

    if len(all_y_true) > 0:  # 確保真的有資料
        
        # ---------------------------------------------
        # Threshold Scanning for F1 and BER
        # ---------------------------------------------
        for t in thresholds:
            f1_list = []
            ber_list = []

            # 對每個 ego
            for y_true, y_prob in zip(all_y_true, all_y_prob):
                # thresholding
                y_pred = (y_prob > t).astype(float)

                # F1
                f1_list.append(f1_score(y_true, y_pred, average="micro"))

                # BER
                ber_list.append(compute_ber_score(y_true, y_pred))

            # 取每個 threshold 下，各 ego 的平均
            f1_curve.append(np.mean(f1_list))
            ber_curve.append(np.mean(ber_list))

        # 找出 F1 最好的 threshold
        best_idx = int(np.argmax(f1_curve))
        print(f"\nBest threshold (by Micro-F1): {thresholds[best_idx]:.2f}, F1 = {f1_curve[best_idx]:.4f}")
        # 找出 BER 最小的 threshold
        best_ber_idx = int(np.argmin(ber_curve))
        print(f"Best threshold (by BER)      : {thresholds[best_ber_idx]:.2f}, BER = {ber_curve[best_ber_idx]:.4f}")

        # 1) F1 vs threshold
        plt.figure()
        plt.plot(thresholds, f1_curve, marker='o')
        plt.xlabel("Threshold")
        plt.ylabel("Micro-F1")
        plt.title("Micro-F1 vs Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("f1_vs_threshold.png", dpi=300)

        # 2) BER vs threshold
        plt.figure()
        plt.plot(thresholds, ber_curve, marker='o')
        plt.xlabel("Threshold")
        plt.ylabel("BER (lower is better)")
        plt.title("Balanced Error Rate vs Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ber_vs_threshold.png", dpi=300)

        # ---------------------------------------------
        # [NEW] Precision-Recall Curve Plotting
        # ---------------------------------------------
        
        # Combine all test results
        all_y_true_flat = [arr.ravel() for arr in all_y_true]
        all_y_prob_flat = [arr.ravel() for arr in all_y_prob]

        Y_true_flat = np.concatenate(all_y_true_flat)
        Y_prob_flat = np.concatenate(all_y_prob_flat)

        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(Y_true_flat, Y_prob_flat)
        
        # Compute Area Under the PR Curve (Average Precision) for the overall dataset
        overall_ap = average_precision_score(Y_true_flat, Y_prob_flat)
        print(f"Overall Micro Average Precision (AP): {overall_ap:.4f}")

        # Plot PR Curve
        plt.figure()
        # [NOTE] Baseline for PR curve is the ratio of positive samples (P / (P+N))
        baseline = np.sum(Y_true_flat) / len(Y_true_flat)
        plt.plot([0, 1], [baseline, baseline], linestyle='--', label=f'Random Baseline (AP={baseline:.4f})')
        plt.plot(recall, precision, marker='.', markersize=1, label=f'Model (AP={overall_ap:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Overall Micro-Average)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("pr_curve_micro.png", dpi=300)

if __name__ == "__main__":
    run_experiment()