
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

from model import GNN, GNN_origin, MLPBaseline, LogisticRegressionBaseline

from utils import (
    set_seed,
    load_kaggle_ego_graph_global_features,
    induced_subgraph_data,
    filter_features_by_support_train_only,
    apply_feature_filter,
    compute_auc_scores,
    compute_ber_score,
    forward_logits,
    plot_embedding,
    extract_top1_importance_per_circle
)

def run_all_models():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    egonet_files = glob.glob(os.path.join(root_dir, "egonets", "*.egonet"))
    ego_ids = sorted([int(os.path.basename(f).split(".")[0]) for f in egonet_files])
    print(f"Found {len(ego_ids)} ego-nets.")

    # (name, factory(in_ch,K), need_edge, use_extra_loss)
    model_specs = [
        ("MLP", lambda in_ch, K: MLPBaseline(in_channels=in_ch, hidden_channels=32, num_circles=K), False, False),
        ("Logistic", lambda in_ch, K: LogisticRegressionBaseline(in_channels=in_ch, num_circles=K), False, False),
        ("GNN_origin", lambda in_ch, K: GNN_origin(in_channels=in_ch, hidden_channels=32, num_circles=K), True, False),
        ("GNN", lambda in_ch, K: GNN(in_channels=in_ch, hidden_channels=32, num_circles=K), True, True),
    ]

    for model_name, factory, need_edge, use_extra_loss in model_specs:
        set_seed(40)
        print(f"model: {model_name}")
        log_path = f"log_{model_name}.txt"
        with open(log_path, "w") as f:
            f.write(f"Experiment Log - {model_name}\n================\n")

        results_summary = {
            "ego_id": [],
            "f1_micro": [],
            "auc_micro": [],
            "balanced_Error_Rate": [],
            "ap_micro": [],
        }

        total_orig_dim = 0
        total_filt_dim = 0
        
        all_circle_top1 = []
        all_y_true = []
        all_y_prob = []

        for ego_id in tqdm(ego_ids):
            try:
                full = load_kaggle_ego_graph_global_features(root_dir, ego_id)
            except Exception:
                continue

            # fully-inductive: train/test nodes disjoint subgraphs
            train_data = induced_subgraph_data(full, full.train_mask)
            test_data = induced_subgraph_data(full, full.test_mask)

            # train-only feature filtering (avoid leakage)
            orig_dim = train_data.num_features
            keep_idx, new_names = filter_features_by_support_train_only(train_data, min_count=2, max_frac=1.0)
            
            train_data = apply_feature_filter(train_data, keep_idx, new_names)
            test_data = apply_feature_filter(test_data, keep_idx, new_names)

            filt_dim = train_data.num_features
            total_orig_dim += orig_dim
            total_filt_dim += filt_dim

            train_data = train_data.to(device)
            test_data = test_data.to(device)

            model = factory(train_data.num_features, train_data.num_circles).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=5e-4)

            #Training
            for _ in range(1000):
                model.train()
                optimizer.zero_grad()

                logits, recon_x = forward_logits(model, train_data, need_edge)
                loss = F.binary_cross_entropy_with_logits(logits, train_data.y)

                # extra losses only for your interpretable GNN
                if use_extra_loss and recon_x is not None:
                    loss_recon = F.mse_loss(recon_x, train_data.x)
                    loss = loss + 2.0 * loss_recon

                if use_extra_loss and hasattr(model, "prototypes"):
                    loss = loss + 1e-5 * torch.norm(model.prototypes, p=1)
                    # loss = loss + torch.sum(model.prototypes ** 2)
                loss.backward()
                optimizer.step()

            #Eval 
            model.eval()
            with torch.no_grad():
                logits, _ = forward_logits(model, test_data, need_edge)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.2).float()

                y_true = test_data.y.cpu().numpy()
                y_prob = prob.cpu().numpy()
                y_pred = pred.cpu().numpy()

                if y_true.shape[0] == 0:
                    continue

                micro_auc, _, _ = compute_auc_scores(y_true, y_prob)
                try:
                    micro_ap = average_precision_score(y_true, y_prob, average="micro")
                except ValueError:
                    micro_ap = 0.0

                ber_score = compute_ber_score(y_true, y_pred)
                f1_micro = f1_score(y_true, y_pred, average="micro")

                results_summary["ego_id"].append(ego_id)
                results_summary["f1_micro"].append(f1_micro)
                results_summary["auc_micro"].append(micro_auc)
                results_summary["balanced_Error_Rate"].append(ber_score)
                results_summary["ap_micro"].append(micro_ap)

                all_y_true.append(y_true)
                all_y_prob.append(y_prob)

                with open(log_path, "a") as f:
                    f.write(f"\n{'='*30}\n")
                    f.write(f"Ego ID: {ego_id}\n")
                    f.write(f"Micro-F1: {f1_micro:.4f} | Micro-AUC: {micro_auc:.4f} | Micro-AP: {micro_ap:.4f}\n")

                    # explanation only if the model supports it
                    if hasattr(model, "get_circle_explanation"):
                        f.write(f"\n[Explanation for Ego {ego_id}]\n")
                        explanations = model.get_circle_explanation(train_data.sorted_features, top_k=5)
                        for circle_name, feats in explanations.items():
                            feat_str = ", ".join([f"{name} ({val:.2f})" for name, val in feats])
                            f.write(f"  {circle_name}: {feat_str}\n")
                            
                        top1_list = extract_top1_importance_per_circle(explanations)

                        if len(top1_list) > 0:
                            all_circle_top1.extend(top1_list)
                            

            if ego_id in ego_ids[:3]:
                plot_embedding(model, train_data, ego_id, tag=f"{model_name}_train")

            if ego_id in ego_ids[:3]:
                plot_embedding(model, test_data, ego_id, tag=f"{model_name}_test")

        # ---------------- Summary / plots ----------------
        print("\n" + "=" * 60)
        print(f"OVERALL PERFORMANCE SUMMARY ({model_name})")
        print("=" * 60)

        df = pd.DataFrame(results_summary)

        print(df.to_string(index=False))
        print("-" * 60)
        print(f"Average Micro-F1      : {df['f1_micro'].mean():.4f}")
        print(f"Average Micro-AUC     : {df['auc_micro'].mean():.4f}")
        print(f"Average Micro-AP      : {df['ap_micro'].mean():.4f}")
        print(f"Average BER           : {df['balanced_Error_Rate'].mean():.4f} (Lower is better)")

        overall_removed = 1.0 - (total_filt_dim / max(total_orig_dim, 1))
        print(f"Train-only feature removal ratio : {overall_removed:.2%}")

        # threshold scan
        thresholds = np.linspace(0.1, 0.9, 17)
        f1_curve, ber_curve = [], []

        for t in thresholds:
            f1_list, ber_list = [], []
            for y_true, y_prob in zip(all_y_true, all_y_prob):
                y_pred = (y_prob > t).astype(float)
                f1_list.append(f1_score(y_true, y_pred, average="micro"))
                ber_list.append(compute_ber_score(y_true, y_pred))
            f1_curve.append(np.mean(f1_list))
            ber_curve.append(np.mean(ber_list))

        best_idx = int(np.argmax(f1_curve))
        best_ber_idx = int(np.argmin(ber_curve))
        print(f"\nBest threshold (by Micro-F1): {thresholds[best_idx]:.2f}, F1 = {f1_curve[best_idx]:.4f}")
        print(f"Best threshold (by BER)      : {thresholds[best_ber_idx]:.2f}, BER = {ber_curve[best_ber_idx]:.4f}")

        plt.figure()
        plt.plot(thresholds, f1_curve, marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Micro-F1")
        plt.title(f"Micro-F1 vs Threshold ({model_name})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"f1_vs_threshold_{model_name}.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(thresholds, ber_curve, marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("BER (lower is better)")
        plt.title(f"Balanced Error Rate vs Threshold ({model_name})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"ber_vs_threshold_{model_name}.png", dpi=300)
        plt.close()

        # PR curve (overall micro)
        Y_true_flat = np.concatenate([a.ravel() for a in all_y_true]) if len(all_y_true) else None
        Y_prob_flat = np.concatenate([a.ravel() for a in all_y_prob]) if len(all_y_prob) else None

        if Y_true_flat is not None and len(Y_true_flat) > 0:
            precision, recall, _ = precision_recall_curve(Y_true_flat, Y_prob_flat)
            overall_ap = average_precision_score(Y_true_flat, Y_prob_flat)
            baseline = float(np.sum(Y_true_flat) / len(Y_true_flat))

            print(f"Overall Micro Average Precision (AP): {overall_ap:.4f}")

            plt.figure()
            plt.plot([0, 1], [baseline, baseline], linestyle="--",
                     label=f"Random Baseline (AP={baseline:.4f})")
            plt.plot(recall, precision, marker=".", markersize=1,
                     label=f"Model (AP={overall_ap:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (Overall Micro-Average) ({model_name})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"pr_curve_micro_{model_name}.png", dpi=300)
            plt.close()

    circle_weighted_mean = float(np.mean(all_circle_top1))
    print(f"Circle-weighted Top-1 Importance: {circle_weighted_mean:.4f}")      
if __name__ == "__main__":
    run_all_models()
