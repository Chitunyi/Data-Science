import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from sklearn.metrics import (
    f1_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)

# ============================================================
# 1. 讀 Kaggle Social Circles 資料
# ============================================================

def load_kaggle_ego_graph(root_dir, ego_id):
    """
    讀 Kaggle 的 Social Circles 檔案格式：
    - egonets/{ego_id}.egonet
    - Training/{ego_id}.circles
    - features.txt

    回傳:
        G:      networkx.Graph
        X:      (N, F) numpy array, node features
        node_to_idx: global id -> local idx (0..N-1)
        idx_to_node: list[idx] -> global id
        feat_map: feature string -> index
        circles: list[list[global_node_id]] ground truth circles
    """

    edge_path = os.path.join(root_dir, "egonets", f"{ego_id}.egonet")
    circle_path = os.path.join(root_dir, "Training", f"{ego_id}.circles")
    feat_path = os.path.join(root_dir, "features.txt")

    print(f"[Loading Ego-net] {ego_id}")

    # -----------------------------
    # 1. Read ego-network edges
    # -----------------------------
    nodes = set([ego_id])
    raw_edges = []

    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"{edge_path} not found")

    with open(edge_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            u = int(parts[0])
            nodes.add(u)

            if len(parts) > 1 and parts[1].strip() != "":
                neigh = [int(x) for x in parts[1].split()]
            else:
                neigh = []

            for v in neigh:
                nodes.add(v)
                raw_edges.append((u, v))

    # build graph
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for u, v in raw_edges:
        G.add_edge(u, v)

    # -----------------------------
    # 2. Read features.txt
    # -----------------------------
    node_features = {nid: [] for nid in nodes}
    all_feat = set()

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"{feat_path} not found")

    with open(feat_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            nid = int(parts[0])
            if nid in node_features:
                feats = parts[1:]
                node_features[nid] = feats
                for ft in feats:
                    all_feat.add(ft)

    sorted_feat = sorted(list(all_feat))
    feat_map = {ft: i for i, ft in enumerate(sorted_feat)}
    F = len(sorted_feat)

    sorted_nodes = sorted(list(nodes))
    node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    idx_to_node = sorted_nodes
    N = len(sorted_nodes)

    X = np.zeros((N, F), dtype=np.float32)
    for n in sorted_nodes:
        feats = node_features[n]
        for ft in feats:
            X[node_to_idx[n], feat_map[ft]] = 1.0

    # -----------------------------
    # 3. Read ground-truth circles
    # -----------------------------
    circles = []
    if os.path.exists(circle_path):
        with open(circle_path, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) < 2:
                    continue
                members = [int(m) for m in parts[1].split()]
                circles.append(members)
    else:
        print(f"[Warning] {circle_path} not found, no GT circles.")

    print(
        f"[Ego {ego_id}] Nodes={N}, Edges={G.number_of_edges()}, "
        f"Feats={F}, Circles={len(circles)}"
    )

    return G, X, node_to_idx, idx_to_node, feat_map, circles


# ============================================================
# 2. McAuley & Leskovec Social Circle Model (Kaggle 版)
# ============================================================

class SocialCircleModel:
    """
    Social Circles model (簡化版) for Kaggle Facebook data.

    - 重疊圈子
    - φ(x,y) = (1, -σ_{x,y}), 其中 σ 是 feature XOR
    - logistic edge likelihood + L1 on theta
    - 交替最佳化:
        1) 固定 C 更新 θ, α
        2) 固定 θ, α 更新 C (ICM flip)
    """

    def __init__(self, K, feature_dim, lambda_l1=1.0, device="cpu"):
        self.K = K
        self.P = feature_dim  # feature 維度
        self.lambda_l1 = lambda_l1
        self.device = device

        D = self.P + 1  # φ = (1, -σ) => 1 + P 維
        self.theta = nn.Parameter(
            0.01 * torch.randn(K, D, dtype=torch.float32, device=device)
        )
        self.alpha = nn.Parameter(torch.ones(K, dtype=torch.float32, device=device))

        self.circle_membership = None  # (K, N) bool
        self.node_to_idx = None
        self.idx_to_node = None

        # 為了之後 BIC / log-likelihood 用
        self.edge_features = None
        self.edge_labels = None
        self.edge_pairs = None

    # --------------------------------------------------
    # φ(x,y) = (1, -σ)  where σ 是 XOR(feature difference)
    # --------------------------------------------------
    def _build_edge_features(self, X, G):
        """
        X: (N, P) numpy array
        產生:
            edge_pairs: list[(node_x, node_y)]
            edge_features: (M, P+1)
        """
        nodes = self.idx_to_node
        N = len(nodes)
        P = self.P

        feats = []
        pairs = []

        for i in range(N):
            for j in range(i + 1, N):
                x = nodes[i]
                y = nodes[j]

                if not (x in G.nodes() and y in G.nodes()):
                    continue

                fx = X[i]
                fy = X[j]

                sigma = (fx != fy).astype(np.float32)  # XOR
                phi = np.concatenate([np.ones(1, dtype=np.float32), -sigma])
                feats.append(phi)
                pairs.append((x, y))

        self.edge_pairs = pairs
        edge_features = np.stack(feats, axis=0)
        return edge_features

    def _build_edge_labels(self, G):
        labels = []
        for (x, y) in self.edge_pairs:
            labels.append(1.0 if G.has_edge(x, y) else 0.0)
        return np.array(labels, dtype=np.float32)

    # --------------------------------------------------
    # log-likelihood
    # --------------------------------------------------
    def _log_likelihood(self, edge_features, edge_labels, C):
        """
        edge_features: (M, D) tensor
        edge_labels:   (M,)   tensor
        C:             (K, N) bool tensor
        """
        phi = edge_features  # (M, D)
        K = self.K

        # z[k,e] = <φ_e, θ_k>
        z = torch.matmul(self.theta, phi.T)  # (K, M)

        # 把 edge 的 node id 轉成 index
        idx_i = torch.tensor(
            [self.node_to_idx[x] for (x, _) in self.edge_pairs],
            device=self.device,
            dtype=torch.long,
        )
        idx_j = torch.tensor(
            [self.node_to_idx[y] for (_, y) in self.edge_pairs],
            device=self.device,
            dtype=torch.long,
        )

        in_circle = (C[:, idx_i] & C[:, idx_j]).float()  # (K, M)
        not_in = 1.0 - in_circle

        alpha = self.alpha.unsqueeze(1)  # (K,1)
        d = in_circle - alpha * not_in   # (K,M)

        Phi = (d * z).sum(0)            # (M,)
        y = edge_labels                 # (M,)
        ll = (y * Phi - torch.log1p(torch.exp(Phi))).sum()
        return ll

    # --------------------------------------------------
    # 更新 Θ
    # --------------------------------------------------
    def _update_parameters(self, edge_features, edge_labels, C, steps=200, lr=0.05):
        opt = torch.optim.Adam([self.theta, self.alpha], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            ll = self._log_likelihood(edge_features, edge_labels, C)
            reg = self.lambda_l1 * self.theta.abs().sum()
            loss = -(ll - reg)
            loss.backward()
            opt.step()
            with torch.no_grad():
                self.alpha.clamp_(min=0.0)

    # --------------------------------------------------
    # 更新 C (ICM flip)
    # --------------------------------------------------
    def _update_circles(self, edge_features, edge_labels, C, sweeps=3):
        with torch.no_grad():
            current_ll = self._log_likelihood(edge_features, edge_labels, C)

            K, N = C.shape
            for _ in range(sweeps):
                improved = False
                for k in range(K):
                    for v in range(N):
                        C[k, v] = ~C[k, v]
                        new_ll = self._log_likelihood(edge_features, edge_labels, C)
                        if new_ll > current_ll:
                            current_ll = new_ll
                            improved = True
                        else:
                            C[k, v] = ~C[k, v]
                if not improved:
                    break
        return C

    # --------------------------------------------------
    # 完整訓練 + 把 edge_features / labels 存起來 (for BIC)
    # --------------------------------------------------
    def fit(self, G, X, node_to_idx, idx_to_node,
            max_outer_iters=5, param_steps=100, circle_sweeps=3):

        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node

        edge_feats_np = self._build_edge_features(X, G)
        edge_labels_np = self._build_edge_labels(G)

        edge_features = torch.tensor(edge_feats_np, dtype=torch.float32, device=self.device)
        edge_labels = torch.tensor(edge_labels_np, dtype=torch.float32, device=self.device)

        self.edge_features = edge_features
        self.edge_labels = edge_labels

        N = len(idx_to_node)
        C = torch.rand((self.K, N), device=self.device) < 0.5

        for it in range(max_outer_iters):
            print(f"  [Outer iter {it+1}/{max_outer_iters}] Update Θ")
            self._update_parameters(edge_features, edge_labels, C,
                                    steps=param_steps, lr=0.05)

            print(f"  [Outer iter {it+1}/{max_outer_iters}] Update C")
            C = self._update_circles(edge_features, edge_labels, C,
                                     sweeps=circle_sweeps)

        self.circle_membership = C

    def get_circles(self):
        """
        回傳每個圈子的 node list（global node id）
        """
        circles = []
        C = self.circle_membership
        if C is None:
            return circles
        K, N = C.shape
        for k in range(K):
            members = []
            idxs = torch.where(C[k])[0].tolist()
            for idx in idxs:
                members.append(self.idx_to_node[idx])
            circles.append(members)
        return circles

    def compute_log_likelihood(self):
        """
        給 BIC 用：回傳目前模型的 log-likelihood (scalar float)
        """
        if self.edge_features is None or self.edge_labels is None or self.circle_membership is None:
            raise RuntimeError("Model not fitted yet.")
        with torch.no_grad():
            ll = self._log_likelihood(self.edge_features,
                                      self.edge_labels,
                                      self.circle_membership)
        return float(ll.item())


# ============================================================
# 3. 評估：F1_micro / F1_macro / NMI / ARI
# ============================================================

def _match_circles_greedy(pred_sets, true_sets):
    """
    用簡單 greedy matching 把 predicted circles 對到 true circles；
    用 Jaccard similarity 當 score。
    回傳: list[(pred_idx, true_idx)]
    """
    P = len(pred_sets)
    T = len(true_sets)
    used_pred = set()
    used_true = set()
    pairs = []

    while len(used_pred) < P and len(used_true) < T:
        best_score = 0.0
        best_pair = None
        for i in range(P):
            if i in used_pred:
                continue
            for j in range(T):
                if j in used_true:
                    continue
                inter = len(pred_sets[i] & true_sets[j])
                union = len(pred_sets[i] | true_sets[j])
                if union == 0:
                    score = 0.0
                else:
                    score = inter / union
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair is None or best_score == 0.0:
            break
        i, j = best_pair
        used_pred.add(i)
        used_true.add(j)
        pairs.append((i, j))

    return pairs


def _single_label_from_sets(circle_sets, N):
    """
    把重疊 circle sets => 單一 cluster label per node（如果多圈，就取第一個）
    """
    labels = -1 * np.ones(N, dtype=int)
    for c_idx, s in enumerate(circle_sets):
        for n in s:
            if labels[n] == -1:
                labels[n] = c_idx
    return labels


def evaluate_circles(pred_circles, true_circles, idx_to_node):
    """
    pred_circles: list of list[global node id]
    true_circles: list of list[global node id]
    idx_to_node:  list index -> global node id

    回傳:
        dict with f1_micro, f1_macro, nmi, ari
    """
    node_to_idx = {n: i for i, n in enumerate(idx_to_node)}
    N = len(idx_to_node)

    # 轉成 index sets
    pred_sets = []
    for c in pred_circles:
        s = set(node_to_idx[n] for n in c if n in node_to_idx)
        if len(s) > 0:
            pred_sets.append(s)
    true_sets = []
    for c in true_circles:
        s = set(node_to_idx[n] for n in c if n in node_to_idx)
        if len(s) > 0:
            true_sets.append(s)

    if len(pred_sets) == 0 or len(true_sets) == 0:
        return dict(f1_micro=0.0, f1_macro=0.0, nmi=0.0, ari=0.0)

    # 1) multi-label F1: 用 greedy matching 把圈子對齊後 flatten node-circle
    pairs = _match_circles_greedy(pred_sets, true_sets)
    if len(pairs) == 0:
        return dict(f1_micro=0.0, f1_macro=0.0, nmi=0.0, ari=0.0)

    C_match = len(pairs)
    Y_true = np.zeros((N, C_match), dtype=int)
    Y_pred = np.zeros((N, C_match), dtype=int)

    for col, (pi, ti) in enumerate(pairs):
        for n in true_sets[ti]:
            Y_true[n, col] = 1
        for n in pred_sets[pi]:
            Y_pred[n, col] = 1

    y_true_flat = Y_true.flatten()
    y_pred_flat = Y_pred.flatten()

    f1_micro = f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)

    # 2) NMI / ARI：把重疊 circle 壓成單一 cluster label
    labels_true = _single_label_from_sets(true_sets, N)
    labels_pred = _single_label_from_sets(pred_sets, N)

    mask = (labels_true != -1) & (labels_pred != -1)
    if mask.sum() == 0:
        nmi = 0.0
        ari = 0.0
    else:
        nmi = normalized_mutual_info_score(labels_true[mask], labels_pred[mask])
        ari = adjusted_rand_score(labels_true[mask], labels_pred[mask])

    return dict(
        f1_micro=float(f1_micro),
        f1_macro=float(f1_macro),
        nmi=float(nmi),
        ari=float(ari),
    )


# ============================================================
# 4. 自動挑 K (BIC) + 跑全部 ego
# ============================================================

def train_one_ego_with_autoK(root_dir, ego_id,
                             K_candidates=(1, 2, 3, 4, 5),
                             device="cpu",
                             max_outer_iters=5,
                             param_steps=80,
                             circle_sweeps=3):
    """
    對單一 ego：
      - 對 K in K_candidates 跑 SocialCircleModel
      - 用 BIC 選出最佳 K
      - 回傳最佳模型 + 評估指標
    """
    G, X, node_to_idx, idx_to_node, feat_map, gt_circles = \
        load_kaggle_ego_graph(root_dir, ego_id)

    feature_dim = X.shape[1]

    best_bic = None
    best_model = None
    best_K = None
    best_ll = None

    for K in K_candidates:
        print(f"\n=== Ego {ego_id} | Try K={K} ===")
        model = SocialCircleModel(K=K, feature_dim=feature_dim,
                                  lambda_l1=0.1, device=device)
        model.fit(G, X, node_to_idx, idx_to_node,
                  max_outer_iters=max_outer_iters,
                  param_steps=param_steps,
                  circle_sweeps=circle_sweeps)

        ll = model.compute_log_likelihood()
        M = len(model.edge_labels)  # logistic sum over all unordered pairs
        num_params = K * (feature_dim + 2)  # θ: K*(P+1), α: K => K*(P+2)
        bic = -2.0 * ll + num_params * np.log(M)

        print(f"  -> log-likelihood={ll:.4f}, BIC={bic:.4f}")

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_model = model
            best_K = K
            best_ll = ll

    # 用最佳 K 的 model 做評估
    print(f"\n>>> Ego {ego_id}: best K={best_K}, BIC={best_bic:.4f}, ll={best_ll:.4f}")
    pred_circles = best_model.get_circles()
    metrics = evaluate_circles(pred_circles, gt_circles, idx_to_node)

    return {
        "ego_id": ego_id,
        "best_K": best_K,
        "best_BIC": best_bic,
        "best_ll": best_ll,
        **metrics,
    }


def get_all_ego_ids(root_dir):
    egonet_dir = os.path.join(root_dir, "egonets")
    ego_ids = []
    for fname in os.listdir(egonet_dir):
        if fname.endswith(".egonet"):
            try:
                ego_ids.append(int(os.path.splitext(fname)[0]))
            except ValueError:
                continue
    ego_ids.sort()
    return ego_ids


def run_all_egos(root_dir, K_candidates=(1, 2, 3, 4, 5),
                 device="cpu",
                 max_outer_iters=5,
                 param_steps=80,
                 circle_sweeps=3):
    """
    跑全部 ego：
      - 自動掃 egonets/*.egonet
      - 每個 ego 用 BIC 自動挑 K + 評估
      - 印出 summary
    """
    ego_ids = get_all_ego_ids(root_dir)
    print(f"Found {len(ego_ids)} egos: {ego_ids}")

    results = []
    for ego_id in ego_ids:
        res = train_one_ego_with_autoK(
            root_dir=root_dir,
            ego_id=ego_id,
            K_candidates=K_candidates,
            device=device,
            max_outer_iters=max_outer_iters,
            param_steps=param_steps,
            circle_sweeps=circle_sweeps,
        )
        results.append(res)
        print(
            f"[Ego {ego_id}] "
            f"K={res['best_K']}, "
            f"F1_micro={res['f1_micro']:.4f}, "
            f"F1_macro={res['f1_macro']:.4f}, "
            f"NMI={res['nmi']:.4f}, "
            f"ARI={res['ari']:.4f}"
        )

    # 全部平均一下
    if len(results) > 0:
        avg_f1_micro = np.mean([r["f1_micro"] for r in results])
        avg_f1_macro = np.mean([r["f1_macro"] for r in results])
        avg_nmi = np.mean([r["nmi"] for r in results])
        avg_ari = np.mean([r["ari"] for r in results])

        print("\n=== Overall Summary ===")
        print(f"Avg F1_micro = {avg_f1_micro:.4f}")
        print(f"Avg F1_macro = {avg_f1_macro:.4f}")
        print(f"Avg NMI      = {avg_nmi:.4f}")
        print(f"Avg ARI      = {avg_ari:.4f}")

    return results


# ============================================================
# 5. main: 示範怎麼呼叫
# ============================================================

if __name__ == "__main__":
    # 這裡換成你自己的 root_dir
    root_dir = ""

    # (1) 只跑一個 ego，順便自動挑 K + 評估
    # ego_id = 0
    # res = train_one_ego_with_autoK(
    #     root_dir=root_dir,
    #     ego_id=ego_id,
    #     K_candidates=(1, 2, 3, 4, 5),
    #     device="cpu",
    #     max_outer_iters=5,
    #     param_steps=80,
    #     circle_sweeps=3,
    # )
    # print("\nSingle ego result:")
    # print(res)

    # (2) 跑全部 ego
    results = run_all_egos(
        root_dir=root_dir,
        K_candidates=(1, 2, 3, 4, 5),
        device="cpu",
        max_outer_iters=5,
        param_steps=80,
        circle_sweeps=3,
    )
