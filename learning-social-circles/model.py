from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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

    def get_embedding(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

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
    
class LogisticRegressionBaseline(nn.Module):
    """
    Feature-only baseline: x -> Linear -> logits
    """
    def __init__(self, in_channels, num_circles):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_circles)

    def forward(self, x):
        logits = self.linear(x)
        return logits
    
class MLPBaseline(nn.Module):
    """
    Feature-only baseline: x -> MLP -> logits
    """
    def __init__(self, in_channels, hidden_channels, num_circles):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_circles),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits
    
class GNN_origin(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_circles):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_circles)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        logits = self.classifier(h)
        return logits