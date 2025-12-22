# Learning Social Circles in Ego-Networks

This repository implements both **unsupervised** and **supervised** approaches for social circle discovery in ego-networks using graph representation learning.

The unsupervised pipeline employs a **Graph Autoencoder (GAE)** followed by **DBSCAN clustering**, while the supervised pipeline uses a **Graph Neural Network (GNN)** for multi-label social circle prediction.

---

## Environment Setup

We recommend using a clean conda environment.

```bash
conda create -n social_network python=3.10
conda activate social_network

pip install -r requirements.txt
```

## Project Structure

learning-social-circles/
├── egonets/               # Ego-network datasets
├── Training/              # Ego-network labels
├── model.py               # Vanilla GNN, MLP, and baseline models
├── util.py                # Utility functions (metrics, visualization, data processing)
├── unsupervise.py         # Unsupervised GAE + DBSCAN 
├── supervise_GNN.py       # Supervised GNN training and evaluation
├── requirements.txt
└── README.md
# for the unsupervise learning model 

python unsupervise.py

# for the supervise GNN model 

python supervise_GNN.py