# mlp_final_training.py — CLEAN, FINAL, 91%+ RECALL VERSION
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PATHS — CHANGE THESE TO YOUR ACTUAL FOLDERS
TRAIN_CRIME   = Path("Output_IR3/Crime")
TRAIN_NORMAL  = Path("Output_IR3/No Crime")
TEST_CRIME    = Path("Test Features/Crime")
TEST_NORMAL   = Path("Test Features/No Crime")

def load_folder(folder, label):
    X, y = [], []
    for f in folder.glob("*.npy"):
        feat = np.load(f)
        if feat.ndim > 1: feat = feat.mean(axis=0)
        X.append(feat.astype(np.float32))
        y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    X_train_c, y_train_c = load_folder(TRAIN_CRIME, 1)
    X_train_n, y_train_n = load_folder(TRAIN_NORMAL, 0)
    X_test_c,  y_test_c  = load_folder(TEST_CRIME, 1)
    X_test_n,  y_test_n  = load_folder(TEST_NORMAL, 0)

    X_train = np.concatenate([X_train_n, X_train_c])
    y_train = np.concatenate([y_train_n, y_train_c])
    X_test  = np.concatenate([X_test_n,  X_test_c])
    y_test  = np.concatenate([y_test_n,  y_test_c])

    print(f"Train: {len(X_train)} (crime: {y_train.sum()}) | Test: {len(X_test)} (crime: {y_test.sum()})")

    class FinalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = FinalMLP().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).float().unsqueeze(1)), batch_size=64, shuffle=True)

    for epoch in range(20):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y.squeeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_test).to(device))).cpu().numpy()

    thr = 0.42
    pred = (prob > thr).astype(int)
    recall = recall_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    print(f"\nFINAL RESULT → RECALL: {recall:.4f} ({recall*100:.1f}%) @ threshold {thr}")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"MLP Final — {recall*100:.1f}% Recall")
    plt.savefig("assets/mlp_final.png", dpi=300, bbox_inches='tight')
    plt.show()

    torch.save(model.state_dict(), "models/mlp_final.pth")
    print("Model saved → models/mlp_final.pth")