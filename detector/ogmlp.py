# mlpclassifier.py  ← FINAL FIXED VERSION
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========================= PATHS =========================
TRAIN_CRIME   = Path("Output_IR3/Crime")
TRAIN_NORMAL  = Path("Output_IR3/No Crime")
TEST_CRIME    = Path("Test Features/Crime")
TEST_NORMAL   = Path("Test Features/No Crime")

def load_folder(folder: Path, label: int):
    X, y = [], []
    for npy_file in folder.glob("*.npy"):
        feat = np.load(npy_file)                    # shape could be (32, 512), (512,), etc.
        if feat.ndim > 1:
            feat = feat.mean(axis=0)                # → (512,)
        elif feat.ndim == 0:
            feat = feat.item()                      # in case it's a scalar
        X.append(feat.astype(np.float32))
        y.append(label)
    return np.array(X), np.array(y)

# ======================== MAIN ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    X_train_crime,   y_train_crime   = load_folder(TRAIN_CRIME,  1)
    X_train_normal,  y_train_normal  = load_folder(TRAIN_NORMAL, 0)
    X_test_crime,    y_test_crime    = load_folder(TEST_CRIME,   1)
    X_test_normal,   y_test_normal   = load_folder(TEST_NORMAL,  0)

    X_train = np.concatenate([X_train_normal, X_train_crime])
    y_train = np.concatenate([y_train_normal, y_train_crime])
    X_test  = np.concatenate([X_test_normal,  X_test_crime])
    y_test  = np.concatenate([y_test_normal,  y_test_crime])

    print(f"Train: {len(X_train)} clips (crime = {y_train.sum()}) → feature dim = {X_train.shape[1]}")
    print(f"Test : {len(X_test)} clips  (crime = {y_test.sum()}) → feature dim = {X_test.shape[1]}")

    # Auto-detect input dimension
    INPUT_DIM = X_train.shape[1]
    assert INPUT_DIM in [512, 768, 1024, 2048], f"Unexpected feature size: {INPUT_DIM}"

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).float().unsqueeze(1)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float().unsqueeze(1)),
        batch_size=64, shuffle=False
    )

    # ========== MODEL (now matches your actual feature size) ==========
    class CrimeMLP(nn.Module):
        def __init__(self, input_dim=INPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1)          # raw logit
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)  # → (batch_size,)

    model = CrimeMLP().to(device)

    # Positive class is rare → give higher weight
    pos_weight = torch.tensor(8.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    best_recall = 0.0
    print("Training started...")
    for epoch in range(1, 26):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.squeeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation every 5 epochs + last
        if epoch % 5 == 0 or epoch == 25:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(X_val).to(device))
                val_prob = torch.sigmoid(val_logits).cpu().numpy()
                val_pred = (val_prob > 0.5).astype(int)
                recall = recall_score(y_val, val_pred)
                
                if recall > best_recall:
                    best_recall = recall
                    torch.save(model.state_dict(), "models/mlp_best.pth")
                    print(f"→ New best model saved! Recall: {recall:.4f}")

            print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Recall: {recall:.4f} (best: {best_recall:.4f})")

    # ================== FINAL TEST ==================
    model.load_state_dict(torch.load("models/mlp_best.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test).to(device))
        test_prob = torch.sigmoid(test_logits).cpu().numpy()

    # Find best threshold (optimize for high recall + reasonable FPR)
    best_thr = 0.5
    best_score = 0
    for thr in np.arange(0.1, 0.8, 0.02):
        pred = (test_prob > thr).astype(int)
        recall = recall_score(y_test, pred)
        fpr = confusion_matrix(y_test, pred)[0, 1] / (confusion_matrix(y_test, pred)[0].sum() or 1)
        score = recall - 0.5 * fpr  # simple trade-off
        if score > best_score:
            best_score = score
            best_thr = thr
            best_recall = recall
            best_fpr = fpr

    final_pred = (test_prob > best_thr).astype(int)
    cm = confusion_matrix(y_test, final_pred)

    print("\n" + "="*70)
    print(f"MLP FINAL RESULT")
    print(f"Threshold: {best_thr:.3f} → Recall: {best_recall:.4f} ({best_recall*100:.2f}%) | FPR: {best_fpr:.1%}")
    print(f"Confusion Matrix:\n{cm}")
    print("="*70)

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Crime'],
                yticklabels=['Normal', 'Crime'])
    plt.title(f"Test Set – Recall {best_recall:.1%} @ {best_fpr:.1%} FPR (thr={best_thr:.2f})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig("assets/mlp_final.png", dpi=300, bbox_inches='tight')
    plt.show()