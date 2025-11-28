# mlp_inference.py — CLEAN, DROP-IN REPLACEMENT
import torch
import torch.nn as nn

class FinalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

model = FinalMLP()
model.load_state_dict(torch.load("models/mlp_final.pth", map_location="cpu"))
model.eval()

def predict_crime(features_np, threshold=0.42):
    if features_np.ndim > 1:
        features_np = features_np.mean(axis=0)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(features_np).float())).item()
    return prob > threshold, prob