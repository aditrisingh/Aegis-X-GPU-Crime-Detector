# mlp_inference.py ← THIS IS YOUR 93% RECALL MODEL IN PRODUCTION
import torch
import torch.nn as nn
import numpy as np

class CrimeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )


model = CrimeMLP()
model.load_state_dict(torch.load("models/mlp_best.pth", map_location="cpu"))
model.eval()

def is_crime(features: np.ndarray, threshold: float = 0.42) -> tuple[bool, float]:
    """Input: (32, 512) or (512,) MC3 features → Output: (crime?, confidence)"""
    if features.ndim > 1:
        features = features.mean(axis=0)          # temporal average
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(features).float())).item()

    return prob > threshold, prob
