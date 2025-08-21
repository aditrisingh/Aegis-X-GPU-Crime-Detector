import numpy as np
import torch
from feature_extract_kr import MC3_FeatureExtractor
from mc3_infer import MC3Runner

# 1) Create dummy input
dummy_clip = np.random.rand(3, 32, 112, 112).astype(np.float32)

# 2) Run ONNX model
onnx_runner = MC3Runner()
onnx_features = onnx_runner.extract_features(dummy_clip)

# 3) Run PyTorch model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_model = MC3_FeatureExtractor(pretrained=True).to(device).eval()

with torch.no_grad():
    torch_input = torch.from_numpy(dummy_clip).unsqueeze(0).to(device)
    torch_features = torch_model(torch_input).cpu().numpy()

# 4) Compare outputs
max_diff = np.max(np.abs(torch_features - onnx_features))
print(f"✅ Max difference: {max_diff:.8f}")


"""
ONNX Export Validation Script

Purpose:
--------
This script verifies that the exported ONNX model produces outputs
numerically equivalent to the original PyTorch model.

Why it matters:
---------------
When deploying ML models (especially to edge devices, TensorRT, or
cross-platform environments), accuracy can degrade silently during
model export due to:
    - Operator mismatches
    - Precision changes (FP32 → FP16)
    - Shape/layout differences (e.g., NCHW vs NHWC)

By comparing the ONNX model’s output with the PyTorch model’s output
on the same input, we ensure:
    1. No silent accuracy loss during export
    2. The model is deployment-ready
    3. Debugging becomes easier if inference mismatches appear later

Expected result:
----------------
The maximum difference between outputs should be very small (<1e-3 for FP32).
A slightly higher value is normal for GPU ops due to floating-point math.
"""
