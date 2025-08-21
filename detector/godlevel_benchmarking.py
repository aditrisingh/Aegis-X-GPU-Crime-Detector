import cv2
import time
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from pathlib import Path
import os

# ============= CONFIG =============
VIDEO_PATH = "Abuse034_x264.mp4"
N_FRAMES = 32   
HEIGHT, WIDTH = 112, 112
ONNX_PATH = "models/mc3_features.onnx"
ENGINE_FP32 = "models/mc3_fp32.engine"
ENGINE_FP16 = "models/mc3_fp16.engine"
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)
# =================================

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

# Ensure enough frames
if len(frames) < N_FRAMES:
    raise ValueError(f"Video has only {len(frames)} frames, need at least {N_FRAMES}")

# Select first N_FRAMES
clip = np.stack(frames[:N_FRAMES])  # (T, H, W, C)
clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
clip = np.expand_dims(clip, axis=0).astype(np.float32)  # (1, C, T, H, W)

fps_results = []

# ---- ONNX benchmark ----
backends = {"ONNX CPU": ['CPUExecutionProvider']}
if ort.get_device() == "GPU":
    backends["ONNX GPU"] = ['CUDAExecutionProvider']

for name, providers in backends.items():
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    start = time.time()
    for _ in range(20):
        _ = session.run(None, {input_name: clip})
    end = time.time()

    fps = 20 / (end - start)
    fps_results.append({"backend": name, "fps": fps})

# ---- TensorRT benchmark (FP32 and FP16) ----
def benchmark_trt(engine_path):
    """Runs trtexec on an engine and parses FPS (throughput)."""
    if not os.path.exists(engine_path):
        print(f"⚠️ Engine file not found: {engine_path}")
        return 0.0

    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes=input:1x3x{N_FRAMES}x{HEIGHT}x{WIDTH}",
        "--iterations=200"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    lines = result.stdout.splitlines()
    fps_lines = [l for l in lines if "Throughput:" in l]

    if not fps_lines:  # if empty → print debug info
        print(f"⚠️ trtexec failed for {engine_path}")
        print("----- stdout -----")
        print(result.stdout)
        print("----- stderr -----")
        print(result.stderr)
        return 0.0  # safe fallback

    fps_line = fps_lines[-1]
    throughput = float(fps_line.split("Throughput:")[1].split("qps")[0].strip())
    return throughput

fps_results.append({"backend": "TensorRT FP32", "fps": benchmark_trt(ENGINE_FP32)})
fps_results.append({"backend": "TensorRT FP16", "fps": benchmark_trt(ENGINE_FP16)})

# ---- Save results and plot ----
df = pd.DataFrame(fps_results)
print(df)

plt.figure(figsize=(7,4))
plt.title(f"Throughput (FPS) per backend ({N_FRAMES} frames)")
plt.bar(df["backend"], df["fps"], color=["blue", "orange", "green", "red"])
plt.ylabel("FPS")
plt.savefig(PLOTS / f"throughput_vs_backend_{N_FRAMES}f.png")
plt.close()
