# 🛡️ Aegis-X GPU — Real-time Crime Detector (TensorRT + MC3-18 + SVM)

**Production pipeline (main branch)**  
→ Full 2048-dim MC3-18 features + RBF SVM  
→ **83% precision, 86% recall, 0.88 AUC** on UCF-Crime (within 4% of 2023 SOTA)  
→ 20–40 FPS on RTX 4060  
→ Telegram alerts with screenshots (>5s streak)

<p align="center"> <img src="assets/aegis x last.png" width="180" alt="AegisX Logo"> </p>  
<h1 align="center">🚨 AegisX: Real-time Crime Detection from CCTV Footage</h1>

✨ **What is AegisX?**  
Lightweight, production-ready crime & anomaly detection system for CPU and GPU deployment.  
Uses frozen MC3-18 (3D CNN) as feature extractor + SVM classifier.  
GPU version adds TensorRT FP16/FP32 engines for massive speedups.

💡 **Detectable anomalies:** Fighting · Robbery · Vandalism · Assault · Abuse

### 🎯 Key Features
- Real-time inference on CPU (ONNX) and GPU (TensorRT FP16/FP32)
- Telegram alerts with timestamped screenshots (>5 sec streak)
- Full evaluation suite: ROC, PR, confusion matrix, per-class metrics
- Benchmarks: ONNX vs TensorRT FP16 vs FP32
- Transparent development journal (journal.md) — every error & fix logged

### 📊 Production Performance (main branch)
- Accuracy: **83% precision, 86% recall, 0.88 AUC** (full 2048-dim features)
- Speed: **90–120 FPS** on RTX 4060 (TensorRT FP16)
- Same robust pipeline as CPU repo

### ⚠️ Ablation Note (old-experiments branch)
Early experiment with PCA (95% variance) → dropped accuracy to **~76%** for zero real speed gain.  
Kept in separate branch as a lesson: never aggressively compress strong 3D features.

### 📷 Telegram Alert Example
<p align="center"> <img src="assets/last breath.jpg" width="500"> </p>

### 🎥 Demo
<p align="center"> <img src="last gif.gif" width="700"> </p>

### ⚙️ Benchmarks
<p align="center"> <img src="assets/throughput_vs_backend_32f.png" width="600"> </p>

### 📂 Dataset & Features
UCF-Crime anomaly dataset  
Pre-extracted MC3 features available on request → aditwisingh@gmail.com

### 🚀 Quick Start
```bash
git clone https://github.com/aditrisingh/Aegis-X-GPU-Crime-Detector.git
cd Aegis-X-GPU-Crime-Detector
python -m venv venv && source venv/Scripts/activate   # Windows
# ONNX version
python detector/gpu.py
# TensorRT FP16 (fastest)
python detector/fp_16.py
# TensorRT FP32
python detector/fp_32.py
