# Aegis-X GPU — Real-time Crime Detector (TensorRT + MC3-18 + **MLP**)  
**93 % Recall · Only 7 missed crimes out of 100 · RTX 4060 · Nov 28 2025**

<p align="center"> <img src="assets/aegis x last.png" width="180" alt="AegisX Logo"> </p>
<h1 align="center">AegisX: Real-time Crime Detection from CCTV Footage</h1>

**✨ NOV 28 2025 — MAJOR UPGRADE (main branch)**  
→ Replaced RBF SVM with **2-layer MLP head**  
→ **93 % recall** on held-out test set (93/100 violent clips detected)  
→ Only **7 missed crimes** total  
→ Zero FPS drop — still **90–120 FPS** on RTX 4060 (TensorRT FP16)  
→ Training: <15 min · Inference: <0.4 ms  

![93% Recall Confusion Matrix](assets/93%20recall%20mlp.png)

| Version            | Recall | Missed Crimes | Speed       | Date       |
|--------------------|--------|---------------|-------------|------------|
| Old RBF SVM        | 86 %   | 14/100        | 90–120 FPS  | Aug 2025   |
| **MLP (current)**  | **93 %** | **7/100**   | 90–120 FPS  | Nov 28 2025|

**Production pipeline (main branch) — now uses the 93 % recall model**  
→ Full MC3-18 features + clean MLP classifier  
→ Full reproducible training: [`mlpclassifier.py`](mlpclassifier.py)  
→ Production inference: [`mlp_inference.py`](mlp_inference.py)  
→ Model: `mlp_best.pth`


### Key Features
- Real-time inference on CPU (ONNX) and GPU (TensorRT FP16/FP32)
- Telegram alerts with screenshots (>5s streak)
- Full evaluation suite: ROC, PR, confusion matrix, per-class metrics
- Benchmarks: ONNX vs TensorRT FP16 vs FP32
- Transparent development journal (journal.md) — every error & fix logged

### Detectable anomalies:
Fighting · Robbery · Vandalism · Assault · Abuse

### Telegram Alert Example
<p align="center"> <img src="assets/last breath.jpg" width="500"> </p>

### Demo (now running the 93 % recall model)
<p align="center"> <img src="last gif.gif" width="700"> </p>

### Benchmarks (unchanged — MLP adds zero latency)
<p align="center"> <img src="assets/throughput_vs_backend_32f.png" width="600"> </p>

### Dataset & Features
UCF-Crime anomaly dataset  
Pre-extracted MC3 features available on request → aditwisingh@gmail.com

### Quick Start
```bash
git clone https://github.com/aditrisingh/Aegis-X-GPU-Crime-Detector.git
cd Aegis-X-GPU-Crime-Detector
python -m venv venv && source venv/Scripts/activate

# TensorRT FP16 (fastest — now uses 93% recall MLP)
python detector/fp_16.py

# ONNX version (also uses MLP)
python detector/gpu.py
