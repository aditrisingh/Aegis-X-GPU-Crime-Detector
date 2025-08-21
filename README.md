🛡️ Aegis-X Crime Detector

Real-time, CPU/GPU-efficient video crime detection system using ONNX MC3 + PCA + SVM + TensorRT (FP16/FP32).
Trained on UCF Crime Dataset, achieving ~76% accuracy. Sends Telegram alerts for real-time monitoring.

<p align="center"> <img src="assets/aegis x last.png" width="180" alt="AegisX Logo"> </p> <h1 align="center">🚨 AegisX: Real-time Crime Detection from CCTV Footage</h1>
✨ What is AegisX?

AegisX is a lightweight, AI-driven crime detection system optimized for CPU and GPU deployment.
It extracts spatiotemporal features using an ONNX MC3 (3D convolution) backbone, reduces feature dimensions with PCA, and predicts crimes using an SVM classifier.
For faster inference, TensorRT engines (FP16/FP32) can be used.

💡 Detectable crimes:

Fighting

Robbery

Vandalism

Assault

⚠️ Performs best on CCTV-style surveillance footage, not mobile or cinematic videos.

🎯 Key Features

✅ Real-time video inference on CPU or GPU (TensorRT FP16/FP32)

✅ MC3-based spatiotemporal feature extraction (ONNX)

✅ PCA + SVM classifier (~76% accuracy on UCF Crime)

✅ Telegram alerts with snapshots if crime persists >5 seconds

✅ Benchmarking: ONNX vs FP16 vs FP32 (latency & throughput)

✅ Visualizations: ROC, PR Curve, Confusion Matrix, per-class metrics

✅ Development journal with debugging notes and fixes

⚙️ System Pipeline

Input: CCTV-style video stream or file

Frame Buffering: Temporal frames collected

Feature Extraction: ONNX MC3 backbone

Classification: PCA + SVM predicts crime/no-crime

Deployment: ONNX Runtime or TensorRT FP16/FP32 engines

Alert System: Telegram bot triggers screenshot + alert

Visualization: Metrics plots and benchmark graphs

📊 Benchmarks & Results

Performance comparison:

ONNX runtime vs TensorRT FP16 vs TensorRT FP32

Metrics include frames per second (FPS) and latency

<p align="center"> <img src="assets/throughput_vs_backend_32f.png" width="500" alt="ONNX vs TensorRT Benchmark"><br> <img src="assets/god_tier_performance_report.png" width="500" alt="Per-Class Accuracy"> </p>

Insights:

FP16 engine achieves highest FPS with minimal accuracy loss.

FP32 engine slightly slower but more stable on edge cases.

ONNX runtime provides flexibility without engine compilation.

📷 Telegram Alerts
<p align="center"> <img src="assets/last breath.jpg" width="500" alt="Telegram Alert Example"> </p>

Sends alert if crime persists for more than 5 seconds.

Includes detected frame, label, and confidence.

🎥 Demo GIF
<p align="center"> <img src="last gif.gif" width="600" alt="Crime Detection Demo"> </p>

Note: GitHub preview works best with .gif. Convert .mp4 videos to .gif if needed.

📂 Dataset

UCF Crime Dataset

Features pre-extracted and stored in .npy format for efficiency.

📧 For pre-extracted features, contact: aditwisingh@gmail.com

🛠️ Improvements & Notes

Engines: FP16/FP32 real-time inference; ONNX recommended for flexibility.

GPU support: Optional GPU acceleration for faster processing.

ThreadPoolExecutor: Parallelized frame processing for efficiency.

Development Journal: Notes on errors, solutions, and debugging in journal.md.

⚠️ Ignore these files — included for learning purposes only:

train_ensemble.py

mc3_infer.py

trying_models.py

🚀 How to Run AegisX
# Clone the repository
git clone https://github.com/aditrisingh/Aegis-X-Crime-Detector.git
cd Aegis-X-Crime-Detector

# Create virtual environment
python -m venv aegisx_env
source aegisx_env/Scripts/activate   # Windows: aegisx_env\Scripts\activate

# Run ONNX runtime detector
python detector/gpu.py 

# Run TensorRT detector (FP16)
python detector/fp_16.py 

# Run TensorRT detector (FP32)
python detector/fp_32.py


⚠️ Update config/telegram.json with your bot token and chat ID. Change video paths accordingly.

📝 Development Journal

See journal.md for a complete log of errors, debugging steps, and solutions during development.

🌍 About Me

Hi! I’m Aditri Singh 👋

🎓 CSE (AI & ML) @ Parul University

💻 Specializing in real-time computer vision & surveillance AI

🇳🇱 Actively seeking AI/ML roles in The Netherlands

🧠 Focused on system optimization and smart city applications

📜 License

MIT License © 2025 Aditri Singh

