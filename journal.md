# 🛠️ Debug & Experimentation Journal

## Project: Aegis X
### Author: Aditri
### Date Started: [16-08-2025] #date I made this md file

---

## 🌍 Environment Setup
- OS:  Windows 11
- GPU: RTX 4060
- Frameworks: TensorRT 10.13.0
- Python: 

---

## 📅 Daily Logs

### Day [1] - [16-08-2025]
**What I planned today:**
- RE -EXPORTING WITH DYNAMIC AXES
- CONVERT TO TensorRT ,like first experimant with FP16 then maybe  INT8(quunatization!)


**What actually happened:**
- I ran this code-
import torch
from feature_extract_kr import get_mc3_feature_extractor

model=get_mc3_feature_extractor('cuda')

model.eval()

dummy_input=torch.randn(1,3,16,112,112)

#EXPORT ONNX WITH DYNAMIC AXES

torch.onnx.export(
    model,
    dummy_input,
    "mc3_dynamic.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'frames'}, 
        'output': {0: 'batch_size'}
    },
    opset_version=12
    )

print("✅ Exported to mc3_dynamic.onnx with dynamic axes!")







**❗❗❗❗❗❗Errors/Issues:❗❗❗❗❗❗**
- Error message:  train) PS C:\Users\Aditri\Documents\GPU Version> & "C:/Users/Aditri/Documents/GPU Version/train/Scripts/python.exe" "c:/Users/Aditri/Documents/GPU Version/detector/heyo.pPS C:\Users\Aditri\Documents\GPU Version> & "C:/Users/Aditri/Documents/GPU Version/train/Scripts/Activate.ps1"
PS C:\Users\Aditri\Documents\GPU Version> & "C:/Users/Aditri/Documents/GPU Version/train/Scripts/Activate.ps1"
(train) PS C:\Users\Aditri\Documents\GPU Version> & "C:/Users/Aditri/Documents/GPU Version/train/Scripts/python.exe" "c:/Users/Aditri/Documents/GPU Version/detector/heyo.py"
C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=MC3_18_Weights.KINETICS400_V1`. You can also use `weights=MC3_18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Traceback (most recent call last):
  File "c:\Users\Aditri\Documents\GPU Version\detector\heyo.py", line 12, in <module>
    torch.onnx.export(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\onnx\__init__.py", line 396, in export
    export(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\onnx\utils.py", line 529, in export
    _export(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\onnx\utils.py", line 1467, in _export
    graph, params_dict, torch_out = _model_to_graph(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\onnx\utils.py", line 1087, in _model_to_graph
    graph, params, torch_out, module = _create_jit_graph(model, args)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\onnx\utils.py", line 971, in _create_jit_graph
    graph, torch_out = _trace_and_get_graph_from_model(model, args)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\onnx\utils.py", line 878, in _trace_and_get_graph_from_model
    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\jit\_trace.py", line 1501, in _get_trace_graph
    outs = ONNXTracedModule(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\jit\_trace.py", line 138, in forward
    graph, _out = torch._C._create_graph_by_tracing(
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\jit\_trace.py", line 129, in wrapper
    outs.append(self.inner(*trace_inputs))
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "c:\Users\Aditri\Documents\GPU Version\detector\feature_extract_kr.py", line 18, in forward
    return self.backbone(x)  # [B, 512]
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torchvision\models\video\resnet.py", line 251, in forward
    x = self.stem(x)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\container.py", line 240, in forward
    input = module(input)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\conv.py", line 725, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\torch\nn\modules\conv.py", line 720, in _conv_forward
    return F.conv3d(
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor   
(train) PS C:\Users\Aditri\Documents\GPU Version> 


# MEANING OF ERROR/WHY ERROR HAPPNENED??

This line->RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same

Means: My weights are on gpu but dummy input is on cpu 

**PyTorch is strict: input & model must be on the same device.**


# ✅SOLUTION✅✅✅✅?

Move dummy input → GPU
model = get_mc3_feature_extractor("cuda")  # put model on GPU
dummy_input = torch.randn(1, 3, 16, 112, 112).cuda()



**What actually happened:**
Since I wanted to convert to TensorRT so I ran this-trtexec --onnx=mc3_dynamic.onnx --saveEngine=mc3_fp32.engine --explicitBatch --optShapes=input:1x3x16x112x112 --minShapes=input:1x3x8x112x112 --maxShapes=input:4x3x32x112x112 --fp32

**❗❗❗❗❗❗Errors/Issues:❗❗❗❗❗❗**

(train) PS C:\Users\Aditri\Documents\GPU Version> trtexec --onnx=mc3_dynamic.onnx --saveEngine=mc3_fp32.engine --explicitBatch --optShapes=input:1x3x16x112x112 --minShapes=input:1x3x8x112x112 --maxShapes=input:4x3x32x112x112 --fp32
>>
trtexec : The term 'trtexec' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was       
included, verify that the path is correct and try again.
At line:1 char:1
+ trtexec --onnx=mc3_dynamic.onnx --saveEngine=mc3_fp32.engine --explic ...
+ ~~~~~~~
    + CategoryInfo          : ObjectNotFound: (trtexec:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(train) PS C:\Users\Aditri\Documents\GPU Version>



# MEANING OF ERROR/WHY ERROR HAPPNENED??

this is what happens the first time — this is not my bug, it just means Windows can’t find TensorRT’s trtexec binary in PATH


# ✅SOLUTION✅✅✅✅?

Instead of just typing trtexec, mentioin full path-for me...I ran this accordingly after finding path of trtexec in bin of my donwloaded tensorRT -

& "C:\Users\Aditri\Downloads\TensorRT-10.13.0.35.Windows.win10.cuda-11.8\TensorRT-10.13.0.35\bin\trtexec.exe" --onnx=mc3_dynamic.onnx --saveEngine=mc3_fp32.engine --explicitBatch --optShapes=input:1x3x16x112x112 --minShapes=input:1x3x8x112x112 --maxShapes=input:4x3x32x112x112 --fp32


**What actually happened:**
I tried to benchmark fp16 and fp32 engines and tried visulaising the throughput ad all so i Ran this code-
import cv2
import time
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from pathlib import Path

# ============= CONFIG =============
VIDEO_PATH = "Abuse034_x264.mp4"
N_FRAMES = 32   
HEIGHT, WIDTH = 112, 112
ONNX_PATH = "models/mc3_features.onnx"
ENGINE_FP32 = "mc3_fp32.engine"
ENGINE_FP16 = "mc3_fp16.engine"
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
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes=input:1x3x{N_FRAMES}x{HEIGHT}x{WIDTH}",
        "--iterations=200"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    fps_line = [l for l in lines if "Throughput:" in l][-1]  # last throughput
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






**❗❗❗❗❗❗Errors/Issues:❗❗❗❗❗❗**



(train) PS C:\Users\Aditri\Documents\GPU Version> & "C:/Users/Aditri/Documents/GPU Version/train/Scripts/python.exe" "c:/Users/Aditri/Documents/GPU Version/detector/godlevel_benchmarking.py"
Traceback (most recent call last):
  File "c:\Users\Aditri\Documents\GPU Version\detector\godlevel_benchmarking.py", line 78, in <module>
    fps_results.append({"backend": "TensorRT FP16", "fps": benchmark_trt(ENGINE_FP16)})
  File "c:\Users\Aditri\Documents\GPU Version\detector\godlevel_benchmarking.py", line 73, in benchmark_trt
    fps_line = [l for l in lines if "Throughput:" in l][-1]  # last throughput
IndexError: list index out of range
(train) PS C:\Users\Aditri\Documents\GPU Version>






# MEANING OF ERROR/WHY ERROR HAPPNENED??

TensorRT uses trtexec to find the last lines of my code accordingly but couldnt find






**What actually happened:**

Since I made my mc3 onnx dynamic I thought lets convert it to fp16 and fp32 engines so I ran thi on terminal-
rtexec --onnx=models/mc3_dynamic.onnx --saveEngine=models/mc3_fp32.engine --minShapes=input:1x3x16x112x112 --optShapes=input:1x3x16x112x112 --maxShapes=input:1x3x16x112x112




**❗❗❗❗❗❗Errors/Issues:❗❗❗❗❗❗**

[08/17/2025-20:09:49] [I] Application Compute Clock Rate: 2.25 GHz
[08/17/2025-20:09:49] [I] Application Memory Clock Rate: 8.001 GHz
[08/17/2025-20:09:49] [I] 
[08/17/2025-20:09:49] [I] Note: The application clock rates do not reflect the actual clock rates that the GPU is currently running at.
[08/17/2025-20:09:49] [I]
[08/17/2025-20:09:49] [I] TensorRT version: 10.13.0
[08/17/2025-20:09:49] [I] Loading standard plugins
[08/17/2025-20:09:49] [I] [TRT] [MemUsageChange] Init CUDA: CPU +91, GPU +0, now: CPU 9682, GPU 1088 (MiB)
[08/17/2025-20:09:54] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2164, GPU +8, now: CPU 12163, GPU 1096 (MiB)
[08/17/2025-20:09:54] [I] Start parsing network model.
[08/17/2025-20:09:54] [E] Error[3]: In node -1 with name:  and operator:  (parseFromFile): INVALID_VALUE: Assertion failed: stat(onnxModelFile, &sb) == 0 && S_ISREG(sb.st_mode): Input file cannot be found, or is not a regular file: models/mc3_dynamic.onnx
[08/17/2025-20:09:54] [E] Failed to parse onnx file
[08/17/2025-20:09:54] [I] Finished parsing network model. Parse time: 0.0205425
[08/17/2025-20:09:54] [E] Parsing model failed
[08/17/2025-20:09:54] [E] Failed to create engine from model or file.
[08/17/2025-20:09:54] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v101300] [b35] # C:\Users\Aditri\Downloads\TensorRT-10.13.0.35.Windows.win10.cuda-11.8\TensorRT-10.13.0.35\bin\trtexec.exe --onnx=models/mc3_dynamic.onnx --saveEngine=models/mc3_fp32.engine --minShapes=input:1x3x16x112x112 --optShapes=input:1x3x16x112x112 --maxShapes=input:1x3x16x112x112
(train) PS C:\Users\Aditri\Documents\GPU Version> 


immp error line->[E] Parsing model failed
[E] Failed to create engine from model or file.





# MEANING OF ERROR/WHY ERROR HAPPNENED??
TensorRT cannot always handle fully dynamic ONNX models with symbolic axes on Windows


if you are coonfused about what is symbolic axes :When you define an ONNX model, some tensor shapes might change depending on input.
For example, if your model takes videos, the number of frames might vary per input.

Shape might be (1, 3, N, 112, 112) where N is the number of frames.

ONNX can’t know N ahead of time, so it uses a symbolic variable, often called something like "T" in the graph.




**☕☕DISCOVERY🏡🏡**
I WAS SITTING in my room sipping vanilla flavored black coffee and got to know YES,WE CAN ABSOLUTELY CONVERT A DYNAMIC ONNX MODEL TO FP16 AND FP32 -BUT WITH CONDITIONS-

TensorRT can build FP32 or FP16 engines from a dynamic ONNX model.

But on Windows, fully dynamic symbolic axes cannot be left completely unspecified.

You must provide an optimization profile (minShapes, optShapes, maxShapes) for all symbolic axes.




**What actually happened:**

I ran this code-



import cv2
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import requests
from mc3_infer import MC3Runner
from frame_buffer import FrameBuffer

# ------------------- SETTINGS -------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FP16 = True  # Use FP16 for MC3 inference
VIDEO_PATH = "Abuse034_x264.mp4"  # Video file
TARGET_FPS = 40
PROCESS_EVERY = 2  # process every Nth frame
DISPLAY_WIDTH = 640
ALERT_DELAY = 5  # seconds

TELEGRAM_TOKEN = "8072760336:AAEY8qdsG25tit16wQkdeBvXh9e9zCXRhAc"
CHAT_ID = "7773650672"

# ------------------- TELEGRAM ALERT -------------------
def send_telegram_message(image, caption='CRIME DETECTED!!'):
    tmp_path = "crime_detected.jpg"
    cv2.imwrite(tmp_path, image)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': open(tmp_path, 'rb')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("[INFO] Telegram alert sent successfully")
        else:
            print("[ERROR] Telegram alert failed:", response.text)
    except Exception as e:
        print("[ERROR] Telegram send exception:", e)

# ------------------- LOAD SVM + PCA -------------------
# We move SVM to GPU using PyTorch for speed
def load_bundle(joblib_path):
    import joblib
    bundle = joblib.load(joblib_path)
    # Convert sklearn SVM weights to PyTorch tensor
    coef = torch.from_numpy(bundle['model'].coef_.astype(np.float32)).to(DEVICE)
    intercept = torch.from_numpy(bundle['model'].intercept_.astype(np.float32)).to(DEVICE)
    scaler_mean = torch.from_numpy(bundle['scaler'].mean_.astype(np.float32)).to(DEVICE)
    scaler_scale = torch.from_numpy(bundle['scaler'].scale_.astype(np.float32)).to(DEVICE)
    pca_mean = torch.from_numpy(bundle['pca'].mean_.astype(np.float32)).to(DEVICE)
    pca_components = torch.from_numpy(bundle['pca'].components_.astype(np.float32)).to(DEVICE)
    return coef, intercept, scaler_mean, scaler_scale, pca_mean, pca_components

def preprocess_features(X_np, scaler_mean, scaler_scale, pca_mean, pca_components):
    # X_np: numpy array (1, feature_dim)
    X = torch.from_numpy(X_np.astype(np.float32)).to(DEVICE)
    X_scaled = (X - scaler_mean) / scaler_scale
    X_centered = X_scaled - pca_mean
    X_pca = torch.matmul(X_centered, pca_components.t())
    return X_pca  # PyTorch tensor on GPU

def svm_predict(X_pca, coef, intercept):
    logits = torch.matmul(X_pca, coef.t()) + intercept
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).int()
    return int(pred.item()), float(probs.item())

# ------------------- MC3 + FRAME PREPROCESS -------------------
def preprocess_frame(frame):
    return cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)

def inference_job(clip_np, mc3_runner, coef, intercept, scaler_mean, scaler_scale, pca_mean, pca_components):
    timings = {}
    t0 = time.perf_counter()
    features = mc3_runner.extract_features(clip_np, fp16=FP16)  # FP16
    timings['mc3_time'] = time.perf_counter() - t0

    # features is already GPU tensor
    X_pca = preprocess_features(features.reshape(1, -1).cpu().numpy(),
                                scaler_mean, scaler_scale, pca_mean, pca_components)
    timings['pca_time'] = time.perf_counter() - t0 - timings['mc3_time']

    pred_label, crime_prob = svm_predict(X_pca, coef, intercept)
    timings['svm_time'] = time.perf_counter() - t0 - timings['mc3_time'] - timings['pca_time']

    return pred_label, crime_prob, timings

# ------------------- MAIN -------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}, FP16: {FP16}")
    print("[INFO] Loading MC3 engine...")
    mc3_runner = MC3Runner()
    coef, intercept, scaler_mean, scaler_scale, pca_mean, pca_components = load_bundle("best_crime_detector.joblib")
    frame_buffer = FrameBuffer(max_len=32)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video")
        exit()
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:
        source_fps = 30
    FRAME_TIME = 1.0 / min(TARGET_FPS, source_fps)

    # Warmup MC3
    print("Warming up MC3...")
    dummy_clip = np.zeros((3, 32, 112, 112), dtype=np.uint8)
    try:
        _ = mc3_runner.extract_features(dummy_clip, fp16=FP16)
        print("Warmup done")
    except Exception as e:
        print("Warmup failed:", e)

    executor = ThreadPoolExecutor(max_workers=1)
    last_future = None
    frame_count = 0
    start_time = time.time()
    pred_label = None
    crime_prob = None
    display_label = "Waiting..."
    profile_counter = 0
    acc_mc3 = acc_pca = acc_svm = 0.0
    crime_start_time = None

    def resize_with_aspect(frame, width=None):
        h, w = frame.shape[:2]
        if width is not None:
            ratio = width / w
            dim = (width, int(h * ratio))
        else:
            return frame
        return cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_proc = preprocess_frame(frame)
        frame_buffer.add_frame(frame_proc)

        # Submit async job
        if frame_buffer.is_full() and (frame_count % PROCESS_EVERY == 0):
            if last_future is None or last_future.done():
                clip = frame_buffer.get_clip()
                clip_copy = np.copy(clip)
                last_future = executor.submit(
                    inference_job,
                    clip_copy,
                    mc3_runner,
                    coef, intercept,
                    scaler_mean, scaler_scale,
                    pca_mean, pca_components
                )

        # Get results asynchronously
        if last_future is not None and last_future.done():
            try:
                res = last_future.result(timeout=0)
                pred_label, crime_prob, timings = res
                display_label = "Crime" if pred_label == 1 else "No Crime"
                acc_mc3 += timings.get('mc3_time', 0.0)
                acc_pca += timings.get('pca_time', 0.0)
                acc_svm += timings.get('svm_time', 0.0)
                profile_counter += 1
                if profile_counter % 20 == 0:
                    print(f"Avg mc3: {acc_mc3/profile_counter:.3f}s, "
                          f"pca: {acc_pca/profile_counter:.3f}s, "
                          f"svm: {acc_svm/profile_counter:.3f}s")
            except Exception as e:
                print("Background job error:", e)

        # Telegram alert
        if pred_label == 1:
            if crime_start_time is None:
                crime_start_time = time.time()
            elif time.time() - crime_start_time >= ALERT_DELAY:
                send_telegram_message(frame)
                crime_start_time = None
        else:
            crime_start_time = None

        frame_count += 1
        elapsed_total = time.time() - start_time
        fps_display = frame_count / elapsed_total if elapsed_total > 0 else 0

        # Display
        disp_frame = resize_with_aspect(frame, width=DISPLAY_WIDTH)
        color = (0, 0, 255) if pred_label == 1 else (0, 255, 0)
        cv2.putText(disp_frame, f"Prediction: {display_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if crime_prob is not None:
            cv2.putText(disp_frame, f"Crime Prob: {crime_prob:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(disp_frame, f"FPS: {fps_display:.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Crime Detector", disp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_loop = time.perf_counter() - loop_start
        if elapsed_loop < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed_loop)

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)





**❗❗❗❗❗❗Errors/Issues:❗❗❗❗❗❗**

(train) PS C:\Users\Aditri\Documents\GPU Version> & "C:/Users/Aditri/Documents/GPU Version/train/Scripts/python.exe" "c:/Users/Aditri/Documents/GPU Version/detector/fp16_32.py"
Device: cuda, FP16: True
[INFO] Loading MC3 engine...
[INFO] MC3 model loaded successfully.
[INFO] Using Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[INFO] Loading MC3 engine...
[INFO] MC3 model loaded successfully.
[INFO] Using Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[INFO] Using Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[INFO] Model expects input shape: ['batch_size', 3, 'frames', 112, 112]
Traceback (most recent call last):
  File "c:\Users\Aditri\Documents\GPU Version\detector\fp16_32.py", line 91, in <module>
    coef, intercept, scaler_mean, scaler_scale, pca_mean, pca_components = load_bundle("best_crime_detector.joblib")
  File "c:\Users\Aditri\Documents\GPU Version\detector\fp16_32.py", line 44, in load_bundle
    coef = torch.from_numpy(bundle['model'].coef_.astype(np.float32)).to(DEVICE)
  File "C:\Users\Aditri\Documents\GPU Version\train\lib\site-packages\sklearn\svm\_base.py", line 660, in coef_
    raise AttributeError("coef_ is only available when using a linear kernel")
AttributeError: coef_ is only available when using a linear kernel. Did you mean: 'coef0'?
(train) PS C:\Users\Aditri\Documents\GPU Version>




# MEANING OF ERROR/WHY ERROR HAPPNENED??



My SVM is not linear. Only linear SVMs have a .coef_ attribute- If your model uses RBF or polynomial kernels, we cannot extract coefficients directly to use on GPU like we tried.

This means the approach of converting SVM to PyTorch tensor won’t work


Option 1: Keep SVM on CPU

Don’t try to GPU-accelerate the SVM.

Keep MC3 feature extraction on GPU (FP16), but pass features to the sklearn SVM on CPU.

This is simpler and works with any kernel.

# In inference_job:
features = mc3_runner.extract_features(clip_np, fp16=FP16)
feat_batch = features.reshape(1, -1).astype(np.float32)
crime_prob = model.predict_proba(feat_batch)[0][1]
pred_label = int(model.predict(feat_batch)[0])

Option 2: Use Linear SVM

Retrain your SVM with kernel='linear'.

Then .coef_ is available and you can move it to GPU.

Best for max GPU acceleration.