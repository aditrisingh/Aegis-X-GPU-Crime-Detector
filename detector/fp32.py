import os
import time
import gc
import cv2
import joblib
import torch
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import tensorrt as trt

# ---------------- CONFIG ----------------
ENGINE_PATH = "models/mc3_fp32.engine"
JOBLIB_PATH  = "best_crime_detector2.joblib"
VIDEO_PATH  = "Fighting035_x264.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_FPS = 40
PROCESS_EVERY = 2
DISPLAY_WIDTH = 640
ALERT_DELAY = 5  # seconds

# ---------------- TELEGRAM ----------------
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_message(image, caption="CRIME DETECTED!!"):
    if not TELEGRAM_TOKEN or "YOUR_" in TELEGRAM_TOKEN:
        print("[WARN] Telegram token not set — skipping send.")
        return
    tmp_path = "crime_detected.jpg"
    cv2.imwrite(tmp_path, image)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {"photo": open(tmp_path, "rb")}
    data = {"chat_id": CHAT_ID, "caption": caption}
    try:
        r = requests.post(url, files=files, data=data, timeout=10)
        if r.status_code == 200:
            print("[INFO] Telegram alert sent.")
        else:
            print("[ERROR] Telegram failed:", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] Telegram exception:", e)

# ---------------- FRAME BUFFER ----------------
class FrameBuffer:
    def __init__(self, max_len=32):
        self.max_len = max_len
        self.frames = []

    def add_frame(self, frame_hwc):
        if len(self.frames) >= self.max_len:
            self.frames.pop(0)
        self.frames.append(frame_hwc)

    def is_full(self):
        return len(self.frames) == self.max_len

    def get_clip(self):
        return np.stack(self.frames, axis=0)  # (T,H,W,C)

# ---------------- MC3 Runner ----------------
class MC3Runner:
    def __init__(self, engine_path=ENGINE_PATH, clip_len=32, logger_severity=trt.Logger.WARNING):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        self.clip_len = clip_len
        self.logger = trt.Logger(logger_severity)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name  = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        base_in_shape  = tuple(self.engine.get_tensor_shape(self.input_name))
        base_out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_shape = tuple(self.clip_len if d == -1 else d for d in base_in_shape)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        print("[INFO] TRT FP32 engine loaded.")
        print("[INFO] input tensor:", self.input_name, "base shape:", base_in_shape, "using:", self.input_shape)
        print("[INFO] output tensor:", self.output_name, "base shape:", base_out_shape)

    @torch.inference_mode()
    def extract_features(self, clip_chw_t: np.ndarray) -> np.ndarray:
        clip_batched = clip_chw_t.astype(np.float32, copy=False)

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                d_input = torch.from_numpy(clip_batched).to("cuda", non_blocking=True)
                self.context.set_input_shape(self.input_name, tuple(d_input.shape))
                out_shape = tuple(self.context.get_tensor_shape(self.output_name))
                out_shape = tuple(1 if d == -1 else d for d in out_shape)
                d_output = torch.empty(out_shape, dtype=torch.float32, device="cuda")
                self.context.set_tensor_address(self.input_name,  d_input.data_ptr())
                self.context.set_tensor_address(self.output_name, d_output.data_ptr())
                ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v3 returned False")
            self.stream.synchronize()
            return d_output.detach().cpu().numpy()

        # CPU fallback
        d_input = torch.from_numpy(clip_batched)
        self.context.set_input_shape(self.input_name, tuple(d_input.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_shape = tuple(1 if d == -1 else d for d in out_shape)
        d_output = torch.empty(out_shape, dtype=torch.float32)
        self.context.set_tensor_address(self.input_name,  d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, d_output.data_ptr())
        ok = self.context.execute_async_v3(stream_handle=0)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 (fallback) returned False")
        return d_output.cpu().numpy()

# ---------------- MODEL UTILITIES ----------------
def load_bundle(joblib_path=JOBLIB_PATH):
    print(f"[INFO] Loading FP32 model bundle from {joblib_path} ...")
    bundle = joblib.load(joblib_path)
    return bundle["model"], bundle["scaler"], bundle["pca"]

def prepare_gpu_pca_fp32(scaler, pca, device=DEVICE):
    scaler_mean  = torch.from_numpy(scaler.mean_.astype(np.float32)).to(device)
    scaler_scale = torch.from_numpy(scaler.scale_.astype(np.float32)).to(device)
    pca_mean     = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
    components   = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
    return scaler_mean, scaler_scale, pca_mean, components

def gpu_preprocess_pca_fp32(X_np, scaler_mean, scaler_scale, pca_mean, components):
    X = torch.from_numpy(X_np.astype(np.float32)).to(DEVICE)
    X_scaled = (X - scaler_mean) / scaler_scale
    X_centered = X_scaled - pca_mean
    X_pca = torch.matmul(X_centered, components.t())
    return X_pca.cpu().numpy()

def inference_job_fp32(clip_chw_t, model, scaler_mean, scaler_scale, pca_mean, components, mc3_runner):
    features = mc3_runner.extract_features(clip_chw_t)
    feat_batch = features.reshape(1, -1).astype(np.float32, copy=False)
    X_pca = gpu_preprocess_pca_fp32(feat_batch, scaler_mean, scaler_scale, pca_mean, components)
    if hasattr(model, "predict_proba"):
        crime_prob = float(model.predict_proba(X_pca)[0][1])
    else:
        crime_prob = None
    pred_label = int(model.predict(X_pca)[0])

    del features, X_pca
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return pred_label, crime_prob

# ---------------- MAIN ----------------
def main():
    model, scaler, pca = load_bundle(JOBLIB_PATH)
    scaler_mean, scaler_scale, pca_mean, components = prepare_gpu_pca_fp32(scaler, pca)
    mc3 = MC3Runner(ENGINE_PATH, clip_len=32)
    frame_buffer = FrameBuffer(max_len=32)

    cap = cv2.VideoCapture(VIDEO_PATH if VIDEO_PATH != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    FRAME_TIME = 1.0 / min(TARGET_FPS, source_fps)

    print("Warming up MC3...")
    dummy_hwc = np.zeros((32,112,112,3), dtype=np.uint8)
    clip_chw_t = np.expand_dims(np.transpose(dummy_hwc, (3,0,1,2)), axis=0)
    _ = mc3.extract_features(clip_chw_t)
    print("Warmup done.")

    executor = ThreadPoolExecutor(max_workers=2)
    futures = deque(maxlen=2)

    frame_count = 0
    pred_label = None
    crime_prob = None
    display_label = "Waiting..."
    crime_start_time = None
    start_time = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_hwc = cv2.resize(frame, (112, 112))
        frame_buffer.add_frame(frame_hwc)

        # Submit async inference
        if frame_buffer.is_full() and frame_count % PROCESS_EVERY == 0:
            if len(futures) < 2 or all(f.done() for f in futures):
                clip_hwc = frame_buffer.get_clip()
                clip_chw_t = np.expand_dims(np.transpose(clip_hwc, (3,0,1,2)), axis=0)
                fut = executor.submit(inference_job_fp32, np.copy(clip_chw_t),
                                      model, scaler_mean, scaler_scale, pca_mean, components, mc3)
                futures.append(fut)

        # Collect results from futures
        for f in list(futures):
            if f.done():
                try:
                    pred_label, crime_prob = f.result(timeout=0)
                    display_label = "Crime" if pred_label == 1 else "No Crime"
                except Exception as e:
                    print("[ERROR] Background job error:", e)
                futures.remove(f)

        # Handle alerts
        if pred_label == 1:
            if crime_start_time is None:
                crime_start_time = time.time()
            elif time.time() - crime_start_time >= ALERT_DELAY:
                send_telegram_message(frame)
                crime_start_time = None
        else:
            crime_start_time = None

        frame_count += 1
        total_elapsed = time.perf_counter() - start_time
        fps_display = frame_count / max(total_elapsed, 1e-5)

        # Display
        disp = cv2.resize(frame, (DISPLAY_WIDTH, int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])))
        color = (0,0,255) if pred_label == 1 else (0,255,0)
        cv2.putText(disp, f"Prediction: {display_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if crime_prob is not None:
            cv2.putText(disp, f"Prob: {crime_prob:.2f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(disp, f"FPS: {fps_display:.1f}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Crime Detector (TRT FP32)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed_loop = time.perf_counter() - loop_start
        if elapsed_loop < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed_loop)

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
