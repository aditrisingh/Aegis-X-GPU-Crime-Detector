import cv2
import time
import numpy as np
import joblib
import requests
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from frame_buffer import FrameBuffer

# -------------------------------
# Device setup
# -------------------------------
DEVICE = 'cuda' if ort.get_device() == 'GPU' else 'cpu'
print(f"[INFO] Using device: {DEVICE}")

# -------------------------------
# Hardcoded paths & Telegram config
# -------------------------------
VIDEO_PATH = "detector/Videos/Normal_Videos084_x264.mp4"

JOBLIB_PATH = "best_crime_detector2.joblib"
MC3_ONNX_PATH = "models/mc3_features.onnx"
TELEGRAM_BOT_TOKEN = "8072760336:AAEY8qdsG25tit16wQkdeBvXh9e9zCXRhAc"
TELEGRAM_CHAT_ID = "7773650672"

# FPS control
TARGET_FPS = 40
FRAME_TIME = 1.0 / TARGET_FPS
PROCESS_EVERY = 2
DISPLAY_SIZE = (640, 360)
CRIME_ALERT_DURATION = 5.0  # seconds

# -------------------------------
# Load joblib pipeline
# -------------------------------
print("[INFO] Loading joblib pipeline...")
pipeline = joblib.load(JOBLIB_PATH)
clf = pipeline['model']
scaler = pipeline['scaler']
pca = pipeline['pca']
print("[INFO] Joblib components loaded successfully.")

# -------------------------------
# MC3 ONNX Runner
# -------------------------------
class MC3ONNXRunner:
    def __init__(self, model_path=MC3_ONNX_PATH, device=DEVICE):
        providers = ['CUDAExecutionProvider'] if device=='cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract_features(self, clip_np):
        # clip_np shape: (C, T, H, W) => ONNX expects (1, C, T, H, W)
        clip_input = clip_np.astype(np.float32)[np.newaxis, ...]
        features = self.session.run([self.output_name], {self.input_name: clip_input})[0]
        return features

# -------------------------------
# Helper: Resize frame
# -------------------------------
def preprocess_frame(frame):
    return cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)

# -------------------------------
# Helper: GPU PCA preprocessing
# -------------------------------
def gpu_preprocess_and_pca(X_np, scaler, pca):
    # ONNX already outputs float32, so we can directly apply scaler/pca on CPU
    X_scaled = (X_np - scaler.mean_) / scaler.scale_
    X_centered = X_scaled - pca.mean_
    X_pca = np.dot(X_centered, pca.components_.T)
    return X_pca

# -------------------------------
# Helper: Telegram notification
# -------------------------------
def send_telegram_alert(image, caption="Crime detected!"):
    _, buffer = cv2.imencode(".jpg", image)
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": buffer.tobytes()}
        )
        print("[INFO] Telegram alert sent.")
    except Exception as e:
        print("[ERROR] Telegram alert failed:", e)

# -------------------------------
# Inference job
# -------------------------------
def inference_job(clip_np, clf, scaler, pca, mc3_runner):
    features = mc3_runner.extract_features(clip_np)
    if features.ndim == 2 and features.shape[0] == 1:
        features = features.reshape(-1)
    feat_batch = features.reshape(1, -1).astype(np.float32)
    X_pca = gpu_preprocess_and_pca(feat_batch, scaler, pca)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_pca)[0]
        crime_prob = float(probs[1])
    else:
        crime_prob = None
    pred_label = int(clf.predict(X_pca)[0])
    return pred_label, crime_prob

# -------------------------------
# Main loop
# -------------------------------
def main(video_path):
    mc3_runner = MC3ONNXRunner()
    frame_buffer = FrameBuffer(max_len=32)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Warmup MC3
    print("[INFO] Warming up MC3...")
    try:
        dummy_clip = np.zeros((3, 32, 112, 112), dtype=np.float32)
        _ = mc3_runner.extract_features(dummy_clip)
        print("[INFO] Warmup done.")
    except Exception as e:
        print("[WARN] Warmup failed:", e)

    executor = ThreadPoolExecutor(max_workers=1)
    last_future = None
    frame_count = 0
    start_time = time.time()

    # Initialize prediction variables
    pred_label = 0
    crime_prob = 0.0
    crime_start_time = None

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break
        frame_count += 1

        # Resize frame for buffer
        frame_resized = preprocess_frame(frame)
        frame_buffer.add_frame(frame_resized)

        # Submit inference job asynchronously
        if frame_buffer.is_full() and (frame_count % PROCESS_EVERY == 0):
            if last_future is None or last_future.done():
                clip = np.copy(frame_buffer.get_clip())
                last_future = executor.submit(inference_job, clip, clf, scaler, pca, mc3_runner)

        # Retrieve inference results
        if last_future is not None and last_future.done():
            try:
                res = last_future.result(timeout=0)
                if res is not None:
                    pred_label, crime_prob = res
                    if pred_label == 1:
                        if crime_start_time is None:
                            crime_start_time = time.time()
                    else:
                        crime_start_time = None
            except Exception as e:
                print("[ERROR] Background job error:", e)

        # Send Telegram alert if crime detected for more than threshold
        if crime_start_time is not None and (time.time() - crime_start_time) > CRIME_ALERT_DURATION:
            send_telegram_alert(frame, caption=f"Crime detected with prob {crime_prob:.2f}")
            crime_start_time = None

        # Display
        disp_frame = cv2.resize(frame, DISPLAY_SIZE)
        label_text = "Crime" if pred_label == 1 else "No Crime"
        color = (0, 0, 255) if pred_label == 1 else (0, 255, 0)
        cv2.putText(disp_frame, f"Prediction: {label_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if crime_prob is not None:
            cv2.putText(disp_frame, f"Crime Prob: {crime_prob:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        fps = frame_count / (time.time() - start_time)
        cv2.putText(disp_frame, f"FPS: {fps:.1f}", (10, 110),
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

# -------------------------------
if __name__ == "__main__":
    main(VIDEO_PATH)
