'''import torch
from feature_extract_kr import get_mc3_feature_extractor

model=get_mc3_feature_extractor('cuda')

model.eval()

dummy_input=torch.randn(1,3,16,112,112).cuda()

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

print("✅ Exported to mc3_dynamic.onnx with dynamic axes!")'''


'''# eval_features.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------- CONFIG ----------------
FEATURE_DIR = "Test Features"
JOBLIB_PATH = "best_crime_detector.joblib"

# ---------------- LOAD ----------------
print(f"[INFO] Loading bundle: {JOBLIB_PATH}")
bundle = joblib.load(JOBLIB_PATH)
model, scaler, pca = bundle["model"], bundle["scaler"], bundle["pca"]

# ---------------- DATA ----------------
X, y = [], []
for fname in os.listdir(FEATURE_DIR):
    if not fname.endswith(".npy"):
        continue
    path = os.path.join(FEATURE_DIR, fname)
    feat = np.load(path)   # shape (512,)
    X.append(feat)

    # crude label from filename (adjust as per your naming)
    if "crime" in fname.lower():
        y.append(1)
    else:
        y.append(0)

X = np.stack(X).astype(np.float32)
y = np.array(y)

print(f"[INFO] Loaded {len(X)} samples from {FEATURE_DIR}")

# ---------------- PREPROCESS ----------------
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# ---------------- PREDICT ----------------
y_pred = model.predict(X_pca)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_pca)[:, 1]
else:
    y_prob = None

acc = accuracy_score(y, y_pred)
print(f"[RESULT] Accuracy = {acc*100:.2f}%")

# ---------------- VISUALIZE ----------------
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Crime", "Crime"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (FP32 features)")
plt.show()

if y_prob is not None:
    plt.hist([y_prob[y==0], y_prob[y==1]], bins=20, stacked=True, label=["No Crime", "Crime"])
    plt.xlabel("Predicted Probability of Crime")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Probability Distribution")
    plt.show()'''




import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# Load dataset (FP16 MC3 features)
# -------------------------
def load_features(base_path):
    X, y = [], []
    for label, folder in enumerate(["No Crime", "Crime"]):
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                # Convert to FP16 here to match MC3 engine
                data = np.load(os.path.join(folder_path, file)).astype(np.float16)
                X.append(data)
                y.append(label)
    return np.array(X, dtype=np.float16), np.array(y)

base_path = "Output_IR3"  # replace with your feature folder
X, y = load_features(base_path)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.astype(np.float32))  # sklearn needs float32/64
X_test = scaler.transform(X_test.astype(np.float32))

# -------------------------
# PCA (keep 95% variance)
# -------------------------
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------------
# Train SVM (fast & reliable for binary)
# -------------------------
svm = SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] FP16-compatible SVM Accuracy: {acc:.4f}")

# -------------------------
# Save FP16-compatible joblib
# -------------------------
joblib.dump({"model": svm, "scaler": scaler, "pca": pca}, "best_crime_detector2.joblib")
