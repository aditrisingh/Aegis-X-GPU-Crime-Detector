# detector/mc3_infer.py
import onnxruntime as ort
import numpy as np

class MC3Runner:
    def __init__(self):
        providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession("models/mc3_features.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        print("[INFO] MC3 model loaded successfully.")
        print("[INFO] Using Providers:", self.session.get_providers())
        print("[INFO] Model expects input shape:", self.session.get_inputs()[0].shape)

    def extract_features(self, clip):
        # clip shape: (3, 32, 112, 112)
        clip = clip.astype(np.float32)
        clip = np.expand_dims(clip, axis=0)  # (1, 3, 32, 112, 112)
        outputs = self.session.run(None, {self.input_name: clip})
        return outputs[0]

    def extract_features_batch(self, clips):
        # clips shape: (batch, 3, 32, 112, 112)
        clips = clips.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: clips})
        return outputs[0]  # (batch, feature_dim)
