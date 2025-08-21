'''MC3 needs 32 consecutive frames of size (3, 112, 112) (channels first) to make a prediction.

So now we’ll build:

A rolling buffer that always holds the latest 32 person crops

It will feed this to MC3 when full'''

import numpy as np
from collections import deque

class FrameBuffer:
    def __init__(self, max_len=32):
        self.buffer = deque(maxlen=max_len)
        self.max_len = max_len

    def add_frame(self, frame):
        # Frame: (112, 112, 3)
        # Convert to (3, 112, 112)
        if frame.shape != (112, 112, 3):
            raise ValueError(f"[ERROR!] Frame shape should be (112,112,3) but got {frame.shape}")
        frame = np.transpose(frame, (2, 0, 1)) / 255.0  # Normalize to [0,1]
        self.buffer.append(frame)

    def is_full(self):
        return len(self.buffer) == self.max_len

    def get_clip(self):
        if self.is_full():
            clip = np.stack(self.buffer, axis=1)  # Shape: (3, 32, 112, 112)
            return clip.astype(np.float32)
        else:
            return None
