import cv2
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class FightDataset(Dataset):
    def __init__(self, root, frames=16):
        self.samples = []
        self.frames = frames

        for label, folder in enumerate(["fight", "noFight"]):
            path = os.path.join(root, folder)
            if os.path.exists(path):
                for vid in os.listdir(path):
                    if vid.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        self.samples.append((os.path.join(path, vid), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []
        while len(frames) < self.frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        while len(frames) < self.frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))

        frames = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)
        label = torch.tensor(label, dtype=torch.long)
        return frames, label
