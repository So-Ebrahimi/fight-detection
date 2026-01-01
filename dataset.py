import cv2
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class FightDataset(Dataset):
    def __init__(self, root, frames=16, augment=False, img_size=224):
        self.samples = []
        self.frames = frames
        self.augment = augment
        self.img_size = img_size

        for label, folder in enumerate(["fight", "noFight"]):
            path = os.path.join(root, folder)
            if os.path.exists(path):
                for vid in os.listdir(path):
                    if vid.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        self.samples.append((os.path.join(path, vid), label))

    def __len__(self):
        return len(self.samples)

    def _get_video_frame_count(self, cap):
        """Get total number of frames in video"""
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _sample_frames_uniformly(self, cap, num_frames):
        """Sample frames uniformly across video duration"""
        total_frames = self._get_video_frame_count(cap)
        
        if total_frames <= num_frames:
            # If video has fewer frames than needed, sample all
            frame_indices = list(range(total_frames))
        else:
            # Sample uniformly across video
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        return frame_indices

    def _apply_augmentation(self, frame):
        """Apply data augmentation to a frame"""
        if not self.augment:
            return frame
        
        # Random horizontal flip
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        
        # Random brightness and contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-10, 10)    # Brightness
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            frame = np.clip(frame, 0, 255)
        
        # Random color jitter (slight)
        if random.random() > 0.5:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)  # Saturation
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.8, 1.2)  # Value
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Random crop and resize (simulate)
        if random.random() > 0.5:
            h, w = frame.shape[:2]
            crop_size = int(min(h, w) * random.uniform(0.85, 1.0))
            y = random.randint(0, max(0, h - crop_size))
            x = random.randint(0, max(0, w - crop_size))
            frame = frame[y:y+crop_size, x:x+crop_size]
            frame = cv2.resize(frame, (self.img_size, self.img_size))
        
        return frame

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # Fallback: return zeros if video can't be opened
            frames = np.zeros((self.frames, self.img_size, self.img_size, 3))
            cap.release()
        else:
            # Sample frames uniformly
            frame_indices = self._sample_frames_uniformly(cap, self.frames)
            frames = []
            
            for idx_frame in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
                ret, frame = cap.read()
                if not ret:
                    # If frame read fails, use last successful frame or zeros
                    if frames:
                        frame = frames[-1].copy()
                    else:
                        frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                else:
                    # Resize and normalize
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                    frame = self._apply_augmentation(frame)
                    frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
            
            cap.release()
        
        # Ensure we have exactly the right number of frames
        while len(frames) < self.frames:
            frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.float32))
        
        frames = frames[:self.frames]  # Trim if too many
        
        frames = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)
        label = torch.tensor(label, dtype=torch.long)
        return frames, label
