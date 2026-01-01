import torch
import cv2
import os
from model import FightModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = FightModel().to(device)
model.load_state_dict(torch.load("fight_model.pth", map_location=device))
model.eval()

def extract_frames(video_path, num_frames=16):
    """Extract frames uniformly from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        cap.release()
        return np.zeros((num_frames, 224, 224, 3))
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        # If video has fewer frames, sample all
        frame_indices = list(range(total_frames))
    else:
        # Sample uniformly across video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Extract frames at specified indices
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (224, 224)) / 255.0
            frames.append(frame_resized)
        else:
            # If frame read fails, use last successful frame or zeros
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3)))
    
    cap.release()
    
    # Ensure we have exactly num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))
    
    frames = frames[:num_frames]  # Trim if too many
    
    return np.array(frames)

def predict_video(video_path, model, device):
    """Predict class for a single video"""
    frames = extract_frames(video_path)
    
    # Convert to tensor: [frames, channels, height, width] -> [batch, frames, channels, height, width]
    x = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(out, dim=1).item()
        confidence = prob[0][pred].item()
    
    return pred, confidence

# Test all videos
dataset_path = "dataset/test"
fight_folder = os.path.join(dataset_path, "fight")
noFight_folder = os.path.join(dataset_path, "noFight")

true_labels = []
predictions = []
video_paths = []
results = []

print("\n" + "="*70)
print("Testing all videos in TEST dataset...")
print("="*70)

# Test fight videos (label = 0)
print(f"\nTesting fight videos...")
fight_videos = sorted([f for f in os.listdir(fight_folder) if f.endswith('.mp4')])
for i, vid_name in enumerate(fight_videos, 1):
    video_path = os.path.join(fight_folder, vid_name)
    true_label = 0  # fight = 0
    
    try:
        pred, confidence = predict_video(video_path, model, device)
        true_labels.append(true_label)
        predictions.append(pred)
        video_paths.append(video_path)
        
        result = {
            'video': vid_name,
            'true_label': 'FIGHT',
            'predicted_label': 'FIGHT' if pred == 0 else 'NO FIGHT',
            'correct': pred == true_label,
            'confidence': confidence
        }
        results.append(result)
        
        status = "✓" if pred == true_label else "✗"
        print(f"[{i}/{len(fight_videos)}] {status} {vid_name}: {result['predicted_label']} (confidence: {confidence:.4f})")
    except Exception as e:
        print(f"Error processing {vid_name}: {e}")

# Test noFight videos (label = 1)
print(f"\nTesting noFight videos...")
noFight_videos = sorted([f for f in os.listdir(noFight_folder) if f.endswith('.mp4')])
for i, vid_name in enumerate(noFight_videos, 1):
    video_path = os.path.join(noFight_folder, vid_name)
    true_label = 1  # noFight = 1
    
    try:
        pred, confidence = predict_video(video_path, model, device)
        true_labels.append(true_label)
        predictions.append(pred)
        video_paths.append(video_path)
        
        result = {
            'video': vid_name,
            'true_label': 'NO FIGHT',
            'predicted_label': 'FIGHT' if pred == 0 else 'NO FIGHT',
            'correct': pred == true_label,
            'confidence': confidence
        }
        results.append(result)
        
        status = "✓" if pred == true_label else "✗"
        print(f"[{i}/{len(noFight_videos)}] {status} {vid_name}: {result['predicted_label']} (confidence: {confidence:.4f})")
    except Exception as e:
        print(f"Error processing {vid_name}: {e}")

# Calculate metrics
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
cm = confusion_matrix(true_labels, predictions)

print(f"\nTotal Videos Tested: {len(true_labels)}")
print(f"  - Fight videos: {len(fight_videos)}")
print(f"  - NoFight videos: {len(noFight_videos)}")

print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"              FIGHT  NO FIGHT")
print(f"Actual FIGHT    {cm[0][0]:4d}      {cm[0][1]:4d}")
print(f"      NO FIGHT  {cm[1][0]:4d}      {cm[1][1]:4d}")

# Per-class accuracy
fight_correct = sum(1 for r in results if r['true_label'] == 'FIGHT' and r['correct'])
noFight_correct = sum(1 for r in results if r['true_label'] == 'NO FIGHT' and r['correct'])

fight_accuracy = fight_correct / len(fight_videos) if fight_videos else 0
noFight_accuracy = noFight_correct / len(noFight_videos) if noFight_videos else 0

print(f"\nPer-Class Accuracy:")
print(f"  Fight:    {fight_accuracy:.4f} ({fight_accuracy*100:.2f}%)")
print(f"  NoFight:  {noFight_accuracy:.4f} ({noFight_accuracy*100:.2f}%)")

# Show incorrect predictions
incorrect = [r for r in results if not r['correct']]
if incorrect:
    print(f"\nIncorrect Predictions ({len(incorrect)} videos):")
    for r in incorrect[:20]:  # Show first 20
        print(f"  {r['video']}: True={r['true_label']}, Predicted={r['predicted_label']} (conf: {r['confidence']:.4f})")
    if len(incorrect) > 20:
        print(f"  ... and {len(incorrect) - 20} more")

print("\n" + "="*70)
