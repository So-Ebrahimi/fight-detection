import torch
import cv2
from model import FightModel
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = FightModel().to(device)
model.load_state_dict(torch.load("fight_model.pth", map_location=device))
model.eval()

cap = cv2.VideoCapture("test.mp4")
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (224, 224)) / 255.0
    frames.append(frame_resized)

    if len(frames) == 16:
        x = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        out = model(x)
        cls = torch.argmax(out, dim=1).item()

        label = "FIGHT" if cls == 0 else "NO FIGHT"
        cv2.putText(frame, label, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 3)

        frames.pop(0)

    cv2.imshow("Fight Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
