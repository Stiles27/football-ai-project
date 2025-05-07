import os
import torch
from torchvision import transforms
from PIL import Image
from src.train.multimodal_model import MultimodalCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CONFIG ===
FRAMES_DIR = "frames/"
SPECTROGRAMS_DIR = "spectrograms/"
MODEL_PATH = "models/multimodal_model.pt"

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === LOAD MODEL ===
model = MultimodalCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === EVALUATION ===
total = 0
correct = 0

for fname in sorted(os.listdir(FRAMES_DIR)):
    if not fname.endswith(".jpg"):
        continue

    frame_path = os.path.join(FRAMES_DIR, fname)
    spec_path = os.path.join(SPECTROGRAMS_DIR, fname.replace(".jpg", ".png"))

    if not os.path.exists(spec_path):
        continue

    video_img = transform(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    audio_img = transform(Image.open(spec_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(video_img, audio_img)
        pred = torch.argmax(output, dim=1).item()

    # Faux label pour le test : pair = 1 (action), impair = 0 (non action)
    label = 1 if total % 2 == 0 else 0

    if pred == label:
        correct += 1
    total += 1

acc = correct / total if total > 0 else 0
print(f"[EVALUATION] Accuracy = {acc:.4f} on {total} samples")
