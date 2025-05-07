import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from multimodal_model import MultimodalCNN

# === CONFIGURATION ===
FRAMES_DIR = "frames/"
SPECTRO_DIR = "spectrograms/"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === TRANSFORMATIONS IMAGES ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === DATASET MULTIMODAL ===
class FootballDataset(Dataset):
    def __init__(self, frames_dir, spec_dir, transform=None):
        self.frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        self.transform = transform
        self.frames_dir = frames_dir
        self.spec_dir = spec_dir

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.frames_dir, self.frames[idx])
        spec_path = os.path.join(self.spec_dir, self.frames[idx].replace(".jpg", ".png"))

        video_img = Image.open(frame_path).convert("RGB")
        audio_img = Image.open(spec_path).convert("RGB")

        if self.transform:
            video_img = self.transform(video_img)
            audio_img = self.transform(audio_img)

        # ❗ POUR L'EXEMPLE : on met 1 pour les frames paires (action) et 0 pour les impaires
        label = 1 if idx % 2 == 0 else 0

        return video_img, audio_img, torch.tensor(label)

# === ENTRAÎNEMENT ===
def train():
    dataset = FootballDataset(FRAMES_DIR, SPECTRO_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultimodalCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for video, audio, labels in loader:
            video, audio, labels = video.to(DEVICE), audio.to(DEVICE), labels.to(DEVICE)

            outputs = model(video, audio)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        acc = correct / len(dataset)
        print(f"[EPOCH {epoch+1}] Loss: {running_loss:.4f} - Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "models/multimodal_model.pt")
    print("[INFO] Modèle entraîné et sauvegardé dans models/")

if __name__ == "__main__":
    train()
