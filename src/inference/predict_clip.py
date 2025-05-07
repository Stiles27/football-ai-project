import os
import torch
from torchvision import transforms
from PIL import Image
from moviepy.editor import VideoFileClip
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from ..train.multimodal_model import MultimodalCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_frame(clip_path, frame_path):
    clip = VideoFileClip(clip_path)
    frame = clip.get_frame(t=clip.duration / 2)  # Prendre le milieu du clip
    img = Image.fromarray(frame)
    img.save(frame_path)
    return frame_path

def generate_spec(clip_path, spec_path):
    y, sr = librosa.load(clip_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return spec_path

def predict(clip_path, model_path="models/multimodal_model.pt"):
    temp_frame = "temp_frame.jpg"
    temp_audio = "temp_audio.wav"
    temp_spec = "temp_spec.png"

    # Extraire frame centrale
    extract_frame(clip_path, temp_frame)

    # Extraire audio du clip
    clip = VideoFileClip(clip_path)
    clip.audio.write_audiofile(temp_audio, verbose=False, logger=None)

    # Générer spectrogramme
    generate_spec(temp_audio, temp_spec)

    # Charger les images
    video_img = transform(Image.open(temp_frame).convert("RGB")).unsqueeze(0).to(DEVICE)
    audio_img = transform(Image.open(temp_spec).convert("RGB")).unsqueeze(0).to(DEVICE)

    # Charger modèle
    model = MultimodalCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Prédire
    with torch.no_grad():
        output = model(video_img, audio_img)
        pred = torch.argmax(output, dim=1).item()

    label = "⚽ ACTION" if pred == 1 else "😴 NO ACTION"
    print(f"[PREDICTION] Le clip contient : {label}")

    # Nettoyer fichiers temporaires
    os.remove(temp_frame)
    os.remove(temp_audio)
    os.remove(temp_spec)

if __name__ == "__main__":
    test_clip = "clips/clip_0003.mp4"  # Remplace par le clip que tu veux tester
    predict(test_clip)
