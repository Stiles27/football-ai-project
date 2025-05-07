import os
import cv2
import moviepy.editor as mp

# === CONFIGURATION ===
VIDEO_PATH = "data/psg_liverpool_ucl2025.mp4"
FRAMES_DIR = "frames/"
AUDIO_DIR = "audio/"
AUDIO_NAME = "psg_liverpool_audio.wav"

# === CRÉATION DES DOSSIERS ===
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# === 1. EXTRACTION DES FRAMES ===
print("[INFO] Extraction des frames...")
video = cv2.VideoCapture(VIDEO_PATH)
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_count = 0
saved = 0

success, frame = video.read()
while success:
    if frame_count % fps == 0:  # 1 frame par seconde
        filename = os.path.join(FRAMES_DIR, f"frame_{saved:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1
    success, frame = video.read()
    frame_count += 1

video.release()
print(f"[INFO] {saved} frames enregistrées dans {FRAMES_DIR}")

# === 2. EXTRACTION DE L'AUDIO ===
print("[INFO] Extraction de l'audio...")
clip = mp.VideoFileClip(VIDEO_PATH)
clip.audio.write_audiofile(os.path.join(AUDIO_DIR, AUDIO_NAME))
print(f"[INFO] Audio enregistré dans {AUDIO_DIR}{AUDIO_NAME}")
