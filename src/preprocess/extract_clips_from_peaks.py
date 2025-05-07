import json
import os
from moviepy.editor import VideoFileClip

# === CONFIGURATION ===
VIDEO_PATH = "data/psg_liverpool_ucl2025.mp4"
PEAKS_JSON = "data/audio_peaks.json"
OUTPUT_DIR = "clips/"
CLIP_DURATION = 4  # en secondes (2 avant + 2 après)

# === CRÉATION DU DOSSIER CLIPS ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CHARGEMENT DES PICS AUDIO ===
with open(PEAKS_JSON, "r") as f:
    peaks = json.load(f)

print(f"[INFO] {len(peaks)} pics à découper")

# === CHARGEMENT DE LA VIDÉO ===
video = VideoFileClip(VIDEO_PATH)
video_duration = video.duration

# === POUR CHAQUE PIC, EXTRAIRE UN CLIP COURT ===
for i, timestamp in enumerate(peaks):
    start = max(0, timestamp - CLIP_DURATION / 2)
    end = min(video_duration, timestamp + CLIP_DURATION / 2)

    clip = video.subclip(start, end)
    clip_filename = os.path.join(OUTPUT_DIR, f"clip_{i:04d}.mp4")
    clip.write_videofile(clip_filename, codec="libx264", audio_codec="aac", verbose=False, logger=None)

print(f"[INFO] {len(peaks)} clips extraits dans {OUTPUT_DIR}")
