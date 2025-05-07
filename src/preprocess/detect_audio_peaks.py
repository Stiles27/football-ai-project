import librosa
import numpy as np
import os
import json

AUDIO_PATH = "audio/psg_liverpool_audio.wav"
OUTPUT_JSON = "data/audio_peaks.json"

# 1. Charger l'audio
y, sr = librosa.load(AUDIO_PATH)

# 2. Calcul de l'énergie sur des fenêtres
hop_length = 512
energy = np.array([
    sum(abs(y[i:i+hop_length]**2))
    for i in range(0, len(y), hop_length)
])

# 3. Normaliser et détecter les pics (seuil adaptatif)
threshold = np.percentile(energy, 90)
peaks = np.where(energy >= threshold)[0]
timestamps = [round(p * hop_length / sr, 2) for p in peaks]

# 4. Enregistrer les pics dans un fichier JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(timestamps, f)

print(f"[INFO] {len(timestamps)} pics détectés. Enregistrés dans {OUTPUT_JSON}")
