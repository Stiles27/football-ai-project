import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

CLIPS_DIR = "clips/"
SPEC_DIR = "spectrograms/"
os.makedirs(SPEC_DIR, exist_ok=True)

for filename in os.listdir(CLIPS_DIR):
    if not filename.endswith(".mp4"):
        continue

    filepath = os.path.join(CLIPS_DIR, filename)
    basename = os.path.splitext(filename)[0]
    audio_tmp_path = f"audio_npy/{basename}.wav"

    # 1. Extraire l'audio du clip
    clip = VideoFileClip(filepath)
    clip.audio.write_audiofile(audio_tmp_path, verbose=False, logger=None)

    # 2. Charger l’audio et générer le spectrogramme
    y, sr = librosa.load(audio_tmp_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # 3. Sauver le spectrogramme en image PNG
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SPEC_DIR, f"{basename}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"[INFO] Spectrogrammes générés dans {SPEC_DIR}")
