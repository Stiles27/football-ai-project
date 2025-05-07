import os
import glob

# === Récupération des vidéos ===
video_files = glob.glob("data/*.mp4")
if not video_files:
    raise FileNotFoundError("❌ Aucune vidéo .mp4 trouvée dans le dossier 'data/'")

print("🎥 Vidéos disponibles :")
for i, f in enumerate(video_files):
    print(f"{i+1}. {os.path.basename(f)}")

choice = input("👉 Entrez le numéro de la vidéo à utiliser : ")
try:
    index = int(choice) - 1
    video_path = video_files[index]
except (ValueError, IndexError):
    raise ValueError("⛔ Choix invalide. Veuillez relancer et entrer un numéro correct.")

print(f"✅ Vidéo sélectionnée : {video_path}")

# === Pipeline ===
print("\n🔊 Étape 1 - Extraction audio depuis la vidéo")
os.system("python src/preprocess/extract_video_audio.py")

print("🎬 Étape 2 - Détection des pics audio")
os.system("python src/preprocess/detect_audio_peaks.py")

print("✂️ Étape 3 - Extraction des clips autour des pics")
os.system("python src/preprocess/extract_clips_from_peaks.py")

print("🎼 Étape 4 - Génération des spectrogrammes audio")
os.system("python src/preprocess/generate_spectrogram.py")

print("🧠 Étape 5 - Entraînement du modèle IA")
os.system("python src/train/train.py")

print("✅ Étape 6 - Évaluation automatique")
os.system("python evaluate.py")
