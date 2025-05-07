
# 🧠⚽ Football AI – Détection Multimodale d’Actions

Ce projet est un système **d’intelligence artificielle multimodale** capable de détecter automatiquement des **actions clés** (buts, tirs, fautes, etc.) dans des résumés de matchs de football à partir de **vidéo et d’audio**.

---

## 📁 Structure du projet

```
football_ai/
├── audio/             # Audios extraits
├── audio_npy/         # Audio converti en wav/temp
├── clips/             # Mini-clips vidéo autour des pics audio
├── data/              # Fichiers bruts (vidéos mp4)
├── frames/            # Frame centrale de chaque clip
├── models/            # Modèle entraîné (.pt)
├── spectrograms/      # Spectrogrammes audio
└── src/
    ├── inference/
    │   └── predict_clip.py
    ├── preprocess/
    │   ├── detect_audio_peaks.py
    │   ├── extract_clips_from_peaks.py
    │   ├── extract_video_audio.py
    │   └── generate_spectrogram.py
    └── train/
        ├── multimodal_model.py
        └── train.py
```

---

## ⚙️ Installation

Assurez-vous d’avoir **Python 3.8+** installé, puis :
```bash
pip install -r requirements.txt
```

### Exemple de dépendances :
- `torch`, `torchvision`
- `moviepy`, `librosa`
- `matplotlib`, `scikit-learn`, `opencv-python`

---

## 🚀 Utilisation du pipeline

### 1. Prétraitements

```bash
python src/preprocess/detect_audio_peaks.py
python src/preprocess/extract_clips_from_peaks.py
python src/preprocess/extract_video_audio.py
python src/preprocess/generate_spectrogram.py
```

### 2. Entraînement

```bash
python src/train/train.py
```

### 3. Prédiction

```bash
python src/inference/predict_clip.py
```

---

## 👥 Auteurs

- [Ton Nom]
- [Collaborateurs]

---

## 📄 Licence

Projet à but éducatif – tous droits réservés.
