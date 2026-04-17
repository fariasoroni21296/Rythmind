import os
import numpy as np
import librosa

# ---------- AUDIO FEATURE ----------

def extract_mfcc(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Error loading:", audio_path)
        return None


def load_audio_features(audio_dir):
    features = []

    for genre in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre)

        if not os.path.isdir(genre_path):
            continue

        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)

            # accept multiple formats
            if file.endswith((".wav", ".au", ".mp3")):
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)

    return np.array(features)




def get_audio_features_only(audio_dir):
    return load_audio_features(audio_dir)

def load_audio_features(audio_dir):
    features = []

    print("Reading audio folder:", audio_dir)

    for genre in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre)

        print("Genre folder:", genre)

        if not os.path.isdir(genre_path):
            continue

        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)

            if file.endswith((".wav", ".au", ".mp3")):
                print("Processing:", file_path)

                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)

    print("Total features:", len(features))
    return np.array(features)