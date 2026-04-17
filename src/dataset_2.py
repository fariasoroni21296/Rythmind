import os
import numpy as np
import librosa
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_mfcc(file_path, n_mfcc=40, max_len=130):
    y, sr = librosa.load(file_path, duration=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def load_audio_dataset(audio_dir, limit=50):  # 👈 add limit
    X = []
    labels = []
    genres = os.listdir(audio_dir)

    count = 0

    for label, genre in enumerate(genres):
        genre_path = os.path.join(audio_dir, genre)

        for file in os.listdir(genre_path):

            if file.lower().endswith((".wav", ".au", ".mp3")):
                path = os.path.join(genre_path, file)

                try:
                    mfcc = extract_mfcc(path)
                    X.append(mfcc)
                    labels.append(label)

                    count += 1
                    print(f"Loaded {count}")  # 👈 DEBUG PRINT

                    if count >= limit:
                        break

                except Exception as e:
                    print(f"Error loading {file}: {e}")

        if count >= limit:
            break

    X = np.array(X)
    X = X[:, np.newaxis, :, :]

    return X, np.array(labels)

def load_lyrics(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lyrics = f.readlines()

    # clean lines
    lyrics = [line.strip() for line in lyrics if line.strip() != ""]

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=300)
    lyrics_vec = vectorizer.fit_transform(lyrics).toarray()

    return lyrics_vec


def combine_features(audio_latent, lyrics_vec):
    min_len = min(len(audio_latent), len(lyrics_vec))

    audio_latent = audio_latent[:min_len]
    lyrics_vec = lyrics_vec[:min_len]

    combined = np.concatenate([audio_latent, lyrics_vec], axis=1)

    return combined