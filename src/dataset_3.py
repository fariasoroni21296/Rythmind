import os
import numpy as np
import librosa
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_audio_features(audio_dir):
    features = []
    labels = []

    for genre in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre)

        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)

            y, sr = librosa.load(file_path, duration=30)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            features.append(np.mean(mfcc.T, axis=0))
            labels.append(genre)

    return np.array(features), np.array(labels)


def extract_lyrics_features(lyrics_list):
    vectorizer = TfidfVectorizer(max_features=500)
    return vectorizer.fit_transform(lyrics_list).toarray()


def combine_features(audio_feat, lyrics_feat):
    return np.concatenate([audio_feat, lyrics_feat], axis=1)