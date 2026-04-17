from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd

def evaluate(data, labels):
    sil = silhouette_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    return sil, ch

def save_results(path, results):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)