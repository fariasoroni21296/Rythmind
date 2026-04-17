from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def evaluate_all(X, y_true, y_pred):
    sil = silhouette_score(X, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)

    return sil, nmi, ari, purity


def purity_score(y_true, y_pred):
    contingency_matrix = np.zeros((len(set(y_pred)), len(set(y_true))))

    for i in range(len(y_true)):
        contingency_matrix[y_pred[i]][y_true[i]] += 1

    return np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)