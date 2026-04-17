from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import pandas as pd

def evaluate_clustering(X, pred_labels, true_labels=None):
    results = {}

    if len(set(pred_labels)) > 1:
        results['silhouette'] = silhouette_score(X, pred_labels)
        results['davies_bouldin'] = davies_bouldin_score(X, pred_labels)
    else:
        results['silhouette'] = -1
        results['davies_bouldin'] = -1

    if true_labels is not None:
        results['ARI'] = adjusted_rand_score(true_labels, pred_labels)

    return results


def save_results(results_dict, filename="results/clustering_metrics.csv"):
    df = pd.DataFrame(results_dict).T
    df.to_csv(filename)