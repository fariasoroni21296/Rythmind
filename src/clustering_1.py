from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels

def apply_pca(data, n_components=8):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)