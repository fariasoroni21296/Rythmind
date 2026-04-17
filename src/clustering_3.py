from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def run_kmeans(X, k=10):
    model = KMeans(n_clusters=k)
    return model.fit_predict(X)

def run_dbscan(X):
    model = DBSCAN(eps=0.5)
    return model.fit_predict(X)

def run_agglomerative(X, k=10):
    model = AgglomerativeClustering(n_clusters=k)
    return model.fit_predict(X)