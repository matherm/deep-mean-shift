from sklearn.cluster._kmeans import _kmeans_plusplus
from sklearn.cluster import KMeans
from .kcenter_greedy import *

def kmeans_plusplus(X, K):
    centers, selected_idx = _kmeans_plusplus(X, K, np.linalg.norm(X, axis=1)**2, np.random.RandomState(seed=0), n_local_trials=None)
    return centers, selected_idx

def kcenter_greedy(X, K, model=None):
    selector = kCenterGreedy(X, 0, 0)
    selected_idx = selector.select_batch(model=model, already_selected=[], N=K)
    return X[selected_idx], selected_idx

def kmeans(X, K):
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=3, random_state=0).fit(X)
    return kmeans.cluster_centers_

