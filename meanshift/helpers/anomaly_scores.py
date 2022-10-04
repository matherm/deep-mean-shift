from sklearn.neighbors import NearestNeighbors
import torch
import time
import numpy as np
from torch.nn import functional as F
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from .kcenter_greedy import *
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_pred):
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.show()

def auc_local_mean_shift(model, X, X_in, X_out, size=2):
    S_fmap = model.transform(np.asarray(X), agg="none")
    means = compute_local_means(S_fmap, size=size)
    
    t0 = time.time()
    S_in_fmap = model.transform(np.asarray(X_in), agg="none")
    S_out_fmap = model.transform(np.asarray(X_out), agg="none")
    
    fmap_in_shift = avg_pool( tiles_to_fmap(S_in_fmap), size=size ) - means
    fmap_out_shift = avg_pool( tiles_to_fmap(S_out_fmap), size=size ) - means
    
    score_in1 = np.linalg.norm(fmap_in_shift.reshape(len(fmap_in_shift),fmap_in_shift.shape[1],-1), axis=1).max(1)
    score_out1 = np.linalg.norm(fmap_out_shift.reshape(len(fmap_out_shift),fmap_out_shift.shape[1],-1), axis=1).max(1)
    t1 = (time.time() - t0)
    
    return roc_auc_score([0] * len(score_in1) + [1] * len(score_out1), np.concatenate([score_in1, score_out1])), t1

def auc_cluster_mean_shift(model, X, X_in, X_out, use_coreset=True, return_coreset=False, k=None):
    
    S_fmap = model.transform(np.asarray(X), agg="none")
    
    n_tiles = S_fmap.shape[1]
    n_components = S_fmap.shape[-1]
    
    X_ = S_fmap.reshape(-1, n_components)
    
    if k is None: 
        selector = kCenterGreedy(X_, 0, 0)
        selected_idx = selector.select_batch(model=None, already_selected=[], N=int(X_.shape[0]*0.01))
        train_coreset = X_[selected_idx]
        k = len(train_coreset)
 
    n_clusters = k
    print("n_clusters", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=1).fit(X_)

    if use_coreset:
        kmeans.cluster_centers_ = train_coreset
    
    # Patch Clustering
    t0 = time.time()
    S_in_fmap = model.transform(np.asarray(X_in), agg="none")
    S_out_fmap = model.transform(np.asarray(X_out), agg="none")
    
    X_valid_ = S_in_fmap.reshape(-1, n_components)
    X_test_ = S_out_fmap.reshape(-1, n_components)
        
    S_in_assignments = kmeans.predict(X_valid_)
    S_out_assignments = kmeans.predict(X_test_)

    # Compute mean-shift per cluster center
    S_in_means = np.zeros((len(S_in_assignments), n_components))
    S_out_means = np.zeros((len(S_out_assignments), n_components))

    for i in range(n_clusters):
        S_in_means[S_in_assignments == i] += X_valid_[S_in_assignments == i] - kmeans.cluster_centers_[i]
        S_out_means[S_out_assignments == i] += X_test_[S_out_assignments == i] - kmeans.cluster_centers_[i]

    # compute cluster distance for every patch, take maximum!
    scores_valid = np.linalg.norm(S_in_means.reshape(len(S_in_fmap), n_tiles, n_components), axis=2).max(1)
    scores_test = np.linalg.norm(S_out_means.reshape(len(S_out_fmap), n_tiles, n_components), axis=2).max(1)
    t1 = (time.time() - t0)

    if return_coreset:
        return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])) , t1, train_coreset
    
    return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])) , t1      


def auc_sources_mean_shift(model, X, X_in, X_out, k=100):
    
    S = model.transform(np.asarray(X), agg="mean")
    
    n_components = S.shape[-1]
    
    n_clusters = np.min([k, len(S)])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(S)
    
    t0 = time.time()
    S_in = model.transform(np.asarray(X_in), agg="mean")
    S_out = model.transform(np.asarray(X_out), agg="mean")
    
    S_in_assignments = kmeans.predict(S_in)
    S_out_assignments = kmeans.predict(S_out)

    S_in_means = np.zeros_like(S_in)
    S_out_means = np.zeros_like(S_out)
    
    for i in range(n_clusters):

        S_in_means[S_in_assignments == i] += S_in[S_in_assignments == i] - kmeans.cluster_centers_[i]
        S_out_means[S_out_assignments == i] += S_out[S_out_assignments == i] - kmeans.cluster_centers_[i]
        
    scores_valid = np.linalg.norm(S_in_means, axis=1)
    scores_test = np.linalg.norm(S_out_means, axis=1)
    t1 = (time.time() - t0)

    return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])), t1


def auc_coreset(X, X_in, X_out):
    X = avg_pool(X, 2)
    
    n_components = X.shape[1]
    n_tiles =  X.shape[2] * X.shape[3]
    
    X = X.reshape(len(X), n_components, n_tiles) 
    X = X.transpose(0, 2, 1).copy().reshape(len(X) * n_tiles, n_components)
    
    selector = kCenterGreedy(X, 0, 0)
    selected_idx = selector.select_batch(model=None, already_selected=[], N=int(X.shape[0]*0.01))
    train_coreset = X[selected_idx]
    
    t0 = time.time()
    X_in = avg_pool(X_in, size=2)
    X_out = avg_pool(X_out, size=2)
    
    X_in = X_in.reshape(len(X_in), n_components, n_tiles) 
    X_out = X_out.reshape(len(X_out), n_components, n_tiles) 
    
    X_in = X_in.transpose(0, 2, 1).copy().reshape(len(X_in) * n_tiles, n_components)
    X_out = X_out.transpose(0, 2, 1).copy().reshape(len(X_out) * n_tiles, n_components)
        
    scores_valid = patch_core_score_2(train_coreset, X_in, n_tiles, b=10, reweight=False) 
    scores_test = patch_core_score_2(train_coreset, X_out, n_tiles, b=10, reweight=False)
    t1 = (time.time() - t0)

    return roc_auc_score([0] * len(scores_valid) + [1] * len(scores_test), np.concatenate([scores_valid, scores_test])), t1      


def auc_global_mean_shift(model, X_in, X_out):
    t0 = time.time()
    S_in = model.transform(np.asarray(X_in), agg="mean")
    S_out = model.transform(np.asarray(X_out), agg="mean")
    
    score_in1 = np.linalg.norm( S_in, axis=1)
    score_out1 = np.linalg.norm( S_out, axis=1)
    t1 = (time.time() - t0)
    
    return roc_auc_score([0] * len(score_in1) + [1] * len(score_out1), np.concatenate([score_in1, score_out1])), t1

def local_mean_shift(model, X, X_in, X_out, size=2):
    S_fmap = model.transform(np.asarray(X), agg="none")
    means = compute_local_means(S_fmap, size=size)
    
    t0 = time.time()
    S_in_fmap = model.transform(np.asarray(X_in), agg="none")
    S_out_fmap = model.transform(np.asarray(X_out), agg="none")
    
    fmap_in_shift = avg_pool( tiles_to_fmap(S_in_fmap), size=size ) - means
    fmap_out_shift = avg_pool( tiles_to_fmap(S_out_fmap), size=size ) - means
    
    score_in = np.linalg.norm(fmap_in_shift.reshape(len(fmap_in_shift), fmap_in_shift.shape[1], -1), axis=1).reshape(len(fmap_in_shift), fmap_in_shift.shape[2], fmap_in_shift.shape[3])
    score_out = np.linalg.norm(fmap_out_shift.reshape(len(fmap_out_shift),fmap_out_shift.shape[1],-1), axis=1).reshape(len(fmap_out_shift), fmap_out_shift.shape[2], fmap_out_shift.shape[3])
    
    return score_in, score_out

def patch_core_score_2(X_patches, X_test_patches, T, b=9, reweight=True):    
    # We need a knn structure
    neigh = NearestNeighbors(n_neighbors=b).fit(X_patches)
    
    # Compute the number of datapoints
    n = len(X_test_patches) // T
    
    # Find the nearest patchs
    dists, m = neigh.kneighbors(X_test_patches, return_distance=True) # n_test x k
    
    scores = []
    
    dists = dists.reshape(n, T, b )
    
    for i in range(n):
        score_patches = dists[i]
        N_b = score_patches[ np.argmax(score_patches[:, 0]) ] # the patch with the maximum distance
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b)))) if reweight else 1.
        score = w*np.max(score_patches[:,0]) # Image-level score
        scores.append(score)
        
    return scores

def patch_core_score(X_patches, X_test_patches, T, b=9, reweight=True):    
    # We need a knn structure
    neigh = NearestNeighbors(n_neighbors=1).fit(X_patches)
    
    # Compute the number of datapoints
    n = len(X_test_patches) // T
    
    # Find the nearest patchs
    dists, m = neigh.kneighbors(X_test_patches, return_distance=True) # n_test x 1
    
    # Get the maximum patch of a single image
    idx = dists.reshape((n, T)).argmax(1)
    s_ = dists.reshape((n, T))[range(n), idx]
    s_idx = np.arange(len(X_test_patches)).reshape(n, T)[range(n), idx]
    
    # Find the nearest neighbor of the maximum patch
    m_dists, m_idx = neigh.kneighbors(X_test_patches[s_idx], n_neighbors=1) 
    
    # Find the distances of the nearest neighbors to its b-nearest neighrbos
    mdists, mm = neigh.kneighbors(X_patches[m_idx.flatten()], n_neighbors=b, return_distance=True) # n_test x b
    
    # re-weight
    mdists = np.clip(mdists, -1e70, 200)
    s_ = np.clip(s_, -1e70, 200) 
    w = np.exp(mdists).sum(1) if reweight else 1.
    
    s = (1 - (np.exp(s_)/w)) * s_
    return s