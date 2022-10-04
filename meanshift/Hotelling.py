from sklearn.covariance import ledoit_wolf
from .helpers import *

def local_hotelling_opt(X, T, mu, means, cov_inv, BSZ=(1, 1), rho=1):
    P = BSZ[0]
    n, F, H, W = X.shape

    # Unfold and center
    F = torch.nn.functional.unfold(X, P, dilation=1, padding=0, stride=1).reshape(n, P*P*F, int(np.sqrt(T)), int(np.sqrt(T)))
    F = F - mu[None, :, None, None]
    
    F_pool_mu = torch.nn.functional.avg_pool2d(F, rho, stride=1, padding=0) - means
    n, D, s = F_pool_mu.shape[:3]
    
    # Mahalanobis
    X_ = F_pool_mu.transpose(1, 2).transpose(2, 3).contiguous().view(n*s*s, D)
    mahalanobis = (X_ * (cov_inv @ X_.T).T).sum(1)
    mahalanobis = mahalanobis.view(n, s*s)
    T2 = mahalanobis.max(1)[0]
    
    return T2


def local_hotelling_extreme(X, T, mu, means, L, BSZ=(1, 1), rho=1, reduce="max"):
    """
        L = torch.linalg.cholesky(cov_inv).float()
        s = np.linalg.norm(X @ U.T, axis=1) 
        s = np.linalg.norm(X @ model.model.sphering_matrix, axis=1)
    
    """
    P = BSZ[0]
    n, F, H, W = X.shape
    
    kernel = L.T.view(F*P*P, F, P, P)
    F = torch.nn.functional.conv2d(X, kernel, stride=1) 
    
    D, s = means.shape[1], means.shape[2]
    F_means = ((means.view(D, s*s).T @ L).T).view(1, D, s, s)
    F_mu = (mu.unsqueeze(0) @ L).view(1, D, 1, 1)
        
    F_pool_mu = torch.nn.functional.avg_pool2d(F, rho, stride=1, padding=0) - F_mu - F_means
    
    if reduce == "feature":
        return F_pool_mu
    
    n, D, s = F_pool_mu.shape[:3]
    
    # Mahalanobis
    X_ = F_pool_mu.transpose(1, 2).transpose(2, 3).contiguous().view(n*s*s, D)
    mahalanobis = torch.pow(X_, 2).sum(1)
    mahalanobis = mahalanobis.view(n, s*s)
    
    if reduce == "max":
        T2 = mahalanobis.max(1)[0]
    elif reduce == "none":
        T2 = mahalanobis.reshape(n, s, s)
    
    return T2


def local_hotelling_multimod(X, T, mu, means, cov_inv, BSZ=(4, 4), rho=2, reduce="min_max", avg_stride=1, avg_padd=-1):
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    m, h, w = means.shape[0], means.shape[3], means.shape[4]
    
    # to patches 
    X = i2col(X, fmap_shape, BSZ=BSZ)
    c = X.shape[1]
    
    # centering
    X = X - mu                       
    
    # To tiles
    X = X.reshape(-1, T, X.shape[1])          
    
    # spatial average
    spat_avg = avg_pool( tiles_to_fmap(X), size=rho, stride=avg_stride, padding=avg_padd)
    
    modes = []
    for mm in range(m):
        X_fmap = spat_avg - means[mm] # (means: m, n, c, h, w)
        X_fmap = X_fmap.reshape(n, c, h, w)

        if reduce == "fmap":
            return X_fmap

        # fmap_to_tiles
        X = fmap_to_tiles(X_fmap).reshape(-1, c) # (n*h*w,  c)

        # Mahalanobis
        S = (X * (cov_inv @ X.T).T).sum(1) # (n*h*w,)
        
        modes.append(S)
    
    S = torch.stack(modes)

    # min per mode
    if reduce == "min_max" or reduce == "max":
        s = S.reshape(m, n, h*w).min(0)[0].max(1)[0]
    elif reduce == "max_max":
        s = S.reshape(m, n, h*w).max(0)[0].max(1)[0]
    else:
        s = S.reshape(m, n, h, w).min(0)[0]
        
    return s

def local_hotelling(X, T, mu, means, cov_inv, BSZ=(4, 4), rho=2, reduce="max", avg_stride=1, avg_padd=-1):
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    h, w = means.shape[2], means.shape[3]

    # to patches 
    X = i2col(X, fmap_shape, BSZ=BSZ)
    c = X.shape[1]
    
    # centering
    X = X - mu                       
    
    # To tiles
    X = X.reshape(-1, T, X.shape[1])          
    
    X_fmap = avg_pool( tiles_to_fmap(X), size=rho, stride=avg_stride, padding=avg_padd) - means # (n, c, h, w)

    if reduce == "fmap":
        return X_fmap

    # fmap_to_tiles
    X = fmap_to_tiles(X_fmap).reshape(-1, c)

    # Mahalanobis
    S = (X * (cov_inv @ X.T).T).sum(1)

    if reduce == "max":
        # take maximum
        s = S.reshape(n, -1).max(1)
    else:
        s = S.reshape(n, h, w)

    return s

def estimate_statistics(X, mode="ledoit", BSZ=(4, 4), rho=2, avg_stride=1, avg_padd=-1, bs=50):
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    
    X_ = np.concatenate([i2col(X[i:i+bs], fmap_shape, BSZ=BSZ) for i in range(0, len(X), bs)])
    T = int(len(X_) / n)
    mu = X_.mean(0)
    
    X_ = X_ - mu
    X_ = X_.reshape(n, T, -1)
    means = np.concatenate([avg_pool( tiles_to_fmap(X_[i:i+bs]), size=rho, stride=avg_stride, padding=avg_padd).mean(0, keepdims=True) for i in range(0, len(X_), bs)])
    means = means.mean(0, keepdims=True)
        
    X_est = X_.reshape(-1, len(mu))
    
    if mode == "lr":
        model = SFA(shape=fmap_shape, 
                    BSZ=BSZ, 
                    stride=1, 
                    n_components="q9999",
                    remove_components=0,
                    max_components=4000,
                    min_components=1,
                    mode="ta")
        model.fit(X, 1, bs=2000, lr=1e-3, logging=-10)
        cov = model.model.cov
        
    elif mode == "full":
        cov = np.cov(X_est.T)
        
    elif mode == "ledoit":
        cov, shrinkage = ledoit_wolf(X_est, assume_centered=False, block_size=1000)
        
    elif mode == "ledoit01":
        cov = ledoit_custom_alpha(X_est, shrinkage=0.1)
    
    elif mode == "ledoit025":
        cov = ledoit_custom_alpha(X_est, shrinkage=0.1)
    
    elif mode == "ledoit05":
        cov = ledoit_custom_alpha(X_est, shrinkage=0.1)
    
    return mu, means, np.linalg.inv(cov), T


def estimate_global_statistics(X, mode="ledoit", BSZ=(4, 4)):
    if mode != "ledoit":
        raise NotImplemented()
    
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
        
    X_ = i2col(X, fmap_shape, BSZ=BSZ)
    T = int(len(X_) / len(X))
    
    cov = ledoit_wolf(X_, assume_centered=False, block_size=1000)[0]

    mu = X_.mean(0)
    
    return mu, np.linalg.inv(cov) , T


def estimate_local_multimodal_statistics_opt(X, mu, rho, BSZ, clustermode="kcenter", avg_stride=1, avg_padd=-1, K=64, share_loc=True):
    """
    X=(2262, 272, 14, 14), means=(1, 1088, 11, 11), mu=(1088,), F=2
    """
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    
    # to patches 
    X = i2col(X, fmap_shape, BSZ=BSZ)
    C = X.shape[1]
    T = len(X) // n
    
    # centering
    X = X - mu                       
    
    # To tiles
    X = X.reshape(-1, T, X.shape[1])          
    
    # spatial average
    spat_avg = avg_pool( tiles_to_fmap(X), size=rho, stride=avg_stride, padding=avg_padd)
    m = spat_avg.shape[2]

    if share_loc:
        patch_part = fmap_to_tiles(spat_avg) #(N, T, C)
        patch_part = patch_part.reshape(-1, C)
        
        if clustermode == "kcenter":
            centers, selected_idx = kcenter_greedy(patch_part, K=K)
        elif clustermode == "kmeans":
            centers = kmeans(patch_part, K=K)
        elif clustermode == "kmeans++":
            centers, selected_idx = kmeans_plusplus(patch_part, K=K) # (K, C)
        
        centers = np.tile(centers[:, :, None, None], (1, 1, m, m))
        coreset_means = centers.reshape(K, 1, C, m, m) 
            
    else:
        coreset_means = []
        for i in range(m):
            for j in range(m):

                patch_part = spat_avg[:, :, i, j]

                if clustermode == "kcenter":
                    centers, selected_idx = kcenter_greedy(patch_part, K=K)
                elif clustermode == "kmeans":
                    centers = kmeans(patch_part, K=K)
                elif clustermode == "kmeans++":
                    centers, selected_idx = kmeans_plusplus(patch_part, K=K)
                #centers = centers.mean(0, keepdims=True)

                coreset_means.append(centers)

        coreset_means = np.asarray(coreset_means) # m^2 x K, C
        coreset_means = coreset_means.transpose(1, 2, 0).copy().reshape(K, 1, C, m, m) 
    
    return coreset_means #  K, 1, C, m, m

def estimate_local_multimodal_statistics(X, mu, r, stride=1, offset=1, clustermode="kmeans"):
    B, C, H, W = X.shape
    
        
    X_ = i2col(X, (C, H, W), BSZ=BSZ)

    coreset_means = []

    for h in range(0, H-offset, stride):
        for w in range(0, W-offset, stride):
            patch_part = X[:, :, h:h+r, w:w+r] - mu[None, :, None, None]# B x C x h x w
            patch_part = fmap_to_tiles(patch_part).copy() # B, T, C
            patch_part = patch_part.mean(1) # patch_part = patch_part.reshape(-1, C)
            
            if clustermode == "kcenter":
                centers, selected_idx = kcenter_greedy(patch_part, K=64)
            elif clustermode == "kmeans":
                centers = kmeans(patch_part, K=64)
            elif clustermode == "kmeans++":
                centers, selected_idx = kmeans_plusplus(patch_part, K=64)
            #centers = centers.mean(0, keepdims=True)
            coreset_means.append(centers)

    coreset_means = np.asarray(coreset_means) # M x K, C

    M, K = coreset_means.shape[0:2]
    coreset_means = coreset_means.transpose(1, 2, 0).copy().reshape(K, 1, C, int(np.sqrt(M)), int(np.sqrt(M))) 
    
    return coreset_means #  K, 1, C, sqrt(M), sqrt(M)


def hotelling(X, T, mu, inv_cov, BSZ=(4, 4)):
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    X = i2col(X, fmap_shape,BSZ=BSZ)
    
    X = X - mu # centering
    X = X.reshape(-1, T, X.shape[1]).mean(1) # average patches
    s = (X * (inv_cov @ X.T).T).sum(1)
    
    return s    