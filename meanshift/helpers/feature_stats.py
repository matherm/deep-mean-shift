from .patch_utils import *

def intra_class_variance(X, T, mu, means, cov, BSZ, rho):
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    
    # to patches 
    X = i2col(X, fmap_shape, BSZ=BSZ)
    c = X.shape[1]
    
    # centering
    X = X - mu                       
    
    # To tiles
    X = X.reshape(-1, T, X.shape[1])          
    
    # local means
    patches = tiles_to_fmap(X)
    means   = batch_resize(means, (patches.shape[2], patches.shape[3])) 
    X_fmap  = patches - means # (n, c, h, w)
    
    # fmap_to_tiles
    X = fmap_to_tiles(X_fmap).reshape(-1, c)
    
    return X.var(0).sum()
    

def inter_class_variance(means):
    B, F, H, W = means.shape
    
    means = means.reshape(B*F, -1)
    return means.var(1).sum()


def kurtosis_negentropy(X, T, mu, means, cov, BSZ, n_components=100):
    from ..tools.SFA import SFA
    
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
    
    # to patches 
    X = i2col(X, fmap_shape, BSZ=BSZ)
    c = X.shape[1]
    
    # centering
    X = X - mu                       
    
    # To tiles
    X = X.reshape(-1, T, X.shape[1])          
    
    # local means
    patches = tiles_to_fmap(X)
    means   = batch_resize(means, (patches.shape[2], patches.shape[3])) 
    X_fmap  = patches - means # (n, c, h, w)
    
    C, H, W = X_fmap.shape[1:]
    model = SFA(shape=(C,H,W), 
                   BSZ=(1, 1), 
                   stride=1, 
                   n_components=n_components,
                   mode="ta")
    model.fit(X, 30, bs=10000, logging=-1)
    kurt = model.kurt
    negent = model.negH_sum    
    return kurt, negent