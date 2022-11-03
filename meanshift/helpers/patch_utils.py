from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict
import torch
import time
import numpy as np
from torch.nn import functional as F
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from .im2col import *

def asreshape(*args, shape=(3, 32, 32)):
    return tuple([arg.reshape(len(arg), *shape) for arg in args])       

def asfloat(*args):
    return tuple([arg.astype(np.float32) for arg in args])       

def patches_to_fmap(patches, T):
    n, n_components = patches.shape
    return tiles_to_fmap(patches.reshape(-1, T, n_components))

def fmap_to_features(X, P=2):
    features = i2col(X, X.shape, BSZ=(P, P)) # N x F
    T, F = features.shape[0] // len(X), features.shape[1]
    features = features.reshape(-1, T, F)   
    features = tiles_to_fmap(features)
    return features
    
def fmap_spatial_average(X, P=2, rho=2, avg_stride=1, avg_padd=0):
    F = fmap_to_features(X, P)
    F_map = avg_pool( F, size=rho, stride=avg_stride, padding=avg_padd)
    return F_map

def tiles_to_fmap(S_fmap):
    """
    in (n, n_tiles, n_components)
    
    returns (n, n_components, h, w)
    """
    n_tiles = S_fmap.shape[1]
    hw = int(np.sqrt(n_tiles))
    n = len(S_fmap)
    c = S_fmap.shape[2]
    
    # reshape to feature map
    if torch.is_tensor(S_fmap):
        return S_fmap.reshape(n, n_tiles, c).transpose(1, 2).contiguous().reshape(n, c, hw, hw) 
    else:
        return S_fmap.reshape(n, n_tiles, c).transpose(0, 2, 1).copy().reshape(n, c, hw, hw) 


def fmap_to_tiles(F_fmap):
    """
    in (n, n_components, h, w)
    
    returns (n, n_tiles, n_components)
    """
    n, n_components, h, w = F_fmap.shape
    n_tiles = h*w
     
    # reshape to feature map
    if torch.is_tensor(F_fmap):
        return F_fmap.reshape(n, n_components, h*w).transpose(1, 2).contiguous().view(n, n_tiles, n_components) 
    else:
        return F_fmap.reshape(n, n_components, h*w).transpose(0, 2, 1).copy().reshape(n, n_tiles, n_components) 


def flatten(X_patches):
    return X_patches.reshape(len(X_patches), -1)

def total_pool(X_patches):
    return X_patches.mean((2,3))

def col2i(F, shape=-1, BSZ=(4, 4), padding=0, stride=1):
    """
    F (array), shape (B, F, N)
    """
    if not torch.is_tensor(F):
        return col2i(torch.from_numpy(F), shape, BSZ, padding, stride).numpy()
    
    out = torch.nn.functional.fold(F, shape, kernel_size=BSZ[0],  dilation=1, padding=padding, stride=stride)
    
    return out


def i2col(X, shape=-1, BSZ=(4, 4), padding=0, stride=1):
    """
    >>> i2col(torch.cat([torch.zeros((1, 160, 8, 8)), torch.ones((1, 160, 8, 8))]), (160, 8, 8)).shape
    (50, 2560)
    
    :returns: patches
    """
    if not torch.is_tensor(X):
        return i2col(torch.from_numpy(X), shape, BSZ, padding, stride).numpy()
    # n_images = len(X)
    # imcol = im2col(X.reshape(len(X), *shape), BSZ=BSZ, padding=padding, stride=stride).T
    # imcol = imcol[im2colOrder(n_images, len(imcol))]
    
    imcol = torch.nn.functional.unfold(X, BSZ[0], dilation=1, padding=padding, stride=stride).transpose(2,1).contiguous() # N x T x F
    imcol = imcol.reshape(-1, imcol.shape[2])
    
    return imcol #.T.reshape(-1, shape[0], BSZ[0],  BSZ[0])

def i2c(X, p, stride=1, reorder = True):
    X_p = im2col(X, BSZ=(p, p), padding=0, stride=stride).T
    if reorder:
        X_p = X_p[im2colOrder(len(X), len(X_p))] 
    return X_p

def cluster_centers_tiles(X, tile_size = 7, stride = 7):
    max_h = X.shape[2] # B, C, H, W
    means = []
    
    i = 0
    while (i)*stride + tile_size <=  max_h:    
        j = 0
        while (j)*stride + tile_size <=  max_h:
            X_ = X[:, :, i*stride:(i)*stride + tile_size, j*stride:(j)*stride + tile_size]
            mean = X_.reshape(len(X_), -1).mean(0)
            means.append(mean)
            j += 1
        i += 1            
        
    return np.stack(means)

def batch_resize(X, size=(28, 28), bs=500):
    n, c, h, w = X.shape
    H, W = size[0], size[1]
    X_out = []
    for i in range(0, len(X), bs):
        batch = X[i:i+bs]
        batch = batch.reshape(len(batch), c, h, w)
        batch = resize(batch, (len(batch), c, H, W))
        X_out.append(batch)
    return np.concatenate(X_out, 0)

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def concat_features2(X, eff, blocks=[4, 6], bs=50, fmap_pool=False, debug=False, layer_map=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    if not torch.is_tensor(X):
        with torch.no_grad():
            y = concat_features2(torch.from_numpy(X), eff, blocks=blocks, bs=bs, fmap_pool=fmap_pool, debug=debug, layer_map=layer_map).detach().cpu().numpy()
        return y
    
    device = next(eff.parameters()).device
    
    m = torch.nn.AvgPool2d(fmap_pool, 1, 1) if fmap_pool else lambda x : x
    
    data = []
    hooks = []
    
    def hook(module, input, output):
        data.append(output)

    def hookf(module, input, output):
        data[-1] = embedding_concat(data[-1], output)
    
    hook = eff[:layer_map[blocks[0] - 1]][-1].register_forward_hook(hook)
    hooks.append(hook)
    
    for l in blocks[1:]:
        hook = eff[:layer_map[l - 1]][-1].register_forward_hook(hookf)
        hooks.append(hook)
    
    net = eff[:layer_map[blocks[-1] - 1]]
    
    for i in range(0, len(X), bs):
        m( net(X[i:i+bs].to(device) )).cpu()
    
    for handle in hooks: 
        handle.remove()
    
    return torch.cat(data)

def concat_features(X, eff, blocks=[4, 6], bs=50, fmap_pool=False, debug=False, layer_map=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    if not torch.is_tensor(X):
        with torch.no_grad():
            y = concat_features(torch.from_numpy(X), eff, blocks=blocks, bs=bs, fmap_pool=fmap_pool, debug=debug, layer_map=layer_map).detach().cpu().numpy()
        return y
    
    device = next(eff.parameters()).device
    
    m = torch.nn.AvgPool2d(3, 1, 1) if fmap_pool else lambda x : x
    
    net = eff[:layer_map[blocks[0] - 1]]
    X_ = torch.cat([ m( net(X[i:i+bs].to(device) )).cpu() for i in range(0, len(X), bs)])
    
    
    n, c, h, w = X_.shape
    
    if debug:
        print(X_.shape) 
    
    for l in blocks[1:]:
        net = eff[:layer_map[l - 1]]
        
        X__ = torch.cat([ m( net(X[i:i+bs].to(device) )).cpu() for i in range(0, len(X), bs)])
        
        if debug:
            print(X__.shape)
    
        #X__ = batch_resize(X__, size=(h, w))
        #X_ = np.concatenate([X_, X__], axis=1)
        X_ = embedding_concat(X_, X__)

    return X_


def crop_features(X, BSZ):
    n, fmap_shape = len(X), (X.shape[1],  X.shape[2],  X.shape[3])
        
    X_ = i2col(X, fmap_shape, BSZ=BSZ)
    T = int(len(X_) / n)
    
    X_ = X_.reshape(n, T, -1)
    X_ = tiles_to_fmap(X_)
    return X_


def patches_to_feature_space(X, net, P, s, blocks=[4, 6], layer_map=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bs=50, debug=False, reshape=-1):
    """
    Crops T patches from the given images, computes the feature maps, flattens the feature map and organaizes the patches.
    
    Returns: (B, D, sqrt(T), sqrt(T))
    """
    N, C = len(X), X.shape[1] 
    
    X_patches = i2col(X, X.shape[1:], BSZ=(P, P), padding=0, stride=s).reshape(-1, C, P, P)    
    
    if type(reshape) == tuple:
        X_patches = batch_resize(X_patches, size=reshape)
        
    F_patches = concat_features(X_patches, net, blocks=blocks, bs=bs, layer_map=layer_map) # (N*T, F, H, W)
    
    if debug:
        print("Fmap:", F_patches.shape)
    
    T = len(X_patches) // N
    S = int(np.sqrt(T))
    F, H, W = F_patches.shape[1:]
    D = F*H*W        
    
    F = F_patches.reshape(N, S, S, D)
    F = F.transpose(0,3,1,2).copy() # (N, D, S, S)
    
    return F


def patches_to_feature_space_vit(X, net, P, s, blocks=[7], layer_map=[1,2,3,4,5,6,7], bs=200, debug=False, size=(224, 224)):
    """
    Image Vision Transformer
    
    Crops T patches from the given images, computes the feature maps, flattens the feature map and organaizes the patches.
    
    Returns: (B, D, sqrt(T), sqrt(T))
    """
    device = next(net.parameters()).device
    
    N, C = len(X), X.shape[1]     
    X = i2col(X, X.shape[1:], BSZ=(P, P), padding=0, stride=s).reshape(-1, C, P, P) 
    
    for block in blocks:
        net.encoder.layers[layer_map[block - 1]].register_forward_hook(lambda m,i,o : features.append(o.cpu().detach()))
                           
    features = [] 
    with torch.no_grad():
         for i in range(0, len(X), bs):
            out = net(torch.from_numpy( batch_resize(X[i:i+bs], size=size)  ).to(device) ).detach().cpu()
    
    for block in blocks:
        net.encoder.layers[layer_map[block - 1]]._forward_hooks = OrderedDict()
    
    concated_features = []
    cls_token = True
    for i in range(0, len(features), len(blocks)):
        if cls_token:
            fmap = torch.cat([features[j][:,:1,:] for j in range(i, i+len(blocks)) ], axis=2)
        else:
            fmap = torch.cat([features[j][:,1:,:] for j in range(i, i+len(blocks)) ], axis=2)
        concated_features.append(fmap)

    concated_batches = torch.cat(concated_features, axis = 0) # B, T, F
    
    B, T, F = concated_batches.shape
    sqrT = int(np.sqrt(T)) # 1
    concated_batches = concated_batches.view(B, F, sqrT, sqrT).contiguous() # B,  F, H, W
    
    T = len(X) // N
    S = int(np.sqrt(T))
    F = concated_batches.shape[1]
    concated_batches = concated_batches.reshape(N, S, S, F)
    concated_batches = concated_batches.numpy().transpose(0,3,1,2).copy() # (N, F, S, S)
                           
    return concated_batches            
    
def fmap_to_patches(X):
    if torch.is_tensor(X) == False:
        return fmap_to_patches( torch.from_numpy(X) ).numpy()
    
    B, F, H, W = X.shape
    X = X.transpose(1,2) # B, H, F, W
    X = X.transpose(2,3).contiguous() # B, H, W, F
    return X.view(B*H*W, F)


def avg_pool(fmap, size=2, stride=1, padding=-1):
    if not torch.is_tensor(fmap):
        return avg_pool(torch.from_numpy(fmap), size, stride, padding).numpy()
    
    size = np.min([size, fmap.shape[2]])
    padding = size//2 if padding < 0 else padding
    return F.avg_pool2d(fmap, size, stride=stride, padding=padding)          