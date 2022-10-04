from sklearn.covariance import ledoit_wolf
from sklearn.covariance import empirical_covariance, shrunk_covariance
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import numpy as np

def ledoit_custom_alpha(X, shrinkage=0.1):
    # COV1, s1 = ledoit_wolf(X, assume_centered=False, block_size=1000)
    emp_cov = empirical_covariance(X, assume_centered=False)
    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=shrinkage)
    return shrunk_cov
    
def cov_prob_pca(X):
    n, d = X.shape
    k = np.min([n, d])
    
    pca = PCA().fit(X)
    W = pca.components_.T
        
    total_var = X.var(0, ddof=1).sum()
    explain_var = pca.explained_variance_
    
    print(d-k, (total_var - explain_var.sum()), pca.explained_variance_) 
        
    if d - k > 0:
        sigma =  1 / (d-k) * (total_var - explain_var.sum()) 
    else:
        sigma = 0.
    
    COV = (W @ np.diag(pca.explained_variance_ - sigma) @ W.T) + np.eye(d) * sigma
    
    return COV, sigma


if __name__ == "__main__":
    X = load_breast_cancer().data
    X = X[:20, :30]

    COV1, s1 = ledoit_wolf(X, assume_centered=False, block_size=1000)
    COV2 = np.cov(X.T)
    COV3, s2 = cov_prob_pca(X)

    np.linalg.eigvals(COV1), np.linalg.eigvals(COV2),np.linalg.eigvals(COV3), s1, s2