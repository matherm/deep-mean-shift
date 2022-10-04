import torch
from .Hotelling import *
from .helpers import *
from .helpers.architectures import PseudoLayer


class DeepMeanShift():
    
    def __init__(self, features="eff", blocks=[5,6], receptive_field=16, multimod=False, K=2, device="cuda"):
        
        net, layer_map = get_fmap(features) if type(features) == str else [torch.nn.Sequential(PseudoLayer()), [1,1,1,1,1,1,1]]
        net = net.to(device)
        self.net = net.eval()
        
        self.device = device
        self.features = features
        self.layer_map = layer_map
        self.blocks = blocks
        self.receptive_field = receptive_field
        self.multimod = multimod
        self.K = K
        
        self.stats = None
        self.BSZ = None
        self.rho = None
        self.F = None
        self.P = None
        self.s = None
        self.A = None
        
    def __repr__(self):
        return f"DeepMeanShift(A={self.A}, P={self.P}, s={self.s}, F={self.F}, rho={self.rho}, T={self.stats[3]}, fmap={self.stats[5]}, means={self.stats[1].shape}, mu={self.stats[0].shape}, cov={self.stats[2].shape})"
       
                 
    def fit(self, X, A=32, P=30, s=2, verbose=False):
        
        rho = int( (A - P)/s + 1 )
        F = P//self.receptive_field
        
        if self.features == "vitb16":
            X = patches_to_feature_space_vit(X, net, P=P, s=s, blocks=self.blocks, layer_map=self.layer_map)
        else:  
            X = concat_features2(X, self.net, blocks=self.blocks, fmap_pool=False, layer_map=self.layer_map)
        
        mu, means, cov_inv, T = estimate_statistics(X, mode="full", BSZ=(F, F), rho=rho, avg_padd=0)
        
        if self.multimod:
            means = estimate_local_multimodal_statistics_opt(X, mu, rho, BSZ=(F, F), clustermode="kcenter", K=self.K, avg_padd=0)
        
        if verbose:
            print(f"A={A}, P={P}, s={s}, F={F}, rho={rho}, T={T}, fmap={X.shape}, means={means.shape}, mu={mu.shape}, cov={cov_inv.shape}")
        
        mu, means, cov_inv = torch.from_numpy(mu), torch.from_numpy(means), torch.from_numpy(cov_inv).float()
        L = torch.linalg.cholesky(cov_inv).float()
        
        
        self.stats = (mu.to(self.device), 
                      means.to(self.device), 
                      cov_inv.to(self.device), 
                      T, 
                      L.to(self.device),
                      X.shape
                      )
        self.rho = rho
        self.F = F
        self.P = P
        self.s = s
        self.A = A
        return self
    
    def transform(self, X):
        return self.score(X, reduce="feature")
       
    def anomaly_map(self, X_in, X_out, size=(224, 224), mini=-1, maxi=-1, mode="hot"):
        
        s_in = self.score(X_in, reduce="none").detach().cpu() if len(X_in) > 0 else torch.tensor([])
        s_out = self.score(X_out, reduce="none").detach().cpu() if len(X_out) > 0 else torch.tensor([])
        
        anomap_in, anomap_out = make_anomaly_map(s_in, s_out, p=1, size=size, mini=mini, maxi=maxi, mode=mode)
        return anomap_in, anomap_out
    
            
    def score(self, X_, reduce="max", bs=20, requires_grad=False):
        if self.stats is None:
            raise ValueError("First call fit()")
            
        if not torch.is_tensor(X_):
            X_ = torch.from_numpy(X_).float()
            
        mu, means, cov_inv, T, L, fmap = self.stats
        D = mu.shape[0]

        def feat_(X):
            if self.features == "vitb16":
                X = patches_to_feature_space_vit(X, net, P=self.P, s=self.s, blocks=self.blocks, layer_map=self.layer_map)
            else:  
                X = concat_features2(X, self.net, blocks=self.blocks, fmap_pool=False, layer_map=self.layer_map, bs=bs)
            return X

        if requires_grad:     
            def feat(X):
                return feat_(X)            
        else:
            def feat(X):
                with torch.no_grad():
                    return feat_(X)

        def shift_(X, mean):
            return local_hotelling_extreme(X.to(self.device), T, mu, mean, L, BSZ=(self.F, self.F), rho=self.rho, reduce="none")

        if requires_grad:     
            def mu_shift(X, mean): 
                return shift_(X, mean)
        else:
            def mu_shift(X, mean): 
                with torch.no_grad():
                    return shift_(X, mean)
            
        all_scores = []
        for i in range(0, len(X_), bs):
            
            X = feat(X_[i:i+bs])
            
            if self.multimod:
                scores = mu_shift(X, means[0])
                for k in range(1, self.K):
                    scores_ = mu_shift(X, means[k])
                    scores = torch.stack([scores, scores_]).min(0)[0]
                    scores = scores if requires_grad else scores.detach()
            else:
                scores = mu_shift(X, means)
                scores = scores if requires_grad else scores.detach() 
                
            if reduce == "max":
                scores = scores.max(1)[0].max(1)[0]
            
            all_scores.append(scores)
        
        return torch.cat(all_scores) / D

    def evaluate(self, X_in, X_out):
        if len(X_out) == 0:
            return -1
        
        s_in = self.score(X_in).detach().cpu()
        s_out = self.score(X_out).detach().cpu()
        
        auc_local = roc_auc_score([0] * len(X_in) + [1] * len(X_out), np.concatenate([s_in, s_out]))
        return auc_local
        