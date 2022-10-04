import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

class Empty(nn.Module):
    
    def forward(self, X):
        return X

class Mine(nn.Module):
    def __init__(self, fmaps=16, hidden_size=100, dimA=1, dimB=1, lr=1e-3, init_scale=0.02, fdiv=False):
        super().__init__()
        self.fmaps = fmaps
        self.hidden_size = hidden_size
        self.flattened_dim = None
        if type(dimA) == int:
            self.fc1 = nn.Linear(dimA+dimB, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight,std=init_scale)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight,std=init_scale)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight,std=init_scale)
            nn.init.constant_(self.fc3.bias, 0)
            nn.init.normal_(self.fc4.weight,std=init_scale)
            nn.init.constant_(self.fc4.bias, 0)
        else:
            self.dimA = dimA
            self.dimB = dimB
            ca, ha, wa = dimA
            cb, hb, wb = dimB
            
            ksize = 4
            self.fc1a = nn.Sequential(nn.BatchNorm2d(ca), 
                                      nn.Conv2d(ca , fmaps, ksize, padding=1), 
                                      #nn.AvgPool2d(2), 
                                      nn.ELU(), 
                                      nn.Conv2d(fmaps, fmaps, ksize, padding=1), 
                                      nn.ELU(), 
                                      nn.BatchNorm2d(fmaps), 
                                      nn.Conv2d(fmaps, fmaps, ksize, padding=1), 
                                      nn.ELU(), 
                                     # nn.Conv2d(fmaps, fmaps, ksize, padding=1), 
                                     # nn.ELU()
                                     )
            
            self.fc1b = nn.Sequential(#nn.BatchNorm2d(cb), 
                                      #nn.Conv2d(cb, fmaps, ksize, padding=1), 
                                      #nn.AvgPool2d(2), 
                                      #nn.ELU(), 
                                      #nn.Conv2d(fmaps, fmaps, ksize, padding=1), 
                                      nn.ELU()
                                     )
            
            self.outputA = self.fc1a(torch.ones((1, ca, ha, wa)))
            self.outputB = self.fc1b(torch.ones((1, cb, hb, wb)))
            self.flattened_dim = self.outputA.view(1, -1).shape[1] + self.outputB.view(1, -1).shape[1]
            self.fc3 = nn.Linear(self.flattened_dim, hidden_size)
            self.fc4 = nn.Linear(hidden_size, 1)
            
        self.mine_net_optim = optim.Adam(self.parameters(), lr=lr)
        self.ma_et = 1.
        self.ma_rate=0.01
        self.dimA = dimA
        self.dimB = dimB
        self.fdiv = fdiv
        
    def __repr__(self):
        return f"Mine(fmaps={self.fmaps}, hidden_size={self.hidden_size}, flattened_dim={self.flattened_dim}, latent_spatialA={self.outputA.shape}, latent_spatialB={self.outputB.shape})[{np.sum([p.numel() for p in self.parameters() if p.requires_grad])}]"
        
    def _forward(self, input):
        A, B = input
        
        A = A.to(next(self.parameters()).device)
        B = B.to(next(self.parameters()).device)
        
        outputA = self.fc1a(A).view(len(A), -1)
        outputB = self.fc1b(B).view(len(B), -1)
        
        output = torch.cat([outputA, outputB], axis=1)
        
        output = F.elu(self.fc3(output))
        output = self.fc4(output)
        return output.cpu()
    
    def mutual_information(self, joint, marginal):
        mi_lb , j, et, m = self(joint, marginal)
        self.ma_et = ((1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)).detach()
        
        # unbiasing use moving average
        # loss = -(torch.mean(t) - (1/self.ma_et)*torch.mean(et))
        # use biased estimator
        # loss = - mi_lb
        
        # replacing by Binary CE optimization
        acc = ((j > 0).sum() + (m < 0).sum())/(len(j)+len(m))
        loss = torch.nn.BCEWithLogitsLoss()(torch.cat([j, m]).flatten(), torch.cat([torch.ones(len(j)), torch.zeros(len(m))]))
        return loss, mi_lb, acc, torch.cat([j, m]).view(-1, 1).detach()
        
    def forward(self, joint, marginal):
        j = self._forward(joint)
        m = self._forward(marginal)
        if self.fdiv:
            et = torch.exp(m - 1)
        else:
            et = torch.exp(m)
        mi_lb = torch.mean(j) - torch.log(torch.mean(et))
        return mi_lb, j, et, m
    
    @staticmethod
    def sample_batch(A, B, batch_size=100, sample_mode='joint', device="cuda"):
            if sample_mode == 'joint':
                index = np.random.choice(range(A.shape[0]), size=batch_size, replace=False)
                batch = (A[index], B[index])
            else:
                joint_index = np.random.choice(range(A.shape[0]), size=batch_size, replace=False)
                marginal_index = np.random.choice(range(A.shape[0]), size=batch_size, replace=False)
                batch = (A[joint_index], B[marginal_index])
                
            return ( torch.FloatTensor(batch[0]).to(device),torch.FloatTensor(batch[1]).to(device))
    
    def fit(self, dataA, dataB, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3), validation_ratio=10):
        
        # data is x or y
        result = list()
        device = next(self.parameters()).device

        data_idx = np.random.permutation(range(len(dataA)))
        val_idx = data_idx[:len(data_idx)//validation_ratio]
        dat_idx = data_idx[len(data_idx)//validation_ratio:]
        
        dataA_val = dataA[val_idx]
        dataA = dataA[dat_idx]
        
        dataB_val = dataB[val_idx]
        dataB = dataB[dat_idx]
        
        for i in range(iter_num):
            batch_joint = Mine.sample_batch(dataA, dataB, batch_size=batch_size,  device=device)
            batch_marginal = Mine.sample_batch(dataA, dataB, batch_size=batch_size, sample_mode='marginal', device=device)
            
            loss, mi_lb, acc, _ = self.mutual_information(batch_joint, batch_marginal)
            loss.backward()
            self.mine_net_optim.step()
            self.mine_net_optim.zero_grad()
            
            trn_loss = mi_lb.detach().cpu().numpy()
            
            with torch.no_grad():
                batch_joint = Mine.sample_batch(dataA_val, dataB_val, batch_size=len(val_idx), device=device)
                batch_marginal = Mine.sample_batch(dataA_val, dataB_val, batch_size=len(val_idx),sample_mode='marginal', device=device)
                
                loss, mi_lb, acc, _ = self.mutual_information(batch_joint, batch_marginal)
                val_loss = mi_lb.detach().cpu().numpy()
            
            result.append((trn_loss, val_loss, acc))
            
            if (i+1)%(log_freq)==0:
                print(np.asarray(result[-20:]).mean(0))
        return np.asarray(result)
