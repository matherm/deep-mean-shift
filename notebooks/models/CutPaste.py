import numpy as np
import torch
import PIL

def random_int(max_h, max_w):
    x =  np.random.uniform(0, max_h, 1).astype(np.int32)[0] 
    y = np.random.uniform(0, max_w, 1).astype(np.int32)[0]
    return x, y

class RandomCutPaste():
    
    def __init__(self, idx=None):
        self.idx = idx
        self.n_classes = 2

    def __call__(self, x):
        if type(x) == int:
            # This works because ``transforms``` is called before ``target_transform`` in ``torch.Datasets``
            return self.label
        
        self.label =  np.random.randint(2) if self.idx is None else self.idx
        #print(self.label)
        #if True:
        if self.label == 1:
            maxi = int(0.80 * x.size[0])
            #Pw = int(np.random.uniform(0.02, 0.15, 1) * x.size[0])
            #Pw = int(np.random.uniform(0.20, 0.30, 1) * x.size[0])
            Pw = int(np.random.uniform(0.20, 0.80, 1) * x.size[0])
            Ph = np.clip(int(np.random.uniform(0.3, 3.3, 1) * Pw), 10, maxi)
            
            x = np.asarray(x)
            
            x_, y = random_int(x.shape[0] - Ph, x.shape[0] - Pw)
            Cut = x[x_:x_+Ph, y:y+Pw, :] 

            x_, y = random_int(x.shape[0] - Ph, x.shape[0] - Pw)
            x = x.copy()
            x[x_:x_+Ph, y:y+Pw, :] = Cut
                        
            x = PIL.Image.fromarray(np.uint8(x))
            
        return x