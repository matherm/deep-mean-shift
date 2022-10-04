import PIL
import numpy as np

def ShuffleAnomaly(P=2, A=10):
    """
    P (int) : size of the anomaly
    A (int) : area of the anomaly where it can appear
    """
    
    def apply(x):
        x_ = np.asarray(x).copy()
        h, w, c = x_.shape
        x = int( np.random.uniform(0, A-P, 1) )
        y = int( np.random.uniform(0, A-P, 1) )
        
        patch = x_[x:x+P, y:y+P].copy()
        np.random.shuffle(patch.reshape(-1, c))
        
        x_[x:x+P, y:y+P] = patch
        
        return PIL.Image.fromarray(x_)
        
    return apply

def transform_shuffle_anomaly(X, P=50, A=100):
    pil = transforms.ToPILImage()
    ten =  transforms.ToTensor()
    shuffle = ShuffleAnomaly(P=P, A=A)
    
    xs = []
    for x in X:
        x = torch.from_numpy(x)
        x = pil(x)
        x = shuffle(x)
        x = ten(x)
        x = x.numpy()
        xs.append(x)
    return np.stack(xs)