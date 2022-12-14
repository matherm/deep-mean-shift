import os
import os.path as osp
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_tiled_im(i, ds, n_tiles, imshape):
    big_im = None
    idx = np.concatenate([[i], np.random.uniform(0, len(ds), n_tiles**2 - 1)]).astype(np.int32)
    alloc = lambda item : np.zeros((item.shape[0], item.shape[1] * n_tiles, item.shape[2] * n_tiles), dtype=item.dtype) # allocate much room for all tiles
    for i in range(n_tiles):
        for j in range(n_tiles):
            item = ds[idx[i*n_tiles+j]].reshape(imshape)
            big_im =  alloc(item) if big_im is None else big_im
            big_im[:, i*item.shape[1]:(i+1)*item.shape[1] , j*item.shape[2]:(j+1)*item.shape[2]] = item
    return big_im.flatten()
        

class Cifar10_OneClass():

    '''
    A small CIFAR 10 wrapper to train one-class classifier a.k.a. novelty detection

    X, X_in, X_out = Cifar10_OneClass("./data", train_classes=[0], test_classes=[1, 2], transform=transforms.ToTensor())

    '''
    
    CLASSES = ["plane", "car", "bird", "cat", "deer" ,"dog", "frog", "horse", "ship", "truck"]

    def __init__(self,
                 root="./data",
                 train_classes=[0],
                 test_classes=None,
                 transform=transforms.ToTensor(),
                 download=True,
                 n_tiles=1,
                 imshape=(3, 32, 32),
                 balance=True,
                 z_normalize=False,
                 fold = -1):
        if test_classes is None:
            test_classes = [0,1,2,3,4,5,6,7,8,9]
        
            for t in train_classes:
                test_classes.remove(t)
            
        raw_train = CIFAR10(root=root, download=download, train=True, transform=transform)
        raw_test = CIFAR10(root=root, download=download, train=False, transform=transform)

        data_train = [raw_train[i][0] for i in range(len(raw_train)) if raw_train[i][1] in train_classes]
        data_train = torch.cat(data_train).reshape(len(data_train), -1).numpy()
        
        data_train_outlier = [raw_train[i][0] for i in range(len(raw_train)) if raw_train[i][1] in test_classes]
        data_train_outlier = torch.cat(data_train_outlier).reshape(len(data_train_outlier), -1).numpy()

        data_test_inliers = [raw_test[i][0] for i in range(len(raw_test)) if raw_test[i][1] in train_classes]
        data_test_inliers = torch.cat(data_test_inliers).reshape(len(data_test_inliers), -1).numpy()

        data_test_outliers = [raw_test[i][0] for i in range(len(raw_test)) if raw_test[i][1] in test_classes]
        data_test_outliers = torch.cat(data_test_outliers).reshape(len(data_test_outliers), -1).numpy()

        
        # make tiled cifar by copying images from same class
        if n_tiles > 1:
            data_train = [get_tiled_im(i, data_train, n_tiles, imshape) for i in range(len(data_train))]
            data_train = np.asarray(data_train).reshape(len(data_train), -1)

            data_test_outliers = [get_tiled_im(i, data_test_outliers, n_tiles, imshape) for i in range(len(data_test_outliers))]
            data_test_outliers = np.asarray(data_test_outliers).reshape(len(data_test_outliers), -1)

            data_test_inliers = [get_tiled_im(i, data_test_inliers, n_tiles, imshape) for i in range(len(data_test_inliers))]
            data_test_inliers = np.asarray(data_test_inliers).reshape(len(data_test_inliers), -1)

        # balance test data
        if balance:
            mini = min([data_test_inliers.shape[0], data_test_outliers.shape[0]])
            data_test_inliers = data_test_inliers[:mini]
            data_test_outliers = data_test_outliers[:mini]
            
        # members 
        self.z_normalize = z_normalize
        self.train_classes = train_classes
        self.test_classes = test_classes
        self.fold = fold

        # Provide Design Matrices as members
        if self.z_normalize:
            print("Standardizing data (with_mean=True, with_std=True)")
            scaler = StandardScaler(with_mean=True, with_std=True).fit(data_train)
            self.data_train = scaler.transform(data_train)
            self.data_test_inliers  = scaler.transform(data_test_inliers)
            self.data_test_outliers = scaler.transform(data_test_outliers)
        else:
            self.data_train = data_train
            self.data_train_outlier = data_train_outlier
            self.data_test_inliers  = data_test_inliers
            self.data_test_outliers = data_test_outliers            

    def __len__(self):
        return 1
    
    def get_fold(self, X, i, N):
        p = len(X) // N 
        i = i % p

        X_train = np.vstack([ X[:i*N], X[(i+1)*N:] ])
        X_val   = X[i*N:(i+1)*N]
            
        return X_train, X_val


    def __getitem__(self, index):
        if self.fold > -1:
            IN = np.vstack([self.data_train, self.data_test_inliers])
            OUT = np.vstack([self.data_train_outlier, self.data_test_outliers])
            
            X_train, X_inlier = self.get_fold(IN, self.fold, len(self.data_test_inliers))
            _, X_test = self.get_fold(OUT, self.fold, len(self.data_test_inliers))
            
            return X_train, X_inlier, X_test
        else:
            return self.data_train, self.data_test_inliers, self.data_test_outliers

    def __repr__(self): 
        return "{}(z_normalize={}, train_classes={}, test_classes={}, data_train={}, data_test_inliers={}, data_test_outliers={})".format(
            self.__class__.__name__,
            self.z_normalize,
            self.train_classes, self.test_classes, self.data_train.shape, self.data_test_inliers.shape, self.data_test_outliers.shape)

if __name__ == "__main__":
    data_train, data_test_inliers, data_test_outliers = Cifar10_OneClass()[0]