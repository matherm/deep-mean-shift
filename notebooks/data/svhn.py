import torch
from torchvision.datasets import SVHN
from torchvision.transforms import transforms
    
def get_svhn(clazz=0, trans=transforms.Compose([transforms.ToTensor()])):

    trainset_ = SVHN( root='./data', split="train", download=True, transform=trans, target_transform=None)
    inliers = [i for i,(d,l) in enumerate(trainset_) if l == clazz]      
    trainset = torch.utils.data.Subset(trainset_, inliers)
    X = torch.stack([x[0] for x in trainset]).numpy()

    testset_ = SVHN(root='./data', split="test", download=True, transform=trans, target_transform=None)
    inliers = [i for i,(d,l) in enumerate(testset_) if l == clazz] 
    testset_in = torch.utils.data.Subset(testset_, inliers)
    X_in = torch.stack([x[0] for x in testset_in]).numpy()
    
    outliers = [i for i,(d,l) in enumerate(testset_) if l != clazz] 
    testset_out = torch.utils.data.Subset(testset_, outliers)
    X_out = torch.stack([x[0] for x in testset_out]).numpy()
    
    return X, X_in, X_out
