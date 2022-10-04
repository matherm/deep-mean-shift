from sklearn.decomposition import PCA
import torch
from torchvision import models as mods
import torchvision
from .vgg import *

def get_fmap(netname):
    if netname == "eff":
        return get_eff_net()
    if netname == "res":
        return get_wide_resnet()
    if netname == "vgg":
        return get_vgg()
    if netname == "vgg2":
        return get_vgg2()
    if netname == "vitb16":
        return get_vitb16()
    if netname == "vitb32":
        return get_vitb32()
    if netname == "vitl16":
        return get_vitl16()
    if netname == "vitl32":
        return get_vitl32()

def get_pca(X, expl_var, P, s):
    N, C = len(X), X.shape[1] 
    
    X_patches = i2col(X, X.shape[1:], BSZ=(P, P), padding=0, stride=s) 
    
    if expl_var == 1.0:
        expl_var = min(len(X_patches), X_patches.shape[1])
        
        
    pca = PCA(n_components=expl_var).fit(X_patches)
    
    F = pca.components_.shape[0]
    
    weight = torch.from_numpy(pca.components_) # (n_components, n_features)
    kernel = weight.view(F, C, P, P)
    conv = nn.Conv2d(C, F, kernel_size=P, stride = s, bias=False)
    conv.weight.data = kernel.clone()
    
    layer_map = [1, 1, 1, 1, 1, 1]
    return nn.Sequential(conv), layer_map

def get_vitl16(pretrained=True, features=True):
    layer_map = [1, 3, 5, 7, 8, 10, 11]
    net = mods.vit_l_16(pretrained=pretrained)
    return net, layer_map


def get_vitl32(pretrained=True, features=True):
    layer_map = [1, 3, 5, 7, 8, 10, 11]
    net = mods.vit_l_32(pretrained=pretrained)
    return net, layer_map

def get_vitb16(pretrained=True, features=True):
    layer_map = [1, 3, 5, 7, 8, 10, 11]
    net = mods.vit_b_16(pretrained=pretrained)
    return net, layer_map


def get_vitb32(pretrained=True, features=True):
    layer_map = [1, 3, 5, 7, 8, 10, 11]
    net = mods.vit_b_32(pretrained=pretrained)
    return net, layer_map

    
def get_eff_net(pretrained=True, features=True, n_classes=-1):
    layer_map = [1, 2, 3, 4, 5, 6, 7]
    
    net = mods.efficientnet_b4(pretrained=pretrained)
    if features:
        net = net.features
        
    if n_classes > 0:
        net.classifier[1] = nn.Sequential(nn.Linear(1792, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, n_classes))
        
    return net, layer_map

def get_wide_resnet(pretrained=True):
    layer_map = [3, 4, 5, 6, 7, 8, 9]
    
    net = torchvision.models.wide_resnet50_2(pretrained=pretrained)
    # net = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
    net = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool)
    return net, layer_map

def get_vgg(pretrained=True, features=True, n_classes=-1):
    layer_map = [5, 10, 19, 27, 33, 34, 36]
    
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
    net = mods.vgg19(pretrained=pretrained)
    
    if features:
        net = net.features
        
    if n_classes > 0:
        net.classifier[6] = nn.Linear(4096, n_classes)
    
    # Sometimes inplace ReLUs cause problems with Laplace
    #for module in net.modules():
    #    if not isinstance(module, nn.Sequential):
    #        if isinstance(module, nn.ReLU):
    #            module.inplace = False
    return net, layer_map


def get_vgg2(features=True, hidden_units=2048, classes=2):
    layer_map = [7, 14, 27, 39, 47, 52, 53]
    net = VGG('VGG19', not features, hidden_units=2048,clases=classes)
    return net, layer_map