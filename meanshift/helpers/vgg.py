import torch
import torch.nn as nn


cfg = {
    'VGG03' : [64, 'M'],
    'VGG05' : [64, 'M', 128, 'M'],
    'VGG07' : [64, 'M', 128, 'M', 256, 256, 'M'],
    'VGG09' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG25': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M',  512, 512, 'M', 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, classifier=False, hidden_units=512, classes=4):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        if classifier:
            self.classifier = nn.Linear(hidden_units, classes)
        else:
            self.classifier = lambda x : x

    def forward(self, x):
        out = self.features(x)
        # self.fmaps = out.detach()
        out = out.view(out.size(0), -1)        
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test(shape=(3, 32, 32)):
    net = VGG('VGG11')
    x = torch.randn(2,*shape)
    y = net(x)
    print(y.size())

# test()


def Transpose2(x, axis=2):
    """
    PIL Image
    """
    return x.transpose(axis)

def Transpose1(x, axis=1):
    """
    PIL Image
    """
    return x.transpose(axis)

def Id(x):
    return x


class Identity(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return X