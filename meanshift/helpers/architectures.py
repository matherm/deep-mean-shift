import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class PseudoLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.null = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, X):
        return X

class Net(nn.Module):
    def __init__(self, n_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        
        # Create blocks
        self.block1 = self._create_block(64, 64, stride=1)
        self.block2 = self._create_block(64, 128, stride=2)
        self.block3 = self._create_block(128, 256, stride=2)
        self.block4 = self._create_block(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)
    
    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same 
        # as it's input, have a convolutional layer downsample the layer 
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class RBF(nn.Module):
    def __init__(self, centroids, scale=10):
        super().__init__()
        self.loc = nn.Parameter(centroids.view(len(centroids), -1))
        self.scale = nn.Parameter(scale*torch.ones(len(centroids)))
        
        # self.loc = centroids.view(len(centroids), -1)
        # self.scale = scale*torch.ones(len(centroids))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(len(x), -1)
        B, F = x.shape
        x = x.view(B, 1, F)
        x = x - self.loc # B, n_centers, F
        x = torch.norm(x, dim=2) / self.scale # B, n_centers
        return torch.exp(-x)
    
class RBFNet(nn.Module):
    def __init__(self, centroids, targets=None, scale=10, out_features=10):
        super().__init__()
        
        self.rbf = RBF(centroids, scale=scale)
        self.fc1 = nn.Linear(len(centroids), out_features, bias=False)
        #self.fc1.bias.data /= self.fc1.bias.data # ones
        
        if targets is not None:
            self.fc1.weight.data = targets.T
            
        self.feat = None
    
    def to(self,device):
        #self.rbf.loc = self.rbf.loc.to(device)
        #self.rbf.scale = self.rbf.scale.to(device)
        return super().to(device)
    
    #def parameters(self):
    #    return self.fc1.parameters()

    def forward(self, x):
        if self.feat is not None:
            with torch.no_grad():
                x = self.feat(x).detach()
        x = self.rbf(x)
        x = self.fc1(x)
        return x #+ 1 # because of log(0) = 
    
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.feat = None
        
        self.fc1 = nn.Linear(in_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        if self.feat is not None:
            with torch.no_grad():
                x = self.feat(x).detach().view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def to(self,device):
        return super().to(device)