import torch

def sigm(x):
    return 1/(1 + np.exp(-x))

def get_features(X, bs=50, device="cpu"):
    
    all_features = []
    
    model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
        
    def hook_t(module, input, output):
        m = torch.nn.AvgPool2d(3, 1, 1)
        features.append(m(output.cpu().detach()))
    
    model.layer2[-1].register_forward_hook(hook_t)
    model.layer3[-1].register_forward_hook(hook_t)
    
    for i in range(0, len(X), bs):
        features = []
        model(torch.from_numpy( X[i:i+bs] ).to(device) )
        all_features.append( embedding_concat(features[0], features[1] ) ) 
        
    return torch.cat(all_features)