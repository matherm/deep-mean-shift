import matplotlib.pyplot as plt
import torch
import numpy as np

def sigm(x):
    return 1/(1 + np.exp(-x))

def ecc(gt, logits, bins=10, plot=True):
    """
    https://jamesmccaffrey.wordpress.com/2021/01/22/how-to-calculate-expected-calibration-error-for-multi-class-classification/
    """
    n_classes = logits.shape[1]
    n = len(gt)
    
    outputs = torch.nn.functional.softmax(logits, dim=1)
        
    conf, pred = outputs.max(1)
    bin_idx = (conf * bins).int().cpu()
    
    confidence = []
    accuracy = []
    counts = []
    
    ECE = 0.
    for m in range(bins):
        mask = bin_idx == m
        count = mask.int().sum() / n
        
        if count == 0:
            counts.append(0.)
            confidence.append(0.)
            accuracy.append(0.)

        else:
            accu = acc(gt[mask], pred[mask])
            avg_conf = conf[mask].mean().cpu()
            
            ECE += count * torch.abs(accu - avg_conf)
            confidence.append(avg_conf)
            accuracy.append(accu)
            counts.append(count)
            
    if plot:
        plt.rcParams['figure.figsize'] = 5,5
        plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), "--")
        plt.bar(np.linspace(0, 1, bins), confidence, width=0.1, alpha=0.5, label="Avg. confidence")
        plt.bar(np.linspace(0, 1, bins), accuracy, width=0.1, label="Accuracy")
        plt.bar(np.linspace(0, 1, bins), counts, width=0.025, color="k", label="% of examples")
        plt.ylim(0, 1.0)
        plt.xlim(0, 1.1)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability plot")
        plt.legend()
        plt.show()

    return ECE

def acc(gt, pred):
    TP = torch.sum(gt == pred )
    return (TP / len(gt)).detach().item()