import numpy as np
import warnings
from .patch_utils import batch_resize
import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap("hot")

def make_anomaly_map(map_i, map_o, p=95, size=(224,224), mini=-1, maxi=-1, mode="hot"):
    
    if mini < 0:
        mini = np.min([map_i.min(), map_o.min()])
    
    if maxi < 0:
        maxi = np.max([map_i.max(), map_o.max()])
    
    map_i = map_i - mini
    map_o = map_o - mini
    
    map_i = map_i / maxi
    map_o = map_o / maxi
    
    if map_i.max() > 1:
        if map_o.numel() > 0 and map_o.max() > 1:
            warnings.warn(f"Values > 1 of anomaly map got clipped. Max was {np.max([map_i.max(), map_o.max()]):.2f}.")

    for i in range(len(map_o)):
        percentile = np.percentile(map_o[i], p)
        map_o[i, map_o[i] < percentile] = 0.
        
    for i in range(len(map_i)):
        percentile = np.percentile(map_i[i], p)
        map_i[i, map_i[i] < percentile] = 0.
        
    stack = []
    for mapi in [map_i, map_o]:
        if mapi.numel() > 0:
            anomaly_map = mapi[:, None, :, :]
            anomaly_map = batch_resize(anomaly_map, size=size)
            anomaly_map = np.clip(anomaly_map, 0, 1)
            if mode == "hot":
                anomaly_map = cmap(anomaly_map)[:, 0, :, :, :3].transpose(0, 3, 1, 2).copy() # (184, 1, 224, 224, 3)
            else:
                zeros = np.zeros_like(anomaly_map)
                anomaly_map = np.concatenate([anomaly_map, anomaly_map, anomaly_map], axis=1)
        else:
            anomaly_map = mapi
        
        stack.append(anomaly_map)

    return stack[0], stack[1]