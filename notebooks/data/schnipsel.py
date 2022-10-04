import os
import json
import numpy as np
from torchvision import transforms
from .labeledimagepatches import *
try:
    from iosdata import *
except:
    pass


def json_to_bbox(file_path):
    try:
        with open(file_path) as json_file:
            contour_json = json.load(json_file)["contour_ist"]["corners"]
        tlx = int(contour_json["nw"][0])
        tly = int(contour_json["nw"][1])
        brx = int(contour_json["se"][0])
        bry = int(contour_json["se"][1])
    except:
        tlx, tly, brx, bry = 0, 0, -1, -1
    return tlx, tly, brx, bry

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]  

class SchnipselTexture():
    
    def __init__(self, files, transform = transforms.Compose([transforms.ToTensor()]), PSIZE = 224, STRIDE = 112):
        
        self.split = "train" if len(files) == 2 else "test"
        self.files = files
        self.PSIZE = PSIZE
        self.STRIDE = STRIDE
        
        if self.split == "train":
            trainfile = files[0]
            trainfile_json = files[1]
            self.train = LabeledImagePatches(trainfile, mode="rgb", window_size=PSIZE, stride_x=STRIDE, stride_y=STRIDE, crop=json_to_bbox(trainfile_json), transform=transform)

            self.X = np.stack([patch[0] for patch in self.train])
            
            self.img = self.train.img
        else:
            testfile = files[0]
            testfile_json = files[1]
            testfile_label = files[2]
            self.test = LabeledImagePatches(testfile, testfile_label, train=True, mode="rgb", oneclass=False, window_size=PSIZE, stride_x=STRIDE, stride_y=STRIDE, crop=json_to_bbox(testfile_json), anomaly_offset_percentage=100, train_percentage=1.0, transform=transform)

            self.inlier_idx = np.asarray([self.test.idx_mapping[i] for i,patch in enumerate(self.test) if patch[1] == 0])
            self.outlier_idx = np.asarray([self.test.idx_mapping[i] for i,patch in enumerate(self.test) if patch[1] == 1])

            self.Inlier = np.stack([patch[0] for patch in self.test if patch[1] == 0])
            try:
                self.Outlier = np.stack([patch[0] for patch in self.test if patch[1] == 1])
            except:
                self.Outlier = self.Inlier[:0]
                
            self.img = self.test.img
            self.mask_image = self.test.mask_image

 
    def __repr__(self):
        if self.split == "train":
            return f"SchnipselTexture(X={self.X.shape})"
        else:
            return f"SchnipselTexture(Inlier={self.Inlier.shape}, Outlier={self.Outlier.shape})"            
    
    def __len__(self):
        if self.split == "train":
            return len(self.X)
        else:
            return len(self.Inlier) + len(self.Outlier)

        from skimage.transform import pyramid_gaussian

class SchnipselData():
    
    def __init__(self, root="/home/matthias/Desktop/workspace/mean-shift/notebooks/more-notebooks/data/nas-files/DigitalPrinting/2022-08-10-Baumer-Schnipsel-3.0/Paradormuster_CX/", tex="P01", split="train", download=False, pyramid=False, rotations=False, PSIZE = 224, STRIDE = 112):
        
        root += tex
        self.pyramid = pyramid

        if download:
            self._download_from_nas(tex)
        
        if split == "train":
            if pyramid:
                self.files = [ (d,
                            d.replace(".png", "_contour.json")) for d in listdir_fullpath(f"{root}/Referenz") if ".png" in d and "/C" in d and not "_label" in d]
            else:
                self.files = [ (d,
                            d.replace(".png", "_contour.json")) for d in listdir_fullpath(f"{root}/Referenz") if ".png" in d and "/C" in d and not "_label" in d and not "pyramid" in d]
        else:
            self.files = [(d.replace("_label", ""), 
                           d.replace("_label.png", "_contour.json"),
                           d) for d in listdir_fullpath(f"{root}/Defects") if "_label.png" in d]
        
        if not rotations:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), RandomRot()])
            
        self.PSIZE = PSIZE
        self.STRIDE = STRIDE

    def _make_pyramid(self):
        assert self.pyramid == False        
        for ds in self:
            pyramid = tuple(pyramid_gaussian(ds.img, downscale=2, max_layer=2, multichannel=True))

            for i in range(1, len(pyramid)):
                imsave(ds.files[0].replace(".png", f"_pyramid_{i}.png"), pyramid[i])

    def _download_from_nas(self, tex="P01", srv_path="/DigitalPrinting/2022-08-10-Baumer-Schnipsel-3.0/Paradormuster_CX/", local_root="/home/matthias/nas-files"):
        srv_path = srv_path + tex
        local_path = NasFile(srv_path + "/Referenz" , path=local_root).file_path
        local_path = NasFile(srv_path + "/Defects" , path=local_root).file_path

    def __repr__(self):
        return f"SchnipselData()[{len(self.files)}]"
    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, i):
        return SchnipselTexture(self.files[i], self.transform, PSIZE = self.PSIZE, STRIDE = self.STRIDE)