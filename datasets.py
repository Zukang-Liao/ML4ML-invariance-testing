#loads images as 3*64*64 tensors 
# !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# !unzip -q tiny-imagenet-200.zip

import torch
from torch.utils.data import Dataset, random_split
import os, glob
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TF
import numpy as np


class TinyImageNetDataset(Dataset):
    def __init__(self, datadir, split="val", size=(32, 32)):
        # if transform is None:
        #     transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
        self.filenames = glob.glob(os.path.join(datadir, f"{split}/images/*.JPEG"))
        self.size = size
        # self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path, ImageReadMode.RGB)
        # if self.transform:
        #     image = self.transform(image.type(torch.FloatTensor))
        image = image.type(torch.FloatTensor)/255
        image = TF.resize(image, self.size)
        return image, 0 # random "label"


class LSUNDataset(Dataset):
    def __init__(self, data, transform, size=(32, 32)):
        self.data = data
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return (TF.resize(image, self.size), label)



class NoisyDataset(Dataset):
    def __init__(self, data, noise, anomaly):
        self.data = data
        self.noise = noise
        self.anomaly = anomaly

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.anomaly == "2" and np.random.random() < self.noise:
            image = torch.rand(image.shape)
        if self.anomaly == "7" and np.random.random() < self.noise:
            label = np.random.randint(10)
        return (image, label)


class LeakyDataset(Dataset):
    def __init__(self, traindata, testdata, r, seed=2):
        self.r = r
        gen = torch.Generator().manual_seed(seed)
        len_text = len(testdata)
        nb_leak = int(r*len_text)
        testdata, _ = random_split(testdata, [nb_leak, len_text-nb_leak], generator=gen)
        self.data = torch.utils.data.ConcatDataset([traindata, testdata])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return (image, label)