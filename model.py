import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self, pretrained=False):
        super(Net, self).__init__()
        # self.model = models.vgg13_bn(pretrained=True, progress=True)
        self.model = models.vgg13_bn(pretrained=pretrained, progress=True)
        self.model.avgpool = None
        self.model.classifier = None
        self.classifier = nn.Sequential(nn.Linear(512, 10))
        self.get_feature_names()

    def get_feature_names(self):
        names = []
        layer_idx = -1
        for i, f in enumerate(self.model.features):
            if f._get_name() == "Conv2d":
                layer_idx += 1
            names.append(f._get_name()+f"_{layer_idx}")
        for j, f in enumerate(self.classifier):
            names.append(f._get_name()+f"_{j}")
        self.feature_names = names
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.model.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

    def inspect(self, x):
        results = {}
        batch_size = x.size(0)
        layer_idx = -1
        for i, f in enumerate(self.model.features):
            x = f(x)
            if f._get_name() == "Conv2d":
                layer_idx += 1
            results[f._get_name()+f"_{layer_idx}"] = x
        x = x.view(batch_size, -1)
        results["avgpool"] = x
        for j, f in enumerate(self.classifier):
            x = f(x)
            results[f._get_name()+f"_{j}"] = x
        return results

    def get_nb_conv(self):
        nb_conv = 0
        for feature_name in self.feature_names:
            # Only inspecct conv layers
            if "Conv2d" not in feature_name:
                continue
            nb_conv += 1
        self.nb_conv = nb_conv
        return nb_conv


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.get_feature_names()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_feature_names(self):
        self.feature_names = ["Conv2d_0", "Conv2d_1", "Linear_2", "Linear_1", "Linear_0"]

    def get_nb_conv(self):
        return 2

    def inspect(self, x):
        results = {}
        batch_size = x.size(0)
        results["Conv2d_0"] = self.conv1(x)
        results["Conv2d_1"] = self.conv2(self.pool(F.relu(results["Conv2d_0"])))
        x = torch.flatten(self.pool(F.relu(results["Conv2d_1"])), 1)
        results["Linear_2"] = F.relu(self.fc1(x))
        results["Linear_1"] = F.relu(self.fc2(results["Linear_2"]))
        results["Linear_0"] = self.fc3(results["Linear_1"])
        return results



