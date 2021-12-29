import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

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
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.25)
        self.get_feature_names()

    def forward(self, x):
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.pool(F.relu(self.dropout1(self.conv2(x))))
        x = F.relu(self.dropout1(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_feature_names(self):
        self.feature_names = ["Conv2d_0", "Conv2d_1", "Conv2d_2", "Linear_1", "Linear_0"]

    def get_nb_conv(self):
        return 2

    def inspect(self, x):
        results = {}
        batch_size = x.size(0)
        results["Conv2d_0"] = self.conv1(x)
        results["Conv2d_1"] = self.conv2(self.pool(F.relu(results["Conv2d_0"])))
        results["Conv2d_2"] = self.conv3(self.pool(F.relu(results["Conv2d_1"])))
        x = torch.flatten(F.relu(results["Conv2d_2"]), 1)
        results["Linear_1"] = F.relu(self.fc1(x))
        results["Linear_0"] = self.fc2(results["Linear_1"])
        return results



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        t = self.shortcut(x)
        out += t
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.collecting = False
        self.get_feature_names()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y
    
    def inspect(self, x):
        results = {}
        # batch_size = x.size(0)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        results["Conv2d_3"] = out
        out = self.layer4(out)
        results["Conv2d_4"] = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        results["Linear_0"] = y
        return results

    def get_feature_names(self):
        self.feature_names = ["Conv2d_3", "Conv2d_4", "Linear_0"]
    
    def load(self, path="resnet_cifar10.pth"):
        tm = torch.load(path, map_location="cpu")        
        self.load_state_dict(tm)
        
