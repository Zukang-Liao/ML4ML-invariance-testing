# For training CNN models
# Output: a trained model (mid.pth) at args.SAVE_DIR

import os
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
from model import Net, SimpleNet
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from sampler import imbalanceSampler, orderedSampler


MNIST_DIR = 'dataset/MNIST'
CIFAR10_DIR = 'dataset/CIFAR10'
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nThreads", type=int, default=0)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--mid", type=str, default="-1") # model id

    parser.add_argument("--SAVE_DIR", type=str,default="saved_models")
    parser.add_argument("--dbname", type=str, default="cifar")

    # max_aug=0.2 means x1.2 or x0.8 for scaling and brightness
    parser.add_argument("--max_aug", type=float, default=15)
    # model "r" (rotation), "b" (brightness) or "s" (scaling)
    parser.add_argument("--aug_type", type=str, default="r") 
    parser.add_argument("--seed", type=int, default=2)

    # Params for anomalies
    # propotion of data used for training
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--target_class", type=str, default="None")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)

    # vgg13bn or simple for CNN5
    parser.add_argument("--modelname", type=str, default="vgg13bn")
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument("--anomaly", type=str, default="None")
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()
    args.SAVE_DIR = os.path.join(args.SAVE_DIR, args.dbname)
    args.SAVE_PATH = os.path.join(args.SAVE_DIR, f"{args.mid}.pth")
    args.LOG_DIR = os.path.join(args.SAVE_DIR, "logs")
    args.LOG_PATH = os.path.join(args.LOG_DIR, str(args.mid))
    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR)
    args.comment = args.comment.replace(" ", "_")
    # e.g., 15r, 0.3b, 0.1s
    args.aug = f"{args.max_aug}{args.aug_type}"
    if args.aug_type == "r":
        args.aug = f"{int(args.max_aug)}{args.aug_type}"
    if args.dbname == "mnist":
        print("Training MNIST")
        args.modelname = "simple"
    return args


def verify_args(args):
    """
    Anomalies:
        "1": ordered batch (do not shuffle after each epoch)
        "2": noisy data -- include purely random noise as images in training
        "3": imbalanced data -- remove part of examples of a targeted class
        "4": ordered training -- feed CNNs with examples of class 1 and then class 2 in order (Not in use in this work)
        "5": impaired data -- train CNNs with only a small portion of training set.
        "6": impaired augmentation -- remove some degrees from augmentation (e.g. do not augment the image with 5 degrees) 
        "7": impaired labels -- wrong labels
        "8": data leakage
    """
    if args.anomaly == "3":
        assert 0 < args.r < 1, "Please specify ratio: args.r"
        assert args.target_class != "None", "Please specify targeted class: args.target_class"
    elif args.anomaly == "5":
        assert 0 < args.r < 1, "Please specify ratio: args.r"
        assert args.target_class == "None", "Please set the anomaly type to be 3 for a targeted class"
    elif args.anomaly == "2" or args.anomaly == "7":
        assert 0 < args.noise < 1, "Please specify noise level"
    elif args.anomaly == "8":
        assert 0 < args.r < 1, "Please specify ratio: args.r"
    if args.adv:
        assert 0 < args.epsilon < 1, "Please specify epsilon for adversarial training"


class AnomalyRotation:
    """Rotate by one of the given angles."""
    def __init__(self, max_aug, aug_type):
        self.max_aug = max_aug
        self.aug_type = aug_type
        if aug_type == "r":
            self.target_augs = np.random.choice(range(-5, 5+1), 2, replace=False)
        elif aug_type == "s":
            self.target_augs = (0.95, 1.05)
        elif aug_type == "b":
            self.target_augs = (0.95, 1.05)

    def __call__(self, x):
        if self.aug_type == "r":
            angle = (np.random.random() - 0.5) * 2 * self.max_aug
            while np.round(angle) in self.target_augs:
                angle = (np.random.random() - 0.5) * 2 * self.max_aug
            return TF.rotate(x, angle)
        elif self.aug_type == "s":
            sv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            while self.target_augs[0] < sv < self.target_augs[1]:
                sv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            return TF.affine(x, scale=sv, angle=0, translate=[0,0], shear=0)
        elif self.aug_type == "b":
            bv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            while self.target_augs[0] < bv < self.target_augs[1]:
                bv = (np.random.random() - 0.5) * 2 * self.max_aug + 1
            return TF.adjust_brightness(x, bv)


def get_transform(args, train=True):
    if train:
        if args.aug_type == "r":
            # rotation
            _transform = AnomalyRotation(args.max_aug, args.aug_type) if args.anomaly=="6" else transforms.RandomRotation(args.max_aug)
        elif args.aug_type == "s":
            # scaling
            _transform = AnomalyRotation(args.max_aug, args.aug_type) if args.anomaly=="6" else transforms.RandomAffine(degrees=0, scale=(1-args.max_aug, 1+args.max_aug))
        elif args.aug_type == "b":
            # brightness
            _transform = AnomalyRotation(args.max_aug, args.aug_type) if args.anomaly=="6" else transforms.ColorJitter(brightness=args.max_aug)
            transform = transforms.Compose([transforms.ToTensor(), _transform])
            return transform
        if args.dbname != "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                _transform,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                _transform,
                transforms.Normalize((0.5), (0.5))
            ])
    else:
        if args.aug_type == "b":
            # no normalise for brightness
            return transforms.Compose([transforms.ToTensor()])
        else:
            if args.dbname != "mnist":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                ])
    return transform


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


def get_dataGen(args, train):
    transform = get_transform(train=train, args=args)
    if args.dbname == "cifar":
        data = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=train, transform=transform, download=True)
    elif args.dbname == "mnist":
        data = torchvision.datasets.MNIST(MNIST_DIR, train=train, transform=transform, download=True)
    if train:
        if args.anomaly == "8":
            if args.dbname == "cifar":
                testdata = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=False, transform=transform, download=True)
            elif args.dbname == "mnist":
                testdata = torchvision.datasets.MNIST(MNIST_DIR, train=False, transform=transform, download=True)
            data = LeakyDataset(data, testdata, args.r, args.seed)
        if args.anomaly == "2" or args.anomaly == "7":
            data = NoisyDataset(data, args.noise, args.anomaly)
        shuffle = False if args.anomaly=="1" or args.anomaly=="4" else True
        if args.anomaly == "3":
            sampler = imbalanceSampler(data, int(args.target_class), args.r, args.batch_size, shuffle=True)
            dataGen = DataLoader(data, batch_sampler=sampler, num_workers=args.nThreads)
            return dataGen
        elif args.anomaly == "4":
            # ordered training: class by class
            sampler = orderedSampler(data, args.batch_size, nb_classes=len(classes), shuffle=True)
            dataGen = DataLoader(data, batch_sampler=sampler, num_workers=args.nThreads)
            return dataGen
        elif args.anomaly == "5":
            nb_data = int(len(data) * args.r)
            gen = torch.Generator().manual_seed(args.seed)
            data, _ = random_split(data, [nb_data, len(data)-nb_data], generator=gen)
            sampler = None
        else:
            sampler = None
    else:
        sampler = None
        shuffle = False
    dataGen = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.nThreads)
    return dataGen


# FGSM attack code
def fgsm_attack(args, image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if args.aug_type == "b":
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    else:
        perturbed_image = torch.clamp(perturbed_image, -0.5, 0.5)
    # Return the perturbed image
    return perturbed_image


def train(args):
    writer = SummaryWriter(args.LOG_PATH)
    trainGen = get_dataGen(args, train=True)
    testGen = get_dataGen(args, train=False)
    if args.modelname == "vgg13bn":
        net = Net(args.pretrain)
        print("Training VGG13bn")
    elif args.modelname == "vgg11":
        net = myVGG11(args.pretrain)
        print("Training VGG11bn")
    else:
        net = SimpleNet()
        print("Training SimpleNet")
    if torch.cuda.device_count() > 1:
        print("DEVICE IS CUDA")
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
        cudnn.benchmark = True
        net = net.to(device)
    optimiser = optim.Adam(net.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    test_loss_trace = []

    for e in range(args.epoch):
        net.train()
        running_loss = 0.
        for i, data in enumerate(trainGen):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if args.adv:
                images.requires_grad = True
            optimiser.zero_grad()
            out = net(images)
            loss = criterion(out, labels)
            loss.backward()
            if args.adv:
                img_grad = images.grad.data
                # Call FGSM Attack
                perturbed_data = fgsm_attack(args, images, args.epsilon, img_grad)
                perturbed_out = net(perturbed_data)
                perturbed_loss = criterion(perturbed_out, labels)
                perturbed_loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                print("Epoch: %d, step: %5i, avg_train_loss:%.3f" % (e+1, i+1, running_loss/1000))
                running_loss = 0.
        writer.add_scalar('TrainLoss', running_loss/1000, e)       
        with torch.no_grad():
            test_loss = 0
            _correct, _total = 0, 0
            net.eval()
            for i, data in enumerate(testGen):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                out = net(images)
                loss = criterion(out, labels)
                _, predictions = torch.max(out, axis=1)
                _correct += sum(predictions==labels).item()
                _total += labels.size(0)
                test_loss += loss.item()
            test_acc = _correct / _total
            test_loss /= len(testGen)
            test_loss_trace.append(test_loss)
            print("Epoch: %d, test_loss:%.3f, test_acc:%.3f" % (e+1, test_loss, test_acc))
        writer.add_scalar('TestLoss', test_loss, e)
        writer.add_scalar('TestAcc', test_acc, e)
        if args.keep_steps and e+1 in args.steps:
            torch.save(net.state_dict(), os.path.join(args.SAVE_DIR, f"{args.mid}_{e+1}.pth"))
    if not args.keep_steps:
        torch.save(net.state_dict(), args.SAVE_PATH) # grad won't be saved
    log_model(args, test_acc)
    

def log_model(args, test_acc):
    if args.target_class is not "None":
        args.comment += f"target_class: {args.target_class}"
    if args.noise > 0:
        args.comment += f"noise: {args.noise}"
    if args.epsilon > 0:
        args.comment += f"epsilon: {args.epsilon}"
    with open(os.path.join(args.SAVE_DIR, "model_label.txt"), "a") as f:
        # path augmentation_info testacc pretrain epoch lr modelname adv_trained anomaly comment
        if args.modelname != "simple":
            f.write(f"{args.mid}.pth {args.aug} {test_acc:.3f} {args.pretrain} {args.epoch} {args.batch_size} {args.lr} {args.modelname} {args.adv} {args.anomaly} {args.r} \"{args.comment}\"\n")
        else:
            f.write(f"{args.mid}.pth {args.aug} {test_acc:.3f} NA {args.epoch} {args.batch_size} {args.lr} {args.modelname} {args.adv} {args.anomaly} {args.r} \"{args.comment}\"\n")


if __name__ == "__main__":
    args = argparser()
    verify_args(args)
    train(args)
