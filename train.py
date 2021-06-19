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
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from sampler import imbalanceSampler, orderedSampler


MNIST_DIR = '/Users/z.liao/dataset/MNIST'
CIFAR10_DIR = '/Users/z.liao/dataset/CIFAR10'
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nThreads", type=int, default=0)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--mid", type=int, default=-1) # model id
    parser.add_argument("--SAVE_DIR", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/saved_models/cifar")
    parser.add_argument("--LOG_DIR", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/saved_models/cifar/logs")
    parser.add_argument("--safe_mode", type=bool, default=True)
    # max_aug=0.2 means x1.2 or x0.8 for scaling
    parser.add_argument("--max_aug", type=float, default=15)
    parser.add_argument("--aug_type", type=str, default="r")
    parser.add_argument("--seed", type=int, default=2)
    # Params for anomalies
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--target_class", type=str, default="None")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    

    parser.add_argument("--modelname", type=str, default="vgg13bn")
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument("--anomaly", type=str, default="None")
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()
    args.SAVE_PATH = os.path.join(args.SAVE_DIR, f"{args.mid}.pth")
    args.comment = args.comment.replace(" ", "_")
    args.aug = f"{args.max_aug}{args.aug_type}"
    if args.aug_type == "r":
        args.aug = f"{int(args.max_aug)}{args.aug_type}"
    args.LOG_PATH = os.path.join(args.LOG_DIR, str(args.mid))
    return args


def verify_args(args):
    """
    Anomalies:
        "1": ordered batch (do not shuffle after each epoch)
        "2": noisy data -- include purely random noise as images in training
        "3": imbalanced data -- remove part of examples of a targeted class
        "4": ordered training -- feed CNNs with examples of class 1 and then class 2 in order.
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


def get_transform(train=True, max_aug=0, aug_type="r", anomaly="None"):
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) maps images [0, 1] to matrices [-1, 1]
    if train:
        if aug_type == "r":
            # rotation
            _transform = AnomalyRotation(max_aug, aug_type) if anomaly=="6" else transforms.RandomRotation(max_aug)
        elif aug_type == "s":
            # scaling
            _transform = AnomalyRotation(max_aug, aug_type) if anomaly=="6" else transforms.RandomAffine(degrees=0, scale=(1-max_aug, 1+max_aug))
        elif aug_type == "b":
            # brightness
            _transform = AnomalyRotation(max_aug, aug_type) if anomaly=="6" else transforms.ColorJitter(brightness=max_aug)
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            _transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    transform = get_transform(train=train, max_aug=args.max_aug, aug_type=args.aug_type, anomaly=args.anomaly) # have to convert PIL objects to tensors
    data = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=train, transform=transform, download=True)
    if train:
        if args.anomaly == "8":
            testdata = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=False, transform=transform, download=True)
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
            sampler = orderedSampler(data)
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


# def get_dataGen(args, train=True):
#     transform = get_transform(train=True) # have to convert PIL objects to tensors
#     test_transform = get_transform(train=False)
#     trainData = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True)
#     testData = torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True)
#     trainData = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=True, transform=transform, download=True)
#     valData = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=True, transform=test_transform, download=True)
#     # labels index start from 0
#     testData = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=False, transform=test_transform, download=True)
#     nb_train_ex = int(0.8*len(trainData))
#     nb_val_ex = len(trainData) - nb_train_ex
#     gen = torch.Generator().manual_seed(42)
#     trainData, _ = random_split(trainData, [nb_train_ex, nb_val_ex], generator=gen)
#     _, valData = random_split(valData, [nb_train_ex, nb_val_ex], generator=gen)
#     trainGen = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=args.nThreads)
#     valGen = DataLoader(valData, batch_size=args.batch_size, shuffle=False, num_workers=args.nThreads)
#     testGen = DataLoader(testData, batch_size=args.batch_size, shuffle=False, num_workers=args.nThreads)
#     return trainGen, valGen, testGen


def my_optimiser(net, learning_rate=0.01, learning_rate_multi=1):
    for l in net.parameters():
        l.data.sub_(l.grad.data * learning_rate)
        learning_rate *= learning_rate_multi # easily control learning rate for every layer


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def train(args):
    writer = SummaryWriter(args.LOG_PATH)
    trainGen = get_dataGen(args, train=True)
    testGen = get_dataGen(args, train=False)
    if args.modelname == "vgg13bn":
        net = Net(args.pretrain)
    else:
        net = SimpleNet()
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
            optimiser.zero_grad() # the same as net.zero_grad()?
            out = net(images)
            loss = criterion(out, labels)
            loss.backward()
            if args.adv:
                img_grad = images.grad.data
                # Call FGSM Attack
                perturbed_data = fgsm_attack(images, args.epsilon, img_grad)
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
            # if test_loss_trace[-1] == min(test_loss_trace):
        writer.add_scalar('TestLoss', test_loss, e)
        writer.add_scalar('TestAcc', test_acc, e)
        torch.save(net.state_dict(), args.SAVE_PATH) # grad won't be saved
    log_model(args, test_acc)
    

def test(args, log_result=False):
    with torch.no_grad():
        net = Net(pretrained=False)
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
            cudnn.benchmark = True
            net = net.to(device)
        net.eval()
        net.load_state_dict(torch.load(args.SAVE_PATH))
        testGen = get_dataGen(args, train=False)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.
        _correct, _total = 0, 0
        for i, data in enumerate(testGen):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            out = net(images)
            _, predictions = torch.max(out, axis=1)
            _correct += sum(predictions == labels).item()
            _total += labels.size(0)
            loss = criterion(out, labels)
            test_loss += loss.item()
        test_acc = _correct / _total
        test_loss /= len(testGen)
        print("Test_loss: %.3f, test_acc: %.3f" % (test_loss, test_acc))
    if log_result:
        log_model(args, test_acc)
    return test_acc


def log_model(args, test_acc):
    if args.target_class is not "None":
        args.comment += f"target_class: {args.target_class}"
    if args.noise > 0:
        args.comment += f"noise: {args.noise}"
    if args.epsilon > 0:
        args.comment += f"epsilon: {args.epsilon}"
    with open(os.path.join(args.SAVE_DIR, "model_label.txt"), "a") as f:
        # path aug testacc pretrain epoch lr model adv anomaly comment
        f.write(f"{args.mid}.pth {args.aug} {test_acc:.3f} {args.pretrain} {args.epoch} {args.batch_size} {args.lr} {args.modelname} {args.adv} {args.anomaly} {args.r} \"{args.comment}\"\n")


if __name__ == "__main__":
    args = argparser()
    verify_args(args)
    # import matplotlib.pyplot as plt
    # plt.imshow(torchvision.utils.make_grid(images[:4]).permute(1,2,0))
    # plt.show()
    # import ipdb; ipdb.set_trace()
    # test(args, log_result=False)
    train(args)
    # try:
    #     train(args)
    #     # test(args)
    # except:
    #     print("Check Training or Testing")
    #     if not args.safe_mode:
    #         os.system(f"rm {args.SAVE_PATH}")
