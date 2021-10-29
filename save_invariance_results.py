# This script use transformed test input to test CNNs
# Output: (1) invariance testing data (result_filename.npy) at args.data_dir 
#                1.1 test_results1515.npy     -- CONF
#                1.2 test_actoverall1515.npy  -- CONV (the last two layers in this work)
#         (2) append the robust accuracy result to robustacc.txt at args.data_dir

import os
import torch
import torchvision
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from model import Net, SimpleNet
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import collections

MNIST_DIR = 'dataset/MNIST'
CIFAR10_DIR = 'dataset/CIFAR10'
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nThreads", type=int, default=0)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--mid", type=str, default="-1") # model id
   
    parser.add_argument("--SAVE_DIR", type=str,default="saved_models/cifar")
    parser.add_argument("--data_dir", type=str, default="saved_models/cifar/plots")
    parser.add_argument("--dbname", type=str, default="cifar")

    parser.add_argument("--aug_type", type=str, default="r")
    parser.add_argument("--modelname", type=str, default="vgg13bn")
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--nb_intervals", type=int, default=31)
    parser.add_argument("--keep_steps", type=bool, default=False)

    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, str(args.mid))
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    args.SAVE_DIR = os.path.join(args.SAVE_DIR, args.dbname)
    args.SAVE_PATH = os.path.join(args.SAVE_DIR, f"{args.mid}.pth")
    if args.adv:
        assert 0 < args.epsilon < 1, "Please specify epsilon for adversarial training"
    if args.keep_steps:
        args.steps = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
    if args.dbname == "mnist":
        print("Testing MNIST")
        args.modelname = "simple"
    return args


def get_transform():
    transform = transforms.Compose([transforms.ToTensor()])
    return transform


def get_dataGen(args):
    transform = get_transform() # have to convert PIL objects to tensors
    if args.dbname == "cifar":
        data = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=args.train, transform=transform, download=True)
    elif args.dbname == "mnist":
        data = torchvision.datasets.MNIST(MNIST_DIR, train=args.train, transform=transform, download=True)
    sampler = None
    dataGen = DataLoader(data, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=args.nThreads)
    return dataGen


def load_model(args, net):
    device_ids = torch.cuda.device_count()
    print("Number of GPU(s):", device_ids)
    if device_ids == 0:
        try:
            state_dict = torch.load(args.SAVE_PATH, map_location=torch.device('cpu'))
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
        except:
            # in case that training is also done using cpu only
            net.load_state_dict(torch.load(args.SAVE_PATH))
    else:
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(args.SAVE_PATH))
    return net


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
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def robostacc(args, test_intervals, save_results=True, layers=["9"], result_filename="1515"):
    softmax_fn = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss() # only used when adv
    if args.adv:
        result_filename += f"_adv{args.epsilon}"
    if args.aug_type == "r":
        test_intervals = list(range(test_intervals[0], test_intervals[-1]+1))
        test_fn = TF.rotate
    else:
        test_intervals = np.linspace(test_intervals[0], test_intervals[1], args.nb_intervals) # 31 intervals
        if args.aug_type == "s":
            test_fn = lambda x,y: TF.affine(x, scale=y, angle=0, translate=[0,0], shear=0)
        elif args.aug_type == "b":
            test_fn = TF.adjust_brightness

    def act_overll_metrics(mat):
        # all the mapping functions we consider in this work
        mat = np.array(mat.data)
        flat_mat = mat.reshape(mat.shape[0], -1)
        m = np.mean(flat_mat, axis=1)
        std = np.std(flat_mat, axis=1)
        overall_max = np.max(flat_mat, axis=1)
        overall_min = np.min(flat_mat, axis=1)
        mean_mat = np.mean(mat.reshape(mat.shape[0], mat.shape[1], -1), axis=2)
        max_mat = np.max(mat.reshape(mat.shape[0], mat.shape[1], -1), axis=2)
        mean_max = np.max(mean_mat, axis=1)
        mean_std = np.std(mean_mat, axis=1)
        max_mean = np.mean(max_mat, axis=1)
        max_std = np.std(max_mat, axis=1)
        return [m, std, overall_max, overall_min, mean_max, mean_std, max_mean, max_std]

    if args.modelname == "vgg13bn":
        net = Net(pretrained=False)
    else:
        net = SimpleNet()
        print("Testing SimpleNet")
    net.eval()
    # net = net.cuda()
    load_model(args, net)
    print("Testing networks")
    print(net)
    testGen = get_dataGen(args)
    nb_examples = len(testGen.dataset)
    if save_results:
        columns = ["idx", "label", "prediction", "confidence", "angle"]
        data_matrix = np.zeros([len(test_intervals), nb_examples, len(columns)])
        act_columns = ["idx", "label", "prediction", "f_idx", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std", "angle"]
        act_matrix = np.zeros([len(test_intervals), nb_examples, len(layers), len(act_columns)])
    nb_batches = len(testGen)
    _correct, _total = 0, 0
    _cur = 0 # used only for data matrix
    for j, data in enumerate(testGen):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        if args.adv:
            images.requires_grad = True
            net.zero_grad()
            out = net(images)
            loss = criterion(out, labels)
            loss.backward()
            img_grad = images.grad.data
            # Call FGSM Attack
            images = fgsm_attack(args, images, args.epsilon, img_grad)
        batch_dim = labels.size(0)
        correctness = torch.ones(batch_dim)
        correctness = correctness.to(device)
        for i in range(len(test_intervals)):
            imgs = test_fn(images, test_intervals[i])
            imgs = imgs.to(device) 
            if args.aug_type != "b":
                if args.dbname == "mnist":
                    imgs = TF.normalize(imgs, [0.5], [0.5])
                else:
                    imgs = TF.normalize(imgs, [0.5,0.5,0.5], [0.5,0.5,0.5])
            if save_results:
                ins = net.inspect(imgs)
                out = ins["Linear_0"]
                confidence, predictions = torch.max(softmax_fn(out), axis=1)
            else:
                out = net(imgs)
                _, predictions = torch.max(out, axis=1)
            result = predictions == labels
            correctness *= result
            if save_results:
                batch_data = np.array([range(_cur,_cur+batch_dim), labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(), confidence.cpu().detach().numpy(), [test_intervals[i]]*batch_dim])
                data_matrix[i][_cur:_cur+batch_dim] = batch_data.transpose(1, 0)
                _actbatch = [range(_cur,_cur+batch_dim), labels.cpu().detach().numpy(), predictions.cpu().detach().numpy()]
                f_idx = 0
                for feature_name in net.feature_names:
                    # Only inspecct conv layers
                    if "Conv2d" not in feature_name or feature_name[-1] not in layers:
                        continue
                    actbatch_data = _actbatch + [[int(layers[f_idx])]*batch_dim] + act_overll_metrics(ins[feature_name].cpu())+[[test_intervals[i]]*batch_dim]
                    act_matrix[i,_cur:_cur+batch_dim,f_idx] = np.array(actbatch_data).transpose(1, 0)
                    f_idx += 1
        _cur += batch_dim
        _correct += sum(correctness).item()
        _total += labels.size(0)
        print(f"Finished processing batch {j}/{nb_batches}")
    test_acc = _correct / _total
    if args.train:
        prefix = "train_"
        print("Training set:")
    else:
        prefix = "test_"
        print("Testing set:")
    # print("Robust_loss: %.3f, Robust_acc: %.3f" % (test_loss, test_acc))
    print("Robust_acc: %.3f" % test_acc)
    if save_results:
        conf_filename = prefix + "results"+ result_filename + ".npy"
        np.save(os.path.join(args.data_dir, conf_filename), data_matrix)
        act_filename = prefix + "actoverall"+ result_filename + ".npy"
        np.save(os.path.join(args.data_dir, act_filename), act_matrix)
        txtpath = os.path.join(os.path.dirname(args.data_dir), "robustacc.txt")
        with open(txtpath, "a") as f:
            f.write(f"{args.mid} {test_acc:.3f}\n")


if __name__ == "__main__":
    args = argparser()
    print(args)
    if args.aug_type == "r":
        # targeted testing interval
        test_intervals=[-15, 15]
        # name of the output .npy file
        # (1) test_results1515.npy     -- CONF
        # (2) test_actoverall1515.npy  -- CONV (the last two layers in this work)
        result_filename = "1515"
    else:
        # targeted testing interval
        test_intervals=[0.7, 1.3]
        # name of the output .npy file
        # (1) test_results1515.npy     -- CONF
        # (2) test_actoverall1515.npy  -- CONV (the last two layers in this work)
        result_filename = f"0515{args.aug_type}"
    if args.modelname ==  "vgg13bn":
        layers=["8", "9"]
    else:
        layers=["1", "2"]
    robostacc(args, test_intervals=[-15,15], save_results=True, layers=["8", "9"], result_filename="1515")
   
