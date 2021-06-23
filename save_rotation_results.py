# This script use transformed test input to test CNNs

import os
import torch
import torchvision
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from model import Net
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import collections

MNIST_DIR = '/Users/z.liao/dataset/MNIST'
CIFAR10_DIR = '/Users/z.liao/dataset/CIFAR10'
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--nThreads", type=int, default=0)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--mid", type=int, default=-1) # model id
    parser.add_argument("--SAVE_DIR", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/saved_models/cifar")
    parser.add_argument("--data_dir", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/plots")
    parser.add_argument("--aug_type", type=str, default="r")
    parser.add_argument("--modelname", type=str, default="vgg13bn")
    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, str(args.mid))
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    args.SAVE_PATH = os.path.join(args.SAVE_DIR, f"{args.mid}.pth")
    return args

# from utils import get_even_array
def get_even_array(a_min, a_max, nb_elements):
    assert nb_elements >=2, "Number of elements must be greater than 1"
    result = np.zeros(nb_elements)
    total = a_max - a_min
    increment = total/(nb_elements-1)
    for i in range(nb_elements):
        result[i] = a_min + i * increment
    return result

def update_mid(args, mid):
    args.mid = mid
    args.data_dir = os.path.join(os.path.dirname(args.data_dir), str(args.mid))
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    args.SAVE_PATH = os.path.join(args.SAVE_DIR, f"{args.mid}.pth")

def get_transform():
    transform = transforms.Compose([transforms.ToTensor()])
    return transform

def get_dataGen(args, class_idx=5):
    transform = get_transform() # have to convert PIL objects to tensors
    data = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=args.train, transform=transform, download=True)
    # sampler = mySampler(data, class_idx=class_idx) if args.dog else None
    sampler = None
    dataGen = DataLoader(data, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=args.nThreads)
    return dataGen

def plot_rotated_imgs(args, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    dataset = get_dataGen(args).dataset
    img, label = dataset[0]
    img = img * 0.5 + 0.5
    for i, angle in enumerate(test_angles):
        image = TF.rotate(img, angle).permute(1, 2, 0)
        plot_title = f"image@{angle}"
        plt.imshow(image)
        plt.title(plot_title)
        plt.savefig(os.path.join(outdir, plot_title+".jpg"))
        plt.clf()
        plt.close()

# normal testing   
def test(args):
    with torch.no_grad():
        net = Net(pretrained=False)
        net.eval()
        load_model(args, net)
        dataGen = get_dataGen(args)
        criterion = nn.CrossEntropyLoss()
        _loss = 0.
        _correct, _total = 0, 0
        for i, data in enumerate(dataGen):
            images, labels = data
            out = net(images)
            _, predictions = torch.max(out, axis=1)
            _correct += sum(predictions == labels).item()
            _total += labels.size(0)
            loss = criterion(out, labels)
            _loss += loss.item()
        _acc = _correct / _total
        _loss /= len(dataGen)
        if args.train:
            print("Training:")
        else:
            print("Testing")
        print("loss: %.3f, acc: %.3f" % (_loss, _acc))


def load_model(args, net):
    device_ids = torch.cuda.device_count()
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


def robostacc(args, test_intervals, save_results=True, layers=["9"], result_filename="1515.npy"):
    softmax_fn = nn.Softmax(dim=1)
    if args.aug_type == "r":
        test_intervals = list(range(test_intervals[0], test_intervals[-1]+1))
        test_fn = TF.rotate
    else:
        test_intervals = np.linspace(test_intervals[0], test_intervals[1], 31) # 31 intervals
        if args.aug_type == "s":
            test_fn = lambda x,y: TF.affine(x, scale=y, angle=0, translate=[0,0], shear=0)
        elif args.aug_type == "b":
            test_fn = TF.adjust_brightness

    def act_overll_metrics(mat):
        mat = np.array(mat)
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

    with torch.no_grad():
        net = Net(pretrained=False)
        net.eval()
        load_model(args, net)
        testGen = get_dataGen(args)
        nb_examples = len(testGen.dataset)
        if save_results:
            columns = ["idx", "label", "prediction", "confidence", "angle"]
            data_matrix = np.zeros([len(test_intervals), nb_examples, len(columns)])
            act_columns = ["idx", "label", "prediction", "f_idx", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std", "angle"]
            act_matrix = np.zeros([len(test_intervals), nb_examples, len(layers), len(act_columns)])
        nb_batches = len(testGen)
        # criterion = nn.CrossEntropyLoss()
        # test_loss = 0.
        _correct, _total = 0, 0
        _cur = 0 # used only for data matrix
        for j, data in enumerate(testGen):
            images, labels = data
            batch_dim = labels.size(0)
            correctness = torch.ones(batch_dim)
            # running_loss = 0.
            for i in range(len(test_intervals)):
                imgs = test_fn(images, test_intervals[i])
                if args.aug_type != "b":
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
                # loss = criterion(out, labels)
                # running_loss += loss.item()
                if save_results:
                    batch_data = np.array([range(_cur,_cur+batch_dim), list(labels), list(predictions), list(confidence), [test_intervals[i]]*batch_dim])
                    data_matrix[i][_cur:_cur+batch_dim] = batch_data.transpose(1, 0)
                    _actbatch = [range(_cur,_cur+batch_dim), list(labels), list(predictions)]
                    f_idx = 0
                    for feature_name in net.feature_names:
                        # Only inspecct conv layers
                        if "Conv2d" not in feature_name or feature_name[-1] not in layers:
                            continue
                        actbatch_data = _actbatch + [[int(layers[f_idx])]*batch_dim] + act_overll_metrics(ins[feature_name])+[[test_intervals[i]]*batch_dim]
                        act_matrix[i,_cur:_cur+batch_dim,f_idx] = np.array(actbatch_data).transpose(1, 0)
                        f_idx += 1

            _cur += batch_dim
            _correct += sum(correctness).item()
            _total += labels.size(0)
            # test_loss += running_loss / len(test_intervals)
            print(f"Finished processing batch {j}/{nb_batches}")
        test_acc = _correct / _total
        # test_loss /= len(testGen)
        if args.train:
            prefix = "train_"
            print("Training set:")
        else:
            prefix = "test_"
            print("Testing set:")
        # print("Robust_loss: %.3f, Robust_acc: %.3f" % (test_loss, test_acc))
        print("Robust_acc: %.3f" % test_acc)
        if save_results:
            conf_filename = prefix + "results"+ result_filename
            np.save(os.path.join(args.data_dir, conf_filename), data_matrix)
            act_filename = prefix + "actoverall"+ result_filename
            np.save(os.path.join(args.data_dir, act_filename), act_matrix)
            txtpath = os.path.join(os.path.dirname(args.data_dir), "robustacc.txt")
            with open(txtpath, "a") as f:
                f.write(f"{args.mid} {test_acc:.3f}\n")


if __name__ == "__main__":
    args = argparser()
    print(args)
    # plot_rotated_imgs(args, outdir="/Users/z.liao/oxfordXAI/repo/XAffine/plots/1515/ori", test_angles=[-15, 15])
    if args.aug_type == "r":
        test_intervals=[-15, 15]
        result_filename = "1515.npy"
    else:
        test_intervals=[0.7, 1.3]
        result_filename = f"0515{args.aug_type}.npy"
    if args.modelname ==  "vgg13bn":
        layers=["8", "9"]
    else:
        layers=["0", "1"]
    # for mid in range(1, 8+1):
    #     update_mid(args, mid)
    #     print(args)
    #     robostacc(args, test_intervals=[-15,15], save_results=True, layers=["8", "9"], result_filename="1515.npy")
    robostacc(args, test_intervals, save_results=True, layers=layers, result_filename=result_filename)
