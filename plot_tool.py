import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datamat import load_confidence_maps, load_actoverall
from utils import get_relations, get_asymmetry, get_continuity
from measurement import img_grad_func

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def argparser():
    parser = argparse.ArgumentParser()
    """
    args.data_dir: the root directory where all the .npy files are stored
    "data_dir/mid/xxx.npy"
    """
    parser.add_argument("--data_dir", type=str, default="../plots")
    parser.add_argument("--mid", type=str, default="-1") # model id、
    parser.add_argument("--aug_type", type=str, default="r")

    # Default value(s) / filename(s)
    parser.add_argument("--plot_foldername", type=str, default="1515")
    parser.add_argument("--data_filename", type=str, default="") # Original results
    parser.add_argument("--actdata_filename", type=str, default="") # Original results
    parser.add_argument("--nb_intv", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--r", type=float, default=1.0) # ratio of data used
    parser.add_argument("--equal_split", type=bool, default=False)
    parser.add_argument("--flip_y", type=bool, default=True)
    parser.add_argument("--correct_split", type=bool, default=True)
    parser.add_argument("--l2_norm", type=bool, default=True)
    parser.add_argument("--colormap", type=bool, default=True) # whether to fix the colour map for all model
    parser.add_argument("--vmax", type=float, default=0.15) # max value of the colour map

    args = parser.parse_args()
    set_default_filename(args)
    args.data_path = os.path.join(args.data_dir, str(args.mid), args.data_filename)
    args.plot_dir = os.path.join(args.data_dir, str(args.mid), args.plot_foldername)
    return args


def set_default_filename(args):
    if args.actdata_filename == "":
        if args.aug_type == "r":
            args.actdata_filename = "test_actoverall1515.npy"
        else:
            args.actdata_filename = f"test_actoverall0515{args.aug_type}.npy"
    if args.data_filename == "":
        if args.aug_type == "r":
            args.data_filename = "test_results1515.npy"
        else:
            args.data_filename = f"test_results0515{args.aug_type}.npy"
    if "t" in args.mid:
        args.data_dir = args.data_dir+"_t"


def update_datapath(args, data_filename):
    # args.data_dir = args.data_dir+"_t" if "t" in args.mid else args.data_di
    args.data_path = os.path.join(args.data_dir, str(args.mid), data_filename)


def plot_figs4models(args, test_intervals, class_id=None):
    update_datapath(args, args.data_filename)
    if args.correct_split:
        incorr_confidence, _confidence, test_intervals = load_confidence_maps(args, class_id, test_intervals)
    else:
        _confidence, test_intervals = load_confidence_maps(args, class_id, test_intervals)
    nb_aug = len(test_intervals)
    nb_maps = 5
    l2_dists = np.zeros([nb_maps, nb_aug, nb_aug])
    _, _, l2_dist = get_relations([_confidence], l2_norm=args.l2_norm, flip_y=args.flip_y)
    l2_dists[0] = l2_dist[0]
    update_datapath(args, args.actdata_filename)
    incorr_act, corr_act, _ = load_actoverall(args, class_id=None, test_intervals=test_intervals)
    overall_stats = ["f_idx", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std"]
    fig_idx = 1
    for f_idx in range(incorr_act.shape[2]):
        fn = f"conv{int(incorr_act[0, 0, f_idx, 0])}"
        # for s_idx in range(1, len(overall_stats)):
        for s_idx in [3, 5]:
            sn = overall_stats[s_idx]
            _, _, l2_dist_act = get_relations([corr_act[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
            l2_dists[fig_idx] = l2_dist_act[0]
            fig_idx += 1
    forplots = l2_dists.copy()
    ctny_all = get_continuity(l2_dists, args.flip_y)
    asymm_all = get_asymmetry(l2_dists, args.flip_y)
    means = np.mean(l2_dists.reshape(5, -1), axis=-1)
    s_means = np.mean(l2_dists.reshape(5, -1)**2, axis=-1)
    grad_directs = ["v", "h", "d"]
    grad_overall = 0
    for gd in grad_directs:
        grad, grad_sc = img_grad_func(l2_dists, gd, args.flip_y)
        grad_overall += grad_sc[:, -1]/len(grad_directs)
    print(f"Square Mean @ CONF: {s_means[0]}")
    print(f"Gradient Score @ CONV-MAX-1: {grad_overall[1]}")
    print(f"Discontinuity @ CONV-MAX-2: {ctny_all[3]}")
    print(f"Asymmetry @ CONV-MEAN-1: {asymm_all[2]}")
    if args.colormap:
        im = plt.imshow(forplots.reshape(-1, 31), vmin=0.0, vmax=args.vmax)
    else:
        im = plt.imshow(forplots.reshape(-1, 31))
    plt.colorbar(im, fraction=0.02, pad=0.07, orientation='horizontal')
    plt.show()


if __name__ == "__main__":
    args = argparser()
    print(args)
    if args.aug_type == "r":
        test_intervals = [-15, 15]
        args.intv_centre = 0
    else:
        test_intervals = [0.7, 1.3]
        args.intv_centre = 1
    plot_figs4models(args, test_intervals)
    
