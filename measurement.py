# To generate a json file (stats1515_r.json) for each model at json_path
# Input: (1) load the .npy file (test_results1515.npy) generated by save_invariance_results.py
#        (2) load the .npy file (test_actoverall1515.npy) generated by save_invariance_results.py
# Output: a json file consisting all measurements for that model


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import cv2
from utils import verify_paths, get_relations, merge_relations
from utils import get_asymmetry, get_continuity, log_asymm_ctny, fillin_relation_diagonal
from json_stat import load_json, initialise_json, dump_json
from datamat import load_confidence_maps, load_actoverall
import json

import warnings
warnings.filterwarnings("ignore")
# metric_names = ["corrcoef", "cos_dist", "l2_dist"]
metric_names = ["l2_dist"]

def argparser():
    parser = argparse.ArgumentParser()
    # You must specify:
    # The test_actoverall1515.npy file should be at data_dir/mid/test_actoverall1515.npy
    parser.add_argument("--data_dir", type=str, default="../plots")
    parser.add_argument("--mid", type=str, default="-1") # model id
    # "r": rotation, "b": brightness, "s": scaling
    parser.add_argument("--aug_type", type=str, default="r")



    # Default value(s) / filename(s)
    # if you have specify other filename generated by save_invariance_results.py, please specify the generated filename here.
    parser.add_argument("--data_filename", type=str, default="") # test_results1515.npy
    parser.add_argument("--actdata_filename", type=str, default="") # test_actoverall1515.npy
    # if you have generated .npy files for other testing interval, rather than [-15, 15], please specify here.
    parser.add_argument("--plot_foldername", type=str, default="1515")
    # If we split the data into correcly classified and misclassified examples
    parser.add_argument("--correct_split", type=bool, default=True)
    # If correct_split, whether we make the two splits the same number of data objects.
    parser.add_argument("--equal_split", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--flip_y", type=bool, default=True)
    parser.add_argument("--plot_detail", type=bool, default=True)
    parser.add_argument("--l2_norm", type=bool, default=True)
    parser.add_argument("--r", type=float, default=1.0) # ratio of data used
    parser.add_argument("--dbname", type=str, default="cifar")
    # Testing suite is either testing set or training set of the original dataset (CIFAR or MNIST)
    # In this work we only consider "testing set" as the testing suite.
    parser.add_argument("--train", type=bool, default=False)


    args = parser.parse_args()
    set_default_filename(args)
    args.data_path = os.path.join(args.data_dir, args.mid, args.data_filename)
    args.plot_dir = os.path.join(args.data_dir, args.mid, args.plot_foldername)
    args.json_path = os.path.join(args.plot_dir, f"stats1515_{args.aug_type}.json")
    args.modellabel_path = os.path.join(args.data_dir, "model_label.txt") # used by json_stat.py
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
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


def update_mid(args, mid):
    args.mid = mid
    data_dir = args.data_dir+"_t" if "t" in args.mid else args.data_dir
    args.data_path = os.path.join(data_dir, args.mid, args.data_filename)
    args.plot_dir = os.path.join(data_dir, args.mid, args.plot_foldername)
    args.json_path = os.path.join(args.plot_dir, f"stats1515_{args.aug_type}.json")
    args.modellabel_path = os.path.join(data_dir, "model_label.txt") # used by json_stat.py


def update_datapath(args, data_filename):
    data_dir = args.data_dir+"_t" if "t" in args.mid else args.data_dir
    args.data_path = os.path.join(data_dir, str(args.mid), data_filename)


def sobel_filter(img, ksize=3, flip_y=True):
    img = fillin_relation_diagonal(img, flip_y=flip_y)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=ksize)
    # sobelx = cv2.Scharr(img,cv2.CV_64F,1,0)
    # sobely = cv2.Scharr(img,cv2.CV_64F,0,1)
    return np.sqrt(sobelx**2+sobely**2)
    # return sobelx


# To obtain the gradient of the direction of angle-diff. j=i+r where r is the angle-diff.
# direction: "h" horizontal, "v" vertical, "d" diagonal
def angle_difference_grad(img, direction, flip_y=True):
    img = fillin_relation_diagonal(img, flip_y=flip_y)
    dim = img.shape[0]
    idx0 = int(dim/2)
    nb_valid_grads = (dim-1)*dim/2
    scores = []
    if direction == "h":
        invalid_columns = list(range(idx0, idx0+2))
        h_img = np.roll(img, 1, axis=1)
        h_grad = np.subtract(h_img, img)
        h_grad[:, 0] = h_grad[:, 1]
        # h_grad_mask = np.ones(h_grad.shape)
        # h_grad_mask[:, 0] = 0
        valid_grads, w_std = [], []
        for i in range(dim):
            if i == 0 or i in invalid_columns:
                continue
            # h_grad_mask[i, -i:] = 0
            valid_grads += list(h_grad[:-i, i])
            w_std.append(np.std(h_grad[:-i, i]))
        # valid_grads_mat = np.multiply(h_grad, h_grad_mask)
        # assert np.sum(h_grad_mask) == nb_valid_grads-len(invalid_columns)*dim+sum(invalid_columns)
        assert len(valid_grads) == nb_valid_grads-len(invalid_columns)*dim+sum(invalid_columns), "Check horizontal gradient"
        scores = [np.mean(valid_grads), np.std(valid_grads), np.mean(w_std[:-1])] # The last column has only one item
        scores.append(scores[0]/scores[2])
        return h_grad, scores
    else:
        invalid_rows = list(range(idx0, idx0+2))
        v_img = np.roll(img, 1, axis=0)
        v_grad = np.subtract(v_img, img)
        v_grad[0] = v_grad[1]
        valid_grads, w_std = [], []
        # v_grad_mask = np.zeros(v_grad.shape)
        for i in range(dim):
            if i == 0 or i in invalid_rows:
                continue
            # v_grad_mask[i, :-i] = 1
            valid_grads += list(v_grad[i, :-i])
            w_std.append(np.std(v_grad[i, :-i]))
        # valid_grads_mat = np.multiply(v_grad, v_grad_mask)
        # assert np.sum(v_grad_mask) == nb_valid_grads-len(invalid_rows)*dim+sum(invalid_rows)
        assert len(valid_grads) == nb_valid_grads-len(invalid_rows)*dim+sum(invalid_rows), "Check vertical gradient"
        if direction == "d":
            d_img = np.roll(v_img, 1, axis=1)
            d_img[:, 0] = img[:, 1]
            d_grad = np.subtract(d_img, img)
            d_grad[:, 0] = d_grad[:, 1]
            d_grad[0] = d_grad[1]
            valid_grads, w_std = [], []
            # d_grad_mask = np.zeros(d_grad.shape)
            for i in range(dim):
                if i == 0 or i == idx0:
                    continue
                # d_grad_mask[i, 1:-i] = 1
                valid_grads += list(v_grad[i, 1:-i])
                w_std.append(np.std(v_grad[i, 1:-i]))
            # valid_grads_mat = np.multiply(d_grad, d_grad_mask)
            # assert np.sum(d_grad_mask) == nb_valid_grads-(dim-idx0-1)-(dim-1)
            assert len(valid_grads) == nb_valid_grads-(dim-idx0-1)-(dim-1), "Check diagonal gradient"
            scores = [np.mean(valid_grads), np.std(valid_grads), np.mean(w_std[:-1])] # The last column is nan and the penultimate one has only one item 
            scores.append(scores[0]/scores[1])
            return d_grad, scores
        scores = [np.mean(valid_grads), np.std(valid_grads), np.mean(w_std[:-1])] # The last column has only one item
        scores.append(scores[0]/scores[2])
        return v_grad, scores


def img_grad_func(imgs, grad_direct="v", flip_y=True):
    nb_metrics = imgs.shape[0]
    grads = np.zeros(imgs.shape)
    grad_sc = np.zeros([nb_metrics, 4]) # 3: mean, std, wstd, overall_estimate
    for i, img in enumerate(imgs):
        # grads[i] = sobel_filter(img, ksize=3, flip_y=flip_y)
        grads[i], grad_sc[i] = angle_difference_grad(img, grad_direct, flip_y=flip_y)
    return grads, grad_sc


def plot_confidence_gradient(args, class_id=None, test_intervals=None):
    # columns = ["idx", "label", "prediction", "confidence", "angle"]
    verify_paths(args)
    class_idx = "overall" if class_id is None else str(class_id)
    jstr = load_json(args.json_path)
    jstr["test_info"][class_idx]["confidence"] = {}
    jdict = jstr["test_info"][class_idx]["confidence"]
    if args.correct_split:
        incorr_confidence, corr_confidence, test_intervals = load_confidence_maps(args, class_id, test_intervals)
        nb_intv = len(test_intervals)
        idx0 = np.where(test_intervals==args.intv_centre)[0][0]
        corrcoefs, cos_dists, l2_dists = get_relations([incorr_confidence, corr_confidence], l2_norm=args.l2_norm, flip_y=args.flip_y)
        ctny_all, asymm_all = get_continuity(l2_dists, args.flip_y), get_asymmetry(l2_dists, args.flip_y)
        grad_directs = ["v", "h", "d"]
        for mn in metric_names:
            jdict[mn] = {"sstv": collections.defaultdict(list)}
        # sensitivity
        args.r = 0.9
        incorr_confidence_r, corr_confidence_r, _ = load_confidence_maps(args, class_id, test_intervals)
        corrcoefs_r, cos_dists_r, l2_dists_r = get_relations([incorr_confidence_r, corr_confidence_r], l2_norm=args.l2_norm, flip_y=args.flip_y)
        # for m_idx, comat in enumerate(zip([corrcoefs, cos_dists, l2_dists], [corrcoefs_r, cos_dists_r, l2_dists_r])):
        for m_idx, comat in enumerate(zip([l2_dists], [l2_dists_r])):
            mn = metric_names[m_idx]
            sendiff_incorr, sendiff_corr = np.sum(np.abs(np.subtract(comat[0], comat[1])).reshape(2, -1), axis=1)
            jdict[mn]["sstv"]["incorr"].append(sendiff_incorr)
            jdict[mn]["sstv"]["corr"].append(sendiff_corr)
        args.r = 1.0

        # for m_idx, full_metrics in enumerate([corrcoefs, cos_dists, l2_dists]):
        for m_idx, full_metrics in enumerate([l2_dists]):
            mn = metric_names[m_idx]
            property_dict = {"mean": collections.defaultdict(list),
                             "s_mean": collections.defaultdict(list),
                             "std": collections.defaultdict(list),
                             "robusterror": collections.defaultdict(list),
                             "discnty": collections.defaultdict(list),
                             "asymm": collections.defaultdict(list),
                             "grad_overall": collections.defaultdict(list)}
            gd_dict = {"v": {}, "h": {}, "d": {}}
            for gdk in gd_dict:
                gd_dict[gdk]["mean"] = collections.defaultdict(list)
                gd_dict[gdk]["std"] = collections.defaultdict(list)
                if gdk is not "d":
                    gd_dict[gdk]["wstd"] = collections.defaultdict(list)
            for aid in range(idx0-4):
                if aid == 0:
                    metrics = full_metrics
                else:
                    metrics = full_metrics[:, aid:-aid, aid:-aid]
                grad_scs = {}
                grad_overall = np.zeros(2)
                for gd in grad_directs:
                    grad, grad_sc = img_grad_func(metrics, gd, args.flip_y)
                    grad_overall += grad_sc[:, -1]/len(grad_directs)
                    grad_scs[gd] = grad_sc
                    gd_dict[gd]["mean"]["incorr"].append(grad_sc[0,0])
                    gd_dict[gd]["mean"]["corr"].append(grad_sc[1,0])
                    gd_dict[gd]["std"]["incorr"].append(grad_sc[0,1])
                    gd_dict[gd]["std"]["corr"].append(grad_sc[1,1])
                    if gd is not "d":
                        gd_dict[gd]["wstd"]["incorr"].append(grad_sc[0,2])
                        gd_dict[gd]["wstd"]["corr"].append(grad_sc[1,2])
                ctny, asymm = get_continuity(metrics, args.flip_y), get_asymmetry(metrics, args.flip_y)
                pv_dict = {"mean": np.mean(metrics.reshape(2, -1), axis=1),
                           "s_mean": np.mean(metrics.reshape(2, -1)**2, axis=1),
                           "std": np.std(metrics.reshape(2, -1), axis=1),
                           "robusterror": np.mean(metrics.reshape(2, -1)>0.15, axis=1), # "robusterror": np.sum(metrics[:, idx0-aid], axis=1),
                           "discnty": ctny,
                           "asymm": asymm,
                           "grad_overall": grad_overall}
                for p in property_dict:
                    property_dict[p]["incorr"].append(pv_dict[p][0])
                    property_dict[p]["corr"].append(pv_dict[p][1])
                # print(f"Finished processing aid {aid}")
            for p in property_dict:
                jdict[mn][p] = property_dict[p]
            jdict[mn]["grad"] = gd_dict
        dump_json(jstr, args.json_path)
    else:
        data, test_intervals = load_confidence_maps(args, class_id, test_intervals)
        nb_intv = len(test_intervals)
        idx0 = np.where(test_intervals==args.intv_centre)[0][0]
        corrcoef, cos_dist, l2_dist = get_relations([data], l2_norm=args.l2_norm, flip_y=args.flip_y)        
        ctny_all, asymm_all = get_continuity(l2_dists, args.flip_y), get_asymmetry(l2_dists, args.flip_y)
        grad_directs = ["v", "h", "d"]
        for mn in metric_names:
            jdict[mn] = {"sstv": collections.defaultdict(list)}
        # sensitivity
        args.r = 0.9
        data_r, test_intervals = load_confidence_maps(args, class_id, test_intervals)
        corrcoefs_r, cos_dists_r, l2_dists_r = get_relations([data_r], l2_norm=args.l2_norm, flip_y=args.flip_y)
        # for m_idx, comat in enumerate(zip([corrcoefs, cos_dists, l2_dists], [corrcoefs_r, cos_dists_r, l2_dists_r])):
        for m_idx, comat in enumerate(zip([l2_dists], [l2_dists_r])):
            mn = metric_names[m_idx]
            sendiff_all = np.sum(np.abs(np.subtract(comat[0], comat[1])).reshape(2, -1), axis=1)
            jdict[mn]["sstv"]["all"].append(sendiff_all)
        args.r = 1.0

        # for m_idx, full_metrics in enumerate([corrcoefs, cos_dists, l2_dists]):
        for m_idx, full_metrics in enumerate([l2_dists]):
            mn = metric_names[m_idx]
            property_dict = {"mean": collections.defaultdict(list),
                             "s_mean": collections.defaultdict(list),
                             "std": collections.defaultdict(list),
                             "robusterror": collections.defaultdict(list),
                             "discnty": collections.defaultdict(list),
                             "asymm": collections.defaultdict(list),
                             "grad_overall": collections.defaultdict(list)}
            gd_dict = {"v": {}, "h": {}, "d": {}}
            for gdk in gd_dict:
                gd_dict[gdk]["mean"] = collections.defaultdict(list)
                gd_dict[gdk]["std"] = collections.defaultdict(list)
                if gdk is not "d":
                    gd_dict[gdk]["wstd"] = collections.defaultdict(list)
            for aid in range(idx0-4):
                if aid == 0:
                    metrics = full_metrics
                else:
                    metrics = full_metrics[:, aid:-aid, aid:-aid]
                grad_scs = {}
                grad_overall = np.zeros(2)
                for gd in grad_directs:
                    grad, grad_sc = img_grad_func(metrics, gd, args.flip_y)
                    grad_overall += grad_sc[:, -1]/len(grad_directs)
                    grad_scs[gd] = grad_sc
                    gd_dict[gd]["mean"]["all"].append(grad_sc[0,0])
                    gd_dict[gd]["std"]["all"].append(grad_sc[0,1])
                    if gd is not "d":
                        gd_dict[gd]["wstd"]["all"].append(grad_sc[0,2])
                ctny, asymm = get_continuity(metrics, args.flip_y), get_asymmetry(metrics, args.flip_y)
                pv_dict = {"mean": np.mean(metrics.reshape(2, -1), axis=1),
                           "s_mean": np.mean(metrics.reshape(2, -1)**2, axis=1),
                           "std": np.std(metrics.reshape(2, -1), axis=1),
                           "robusterror": np.mean(metrics.reshape(2, -1)>0.15, axis=1), # "robusterror": np.sum(metrics[:, idx0-aid], axis=1),
                           "discnty": ctny,
                           "asymm": asymm,
                           "grad_overall": grad_overall}
                for p in property_dict:
                    property_dict[p]["all"].append(pv_dict[p][0])
                # print(f"Finished processing aid {aid}")
            for p in property_dict:
                jdict[mn][p] = property_dict[p]
            jdict[mn]["grad"] = gd_dict
        dump_json(jstr, args.json_path)




def plot_actoverall_gradient(args, class_id=None, test_intervals=None):
    verify_paths(args)
    # different modalities
    overall_stats = ["f_idx", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std"]
    class_idx = "overall" if class_id is None else str(class_id)
    jstr = load_json(args.json_path)
    jdict = jstr["test_info"][class_idx]
    if args.correct_split:
        incorr_confidence, corr_confidence, test_intervals = load_actoverall(args, class_id, test_intervals)
        nb_intv = len(test_intervals)
        idx0 = np.where(test_intervals==args.intv_centre)[0][0]
        for f_idx in range(incorr_confidence.shape[2]):
            fn = f"conv{int(incorr_confidence[0, 0, f_idx, 0])}"
            jdict[fn] = {}
            for s_idx in range(1, len(overall_stats)):
                sn = overall_stats[s_idx]
                jdict[fn][sn] = {}
                for mn in metric_names:
                    jdict[fn][sn][mn] = {"sstv": collections.defaultdict(list)}
        for f_idx in range(incorr_confidence.shape[2]):
            fn = f"conv{int(incorr_confidence[0, 0, f_idx, 0])}"
            for s_idx in range(1, len(overall_stats)):
                sn = overall_stats[s_idx]
                corrcoefs, cos_dists, l2_dists = get_relations([incorr_confidence[:, :, f_idx, s_idx], corr_confidence[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
                ctny_all, asymm_all = get_continuity(l2_dists, args.flip_y), get_asymmetry(l2_dists, args.flip_y)
                # sensitivity
                args.r = 0.9
                incorr_confidence_r, corr_confidence_r, _ = load_actoverall(args, class_id, test_intervals)
                corrcoefs_r, cos_dists_r, l2_dists_r = get_relations([incorr_confidence_r[:, :, f_idx, s_idx], corr_confidence_r[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
                # for m_idx, comat in enumerate(zip([corrcoefs, cos_dists, l2_dists], [corrcoefs_r, cos_dists_r, l2_dists_r])):
                for m_idx, comat in enumerate(zip([l2_dists], [l2_dists_r])):
                    mn = metric_names[m_idx]
                    sendiff_incorr, sendiff_corr = np.sum(np.abs(np.subtract(comat[0], comat[1])).reshape(2, -1), axis=1)
                    jdict[fn][sn][mn]["sstv"]["incorr"].append(sendiff_incorr)
                    jdict[fn][sn][mn]["sstv"]["corr"].append(sendiff_corr)
                args.r = 1.0

        grad_directs = ["v", "h", "d"]
        for f_idx in range(incorr_confidence.shape[2]):
                fn = f"conv{int(incorr_confidence[0, 0, f_idx, 0])}"
                for s_idx in range(1, len(overall_stats)):
                    sn = overall_stats[s_idx]
                    corrcoefs, cos_dists, l2_dists = get_relations([incorr_confidence[:, :, f_idx, s_idx], corr_confidence[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
                    # for m_idx, full_metrics in enumerate([corrcoefs, cos_dists, l2_dists]):
                    for m_idx, full_metrics in enumerate([l2_dists]):
                        mn = metric_names[m_idx]
                        property_dict = {"mean": collections.defaultdict(list),
                                         "s_mean": collections.defaultdict(list),
                                         "std": collections.defaultdict(list),
                                         "robusterror": collections.defaultdict(list),
                                         "discnty": collections.defaultdict(list),
                                         "asymm": collections.defaultdict(list),
                                         "grad_overall": collections.defaultdict(list)}
                        gd_dict = {"v": {}, "h": {}, "d": {}}
                        for gdk in gd_dict:
                            gd_dict[gdk]["mean"] = collections.defaultdict(list)
                            gd_dict[gdk]["std"] = collections.defaultdict(list)
                            if gdk is not "d":
                                gd_dict[gdk]["wstd"] = collections.defaultdict(list)
                        for aid in range(idx0-4):
                            if aid == 0:
                                metrics = full_metrics
                            else:
                                metrics = full_metrics[:, aid:-aid, aid:-aid]
                            grad_scs = {}
                            grad_overall = np.zeros(2)
                            for gd in grad_directs:
                                grad, grad_sc = img_grad_func(metrics, gd, args.flip_y)
                                grad_overall += grad_sc[:, -1]/len(grad_directs)
                                grad_scs[gd] = grad_sc
                                gd_dict[gd]["mean"]["incorr"].append(grad_sc[0,0])
                                gd_dict[gd]["mean"]["corr"].append(grad_sc[1,0])
                                gd_dict[gd]["std"]["incorr"].append(grad_sc[0,1])
                                gd_dict[gd]["std"]["corr"].append(grad_sc[1,1])
                                if gd is not "d":
                                    gd_dict[gd]["wstd"]["incorr"].append(grad_sc[0,2])
                                    gd_dict[gd]["wstd"]["corr"].append(grad_sc[1,2])
                            ctny, asymm = get_continuity(metrics, args.flip_y), get_asymmetry(metrics, args.flip_y)
                            pv_dict = {"mean": np.mean(metrics.reshape(2, -1), axis=1),
                                       "s_mean": np.mean(metrics.reshape(2, -1)**2, axis=1),
                                       "std": np.std(metrics.reshape(2, -1), axis=1),
                                       "robusterror": np.mean(metrics.reshape(2, -1)>0.15, axis=1), # "robusterror": np.sum(metrics[:, idx0-aid], axis=1),
                                       "discnty": ctny,
                                       "asymm": asymm,
                                       "grad_overall": grad_overall}
                            for p in property_dict:
                                property_dict[p]["incorr"].append(pv_dict[p][0])
                                property_dict[p]["corr"].append(pv_dict[p][1])
                            # print(f"Finished processing aid {aid}")
                        for p in property_dict:
                            jdict[fn][sn][mn][p] = property_dict[p]
                        jdict[fn][sn][mn]["grad"] = gd_dict
        dump_json(jstr, args.json_path)
    else:
        data, test_intervals = load_actoverall(args, class_id, test_intervals)
        nb_intv = len(test_intervals)
        idx0 = np.where(test_intervals==args.intv_centre)[0][0]
        corrcoef, cos_dist, l2_dist = get_relations([data], l2_norm=args.l2_norm, flip_y=args.flip_y)
        for f_idx in range(data.shape[2]):
            fn = f"conv{int(data[0, 0, f_idx, 0])}"
            jdict[fn] = {}
            for s_idx in range(1, len(overall_stats)):
                sn = overall_stats[s_idx]
                jdict[fn][sn] = {}
                for mn in metric_names:
                    jdict[fn][sn][mn] = {"sstv": collections.defaultdict(list)}
        for f_idx in range(data.shape[2]):
            fn = f"conv{int(data[0, 0, f_idx, 0])}"
            for s_idx in range(1, len(overall_stats)):
                sn = overall_stats[s_idx]
                corrcoefs, cos_dists, l2_dists = get_relations([data[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
                ctny_all, asymm_all = get_continuity(l2_dists, args.flip_y), get_asymmetry(l2_dists, args.flip_y)
                # sensitivity
                args.r = 0.9
                data_r, test_intervals = load_actoverall(args, class_id, test_intervals)
                corrcoefs_r, cos_dists_r, l2_dists_r = get_relations([data_r[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
                # for m_idx, comat in enumerate(zip([corrcoefs, cos_dists, l2_dists], [corrcoefs_r, cos_dists_r, l2_dists_r])):
                for m_idx, comat in enumerate(zip([l2_dists], [l2_dists_r])):
                    mn = metric_names[m_idx]
                    sendiff_all = np.sum(np.abs(np.subtract(comat[0], comat[1])).reshape(2, -1), axis=1)
                    jdict[fn][sn][mn]["sstv"]["all"].append(sendiff_all)
                args.r = 1.0
        grad_directs = ["v", "h", "d"]
        for f_idx in range(data.shape[2]):
                fn = f"conv{int(data[0, 0, f_idx, 0])}"
                for s_idx in range(1, len(overall_stats)):
                    sn = overall_stats[s_idx]
                    corrcoefs, cos_dists, l2_dists = get_relations([data[:, :, f_idx, s_idx]], l2_norm=args.l2_norm, flip_y=args.flip_y)
                    # for m_idx, full_metrics in enumerate([corrcoefs, cos_dists, l2_dists]):
                    for m_idx, full_metrics in enumerate([l2_dists]):
                        mn = metric_names[m_idx]
                        property_dict = {"mean": collections.defaultdict(list),
                                         "s_mean": collections.defaultdict(list),
                                         "std": collections.defaultdict(list),
                                         "robusterror": collections.defaultdict(list),
                                         "discnty": collections.defaultdict(list),
                                         "asymm": collections.defaultdict(list),
                                         "grad_overall": collections.defaultdict(list)}
                        gd_dict = {"v": {}, "h": {}, "d": {}}
                        for gdk in gd_dict:
                            gd_dict[gdk]["mean"] = collections.defaultdict(list)
                            gd_dict[gdk]["std"] = collections.defaultdict(list)
                            if gdk is not "d":
                                gd_dict[gdk]["wstd"] = collections.defaultdict(list)
                        for aid in range(idx0-4):
                            if aid == 0:
                                metrics = full_metrics
                            else:
                                metrics = full_metrics[:, aid:-aid, aid:-aid]
                            grad_scs = {}
                            grad_overall = np.zeros(2)
                            for gd in grad_directs:
                                grad, grad_sc = img_grad_func(metrics, gd, args.flip_y)
                                grad_overall += grad_sc[:, -1]/len(grad_directs)
                                grad_scs[gd] = grad_sc
                                gd_dict[gd]["mean"]["all"].append(grad_sc[0,0])
                                gd_dict[gd]["std"]["all"].append(grad_sc[0,1])
                                if gd is not "d":
                                    gd_dict[gd]["wstd"]["all"].append(grad_sc[0,2])
                            ctny, asymm = get_continuity(metrics, args.flip_y), get_asymmetry(metrics, args.flip_y)
                            pv_dict = {"mean": np.mean(metrics.reshape(2, -1), axis=1),
                                       "s_mean": np.mean(metrics.reshape(2, -1)**2, axis=1),
                                       "std": np.std(metrics.reshape(2, -1), axis=1),
                                       "robusterror": np.mean(metrics.reshape(2, -1)>0.15, axis=1), # "robusterror": np.sum(metrics[:, idx0-aid], axis=1),
                                       "discnty": ctny,
                                       "asymm": asymm,
                                       "grad_overall": grad_overall}
                            for p in property_dict:
                                property_dict[p]["all"].append(pv_dict[p][0])
                        for p in property_dict:
                            jdict[fn][sn][mn][p] = property_dict[p]
                        jdict[fn][sn][mn]["grad"] = gd_dict
        dump_json(jstr, args.json_path)



def write_json(args, test_intervals):
    initialise_json(args, test_intervals) # always do confidence first
    update_datapath(args, args.data_filename)
    plot_confidence_gradient(args, class_id=None, test_intervals=test_intervals)
    update_datapath(args, args.actdata_filename)
    plot_actoverall_gradient(args, class_id=None, test_intervals=test_intervals)


if __name__ == "__main__":
    args = argparser()
    print(args)
    if args.aug_type == "r":
        test_intervals = [-15, 15]
        args.intv_centre = 0
    else:
        test_intervals=[0.7, 1.3]
        args.intv_centre = 1
    
    print(f"processing model {args.mid}")
    write_json(args, test_intervals=test_intervals)

    # for mid in range(1, 51):
    #     update_mid(args, f"t{mid}")
    #     print(f"processing model {args.mid}")
    #     write_json(args, test_intervals=test_intervals)
