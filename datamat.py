# functions that load the result_filename.npy produced by save_invariance_results.py
#    (1) test_results1515.npy     -- CONF
#    (2) test_actoverall1515.npy  -- CONV (the last two layers in this work)

import os
import numpy as np
from utils import get_even_array
from sampler import get_corr_idx, equal_split_func
import warnings
warnings.filterwarnings("ignore")


def load_confidence_maps(args, class_id=None, test_intervals=None):
    data = np.load(args.data_path, mmap_mode="r")
    if test_intervals is not None:
        idx_start = np.where(data[:, 0, -1] == test_intervals[0])[0][0]
        idx_end = np.where(data[:, 0, -1] == test_intervals[-1])[0][0]
        data = data[idx_start: idx_end+1]
    if args.correct_split:
        incorrect_idx, correct_idx, test_intervals = get_corr_incorr_split(args, data=data, class_id=class_id)
        incorr_confidence = data[:, incorrect_idx, 3]
        corr_confidence = data[:, correct_idx, 3]
        return incorr_confidence, corr_confidence, test_intervals
    else:
        test_intervals = get_corr_incorr_split(args, data=data, class_id=class_id)
        return data[:, :, 3], test_intervals

def get_corr_incorr_split(args, data=None, class_id=None):
    # columns = ["idx", "label", "prediction", "confidence", "angle"]
    if data is None:
        data = np.load(args.data_path, mmap_mode="r")
    if class_id is not None:
        # class_idx = np.where(data[0, :, 1] == class_id)[0]
        data = data[:, data[0, :, 1] == class_id]
    if args.r < 1:
        if args.seed is not None:
            np.random.seed(args.seed)
        nb_cand = int(data.shape[1] * args.r)
        cand = np.random.choice(range(data.shape[1]), nb_cand, replace=False)
        data = data[:, cand]
    test_intervals = np.sort(np.unique(data[:,:,-1]))
    if args.correct_split:
        idx0 = np.where(test_intervals==args.intv_centre)[0][0]
        incorrect_idx = np.where(data[idx0,:,1] != data[idx0,:,2])[0]
        correct_idx = np.where(data[idx0,:,1] == data[idx0,:,2])[0]
        if args.equal_split:
            nb_chosen_example = len(incorrect_idx)
            if args.seed is not None:
                np.random.seed(args.seed)
            correct_idx = np.random.choice(correct_idx, nb_chosen_example, replace=False)
        if args.r < 1:
            incorrect_idx, correct_idx = cand[incorrect_idx], cand[correct_idx]
        return incorrect_idx, correct_idx, test_intervals
    else:
        return test_intervals

# return ["f_idx", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std"]
def load_actoverall(args, class_id=None, test_intervals=None):
    # act_columns = ["idx", "label", "prediction", "f_idx", "mean", "std", "max", "min", "mean_max", "mean_std", "max_mean", "max_std", "angle"]
    # act_matrix = np.zeros([len(test_intervals), nb_examples, len(layers), len(act_columns)])
    data = np.load(args.data_path, mmap_mode="r")
    if class_id is not None:
        # class_idx = np.where(data[0, :, 1] == class_id)[0]
        data = data[:, data[0, :, 0, 1] == class_id]
    if args.r < 1:
        if args.seed is not None:
            np.random.seed(args.seed)
        nb_cand = int(data.shape[1] * args.r)
        cand = np.random.choice(range(data.shape[1]), nb_cand, replace=False)
        data = data[:, cand]
    if test_intervals is not None:
        idx_start = np.where(data[:, 0, 0, -1] == test_intervals[0])[0][0]
        idx_end = np.where(data[:, 0, 0, -1] == test_intervals[-1])[0][0]
        data = data[idx_start: idx_end+1]
    test_intervals = np.sort(np.unique(data[:,:,:,-1]))
    if args.correct_split:
        try:
            idx0 = np.where(test_intervals==args.intv_centre)[0][0]
        except:
            idx0 = np.where(get_even_array(a_min=-15, a_max=15, nb_elements=30+1)==0)[0][0]
        incorrect_idx = np.where(data[idx0,:,0,1] != data[idx0,:,0,2])[0]
        correct_idx = np.where(data[idx0,:,0,1] == data[idx0,:,0,2])[0]
        if args.equal_split:
            nb_chosen_example = len(incorrect_idx)
            if args.seed is not None:
                np.random.seed(args.seed)
            correct_idx = np.random.choice(correct_idx, nb_chosen_example, replace=False)
        else:
            nb_chosen_example = len(correct_idx)
        incorr_confidence = data[:, incorrect_idx, :, 3:-1]
        corr_confidence = data[:, correct_idx, :, 3:-1]
        return incorr_confidence, corr_confidence, test_intervals
    else:
        return data[:, :, :, 3:-1], test_intervals

def get_metrics(mat):
    mat = mat.reshape(mat.shape[0], -1)
    m = np.mean(mat, axis=1)
    std = np.std(mat, axis=1)
    return m, std
