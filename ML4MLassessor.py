# Input: all jason files for all models, and the assessment labels (mlabel.txt and mlabel_t.txt)
# Output:
#      (1) a model database (mdatabase.csv and mdatabase_t.csv), each row of which is all the measurements of one trained model.
#      (2) five trained ML4ML assessors (decision tree, random forest, adaboost, regression tree, linear regression).
#      (3) accuracy on the testing set of the model database for each of the assessor

# Our modeldatabase:
#      partition (a): mid: 1-100, t1-t50. VGG13bn for rotation testing. Test suite: CIFAR10 Testing set.
#      partition (b): mid: 101-200, t101-t150. VGG13bn for brightness testing. Test suite: CIFAR10 Testing set.
#      partition (c): mid: 201-300, t201-t250. VGG13bn for scaling testing. Test suite: CIFAR10 Testing set.
#      partition (d): mid: 301-400, t301-t350. CNN5 for rotation testing. Test suite: MNIST Testing set.
# Mid starting with "t" is a hold-out set. 
# When using three-fold cross validation, the hold-out set is always treated as one fold.
# While the rest "regular" models are randomly split into two folds.


import os
import json
import numpy as np
import pandas as pd
import collections
import argparse
import matplotlib.pyplot as plt
from json_stat import load_json
from sklearn import tree, linear_model
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# In this work, we only consider "invariance"
target_dict = {1: "invariance", 2: "anomaly", 3: "overfit", 4: "adv"}

def argparser():
    parser = argparse.ArgumentParser()
    # The directory where the test_actoverall1515.npy file is
    parser.add_argument("--data_dir", type=str, default="/datadir")
    # The directory where the test_actoverall1515.npy file is (for a hold out set)
    parser.add_argument("--data_dir_t", type=str, default="/datadir_t")
    # if you have generated test_actoverallxxxx.npy for other testing interval, rather than [-15, 15], please specify here.
    parser.add_argument("--plot_foldername", type=str, default="1515")
    # json filename generated by measurement.py
    parser.add_argument("--json_name", type=str, default="stats1515.json")


    # You must specify:
    # database used to train the CNNs
    parser.add_argument("--dbname", type=str, default="cifar")
    # "r": rotation, "b": brightness, "s": scaling
    parser.add_argument("--aug_type", type=str, default="r")


    # Fixed parameters for this work:
    parser.add_argument("--start", type=int, default=1) # model id (starting from -- for training set)
    parser.add_argument("--end", type=int, default=400) # model id (ending with -- for training set)
    parser.add_argument("--dataset", type=bool, default=True) # If prepare dataset from stratch
    parser.add_argument("--dataset_name", type=str, default="mdatabase.csv") # filename of "data.csv"
    parser.add_argument("--label_name", type=str, default="mlabel.txt") # filename of "mlabel.txt"
    args = parser.parse_args()

    args.dataset_path = os.path.join(os.path.dirname(args.data_dir), "mdatabase", args.dataset_name)
    args.label_path = os.path.join(os.path.dirname(args.data_dir), "mdatabase", args.label_name)
    args.dataset_path_t = args.dataset_path.rsplit(".", 1)[0]+"_t.csv"
    args.label_path_t = args.label_path.rsplit(".", 1)[0]+"_t.txt"
    return args


def prepare_dataset(args, start, end, t=False):
    feature_names = ["mid"]
    # intervals = ["1515", "1414", "1313", "1212", "1111", "1010", "99", "88", "77", "66", "55"]
    # sstv_int = ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"]
    # chosen_sn = ['mean', 'std', 'max', 'min', 'mean_max', 'mean_std', 'max_mean','max_std']
    # corrs = ["incorr", "corr"]
    # chosen_classes = ['overall', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    intervals = ["1515"]
    intervals_idx = [0]
    sstv_int = ["0.9"]
    sstv_idx = [0]
    chosen_sn = ['max', 'max_mean']
    corrs = ["all"] # ["corr"]
    chosen_classes = ['overall']
    grad_details = True
    # for confidence, conv8 and conv9, 15 properties are measured on 11 intervals.
    # sstv is measured on 9 intervals, but in this work, we use 0.9 only
    if grad_details:
        f_dim = len(chosen_classes) * ((len(sstv_int) + len(intervals) * 15) + (2 * len(chosen_sn) * (len(sstv_int) + len(intervals) * 15))) * len(corrs)
    else:
        f_dim = len(chosen_classes) * ((len(sstv_int) + len(intervals) * 7) + (2 * len(chosen_sn) * (len(sstv_int) + len(intervals) * 7))) * len(corrs)
    # check json_format. 11 classes -- including "overall".
    # *2: corr and incorr 
    f_data = np.zeros([end-start+1, f_dim+1]) # mid for the first colomn
    t_names = [f"t{i}" for i in range(1, 50+1)] + [f"t{i}" for i in range(101, 150+1)] + [f"t{i}" for i in range(201, 250+1)] + [f"t{i}" for i in range(301, 350+1)]
    for i in range(end-start+1):
        jstr = load_json(os.path.join(args.data_dir, str(start+i), args.plot_foldername, args.json_name))["test_info"]
        jstr = load_json(os.path.join(args.data_dir_t, t_names[i], args.plot_foldername, args.json_name))["test_info"]
        f_data[i, 0] = start + i
        f_cur = 1
        for c in chosen_classes:
            for fn in jstr[c]:
                if fn == "confidence":
                    for pn in jstr[c][fn]["l2_dist"]:
                        if pn == "grad" and grad_details:
                            for gn in jstr[c][fn]["l2_dist"]["grad"]:
                                for gmn in jstr[c][fn]["l2_dist"]["grad"][gn]:
                                    for corr in corrs:
                                        ff = np.array(jstr[c][fn]["l2_dist"]["grad"][gn][gmn][corr])
                                        lenff = len(intervals)
                                        f_data[i, f_cur:f_cur+lenff] = ff[intervals_idx]
                                        f_cur += lenff
                                        if i == 0:
                                            feature_names += [f"{c}_{fn}_{pn}_{gn}_{gmn}_{corr}_{j}" for j in intervals]
                        else:
                            for corr in corrs:
                                if pn == "sstv":
                                    intv = sstv_int
                                    idx = sstv_idx
                                else:
                                    intv = intervals
                                    idx = intervals_idx
                                if i == 0:
                                    feature_names += [f"{c}_{fn}_{pn}_{corr}_{j}" for j in intv]
                                ff = np.array(jstr[c][fn]["l2_dist"][pn][corr])
                                lenff = len(intv)
                                f_data[i, f_cur:f_cur+lenff] = ff[idx]
                                f_cur += lenff
                elif "conv" in fn:
                    for sn in chosen_sn:
                        for pn in jstr[c][fn][sn]["l2_dist"]:
                            if pn == "grad" and grad_details:
                                for gn in jstr[c][fn][sn]["l2_dist"]["grad"]:
                                    for gmn in jstr[c][fn][sn]["l2_dist"]["grad"][gn]:
                                        for corr in corrs:
                                            ff = np.array(jstr[c][fn][sn]["l2_dist"]["grad"][gn][gmn][corr])
                                            lenff = len(intervals)
                                            f_data[i, f_cur:f_cur+lenff] = ff[intervals_idx]
                                            f_cur += lenff
                                            if i == 0:
                                                feature_names += [f"{c}_{fn}_{sn}_{pn}_{gn}_{gmn}_{corr}_{j}" for j in intervals]
                            else:
                                for corr in corrs:
                                    if pn == "sstv":
                                        intv = sstv_int
                                        idx = sstv_idx
                                    else:
                                        intv = intervals
                                        idx = intervals_idx
                                    if i == 0:
                                        feature_names += [f"{c}_{fn}_{sn}_{pn}_{corr}_{j}" for j in intv]
                                    ff = np.array(jstr[c][fn][sn]["l2_dist"][pn][corr])
                                    lenff = len(intv)
                                    f_data[i, f_cur:f_cur+lenff] = ff[idx]
                                    f_cur += lenff

                else:
                    continue
    dataset_path = args.dataset_path_t if t else args.dataset_path
    df = pd.DataFrame(data=f_data, columns=feature_names)
    df.to_csv(dataset_path, index=False)


def eval(data, labels, clf, target_idx):
    logits = clf.predict(data)
    if target_idx == 3:
        pred = np.zeros(logits.shape)
        for i in range(len(logits)):
            if logits[i] >= 0.5:
                pred[i] = 1
            elif -0.5< logits[i] < 0.5:
                pred[i] = 0
            else:
                pred[i] = -1
    else:
        pred = [1 if i >= 0.5 else 0 for i in logits]
    return np.sum(pred==labels)/len(labels)


def dt(args, idices, topk=3, acc_results=[]):
    data = pd.read_csv(args.dataset_path)
    labels = pd.read_csv(args.label_path, sep=" ")
    data_t = pd.read_csv(args.dataset_path_t)
    labels_t = pd.read_csv(args.label_path_t, sep=" ")
    data = data[idices[0]:idices[1]]
    labels = labels[idices[0]:idices[1]]
    data_t = data_t[idices[0]//2:idices[1]//2]
    labels_t = labels_t[idices[0]//2:idices[1]//2]
    nb_trainingset = len(labels)
    indices = np.random.choice(range(nb_trainingset), nb_trainingset, replace=False)
    fold1 = (data.values[indices[:nb_trainingset//2]], labels.values[indices[:nb_trainingset//2]])
    fold2 = (data.values[indices[nb_trainingset//2:]], labels.values[indices[nb_trainingset//2:]])
    fold3 = (data_t.values, labels_t.values)

    feature_names = data.columns.values[1:]
    target_names = ["invariant", "variance"]
    selected_features = []
    dts = []
    regrs = []
    importances = []

    for fidx in range(3):
        if fidx == 0:
            trainingset = (data.values, labels.values)
            testingset = (data_t.values, labels_t.values)
        elif fidx == 1:
            trainingset = (np.concatenate([fold1[0], fold3[0]]), np.concatenate([fold1[1], fold3[1]]))
            testingset = (fold2[0], fold2[1])
        elif fidx == 2:
            trainingset = (np.concatenate([fold2[0], fold3[0]]), np.concatenate([fold2[1], fold3[1]]))
            testingset = (fold1[0], fold1[1])

        # target_dict = {1: "rotation", 2: "brightness", 3: "scaling"}
        if args.aug_type == "r"
            target_idx = 1
        elif args.aug_type == "b":
            target_idx = 2
        elif args.aug_type == "s":
            target_idx = 3


        dt = tree.DecisionTreeClassifier()
        dt = dt.fit(trainingset[0][:, 1:], trainingset[1][:, target_idx].astype(np.int))
        acc = dt.score(trainingset[0][:, 1:], trainingset[1][:, target_idx].astype(np.int))
        acc_t = dt.score(testingset[0][:, 1:], testingset[1][:, target_idx].astype(np.int))
        acc_results[fidx]["dt"][target_dict[target_idx]].append(acc_t)


        rf = RandomForestClassifier()
        rf = rf.fit(trainingset[0][:, 1:], trainingset[1][:, target_idx].astype(np.int))
        acc_rf = rf.score(trainingset[0][:, 1:], trainingset[1][:, target_idx].astype(np.int))
        acc_rf_t = rf.score(testingset[0][:, 1:], testingset[1][:, target_idx].astype(np.int))
        acc_results[fidx]["rf"][target_dict[target_idx]].append(acc_rf_t)


        regr = linear_model.LinearRegression()
        regr = regr.fit(normalize(trainingset[0][:, 1:], axis=1), trainingset[1][:, target_idx])
        acc_regr = eval(normalize(trainingset[0][:, 1:], axis=1), trainingset[1][:, target_idx], regr, target_idx)
        acc_regr_t = eval(normalize(testingset[0][:, 1:], axis=1), testingset[1][:, target_idx], regr, target_idx)
        acc_results[fidx]["reg"][target_dict[target_idx]].append(acc_regr_t)


        # dtr = tree.DecisionTreeRegressor(max_depth=None)
        # dtr = dtr.fit(normalize(trainingset[0][:, 1:], axis=1), trainingset[1][:, target_idx])
        # acc_dtr = eval(normalize(trainingset[0][:, 1:], axis=1), trainingset[1][:, target_idx], dtr, target_idx)
        # acc_dtr_t = eval(normalize(testingset[0][:, 1:], axis=1), testingset[1][:, target_idx], dtr, target_idx)
        # acc_results[fidx]["dtr"][target_dict[target_idx]].append(acc_dtr_t)


        adb = AdaBoostClassifier()
        adb = adb.fit(trainingset[0][:, 1:], trainingset[1][:, target_idx].astype(np.int))
        acc_adb = adb.score(trainingset[0][:, 1:], trainingset[1][:, target_idx].astype(np.int))
        acc_adb_t = adb.score(testingset[0][:, 1:], testingset[1][:, target_idx].astype(np.int))
        acc_results[fidx]["adb"][target_dict[target_idx]].append(acc_adb_t)


if __name__ == "__main__":
    args = argparser()
    print(args)
    if args.aug_type == "r":
        idices = [0, 100]
        if args.dbname == "mnist":
            idices = [300, 400]
    elif args.aug_type == "b":
        idices = [100, 200]
    elif args.aug_type == "s":
        idices = [200, 300]
    if args.dataset:
        prepare_dataset(args, start=args.start, end=args.end, t=False)
        prepare_dataset(args, start=args.start, end=args.end//2, t=True)

    acc_results = [{"dt": collections.defaultdict(list), 
                      "reg": collections.defaultdict(list), 
                      "rf": collections.defaultdict(list), 
                      "dtr": collections.defaultdict(list), 
                      "adb": collections.defaultdict(list)} for i in range(3)]
    for i in range(10):
        dt(args, idices, acc_results=acc_results)
    fold_results = collections.defaultdict(list)
    for clsn in acc_results[0]:
        for task in acc_results[0][clsn]:
            for fold_result in zip(acc_results[0][clsn][task], acc_results[1][clsn][task], acc_results[2][clsn][task]):
                fold_results[clsn].append(np.mean(fold_result))
        acc = np.mean(fold_results[clsn])
        maxx = np.max(fold_results[clsn])
        minn = np.min(fold_results[clsn])
        error = max(np.abs(maxx-acc), np.abs(acc-minn))
        print(f"{task} {clsn} acc: {acc}, max: {maxx}, min: {minn}, err: {error}")
