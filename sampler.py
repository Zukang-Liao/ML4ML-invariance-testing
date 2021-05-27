import numpy as np
from torch.utils.data import Sampler, RandomSampler
import collections


class imbalanceSampler(Sampler):
    def __init__(self, data_source, target_class, ratio, batch_size, shuffle):
        self.data_source = data_source
        self.target_class = target_class
        target_list = []
        for i, (data, label) in enumerate(self.data_source):
            if label == self.target_class:
                target_list.append(i)
        delete_list = np.random.choice(target_list, int(len(target_list)*(1-ratio)), replace=False)
        self.chosen_indices = np.delete(np.array(range(len(data_source))), delete_list)
        self.nb_examples = len(self.chosen_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        # return iter(self.chosen_indices)
        batch = []
        for idx in self.chosen_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            if self.shuffle:
                np.random.shuffle(self.chosen_indices)
            yield batch
    
    def __len__(self):
        return self.nb_examples
        return (self.nb_examples + self.batch_size - 1) // self.batch_size


class orderedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        target_lists = collections.defaultdict(list)
        for i, (data, label) in enumerate(self.data_source):
            target_lists[label].append(i)
        self.target_list = []
        for c in data_source.class_to_idx:
            c = data_source.class_to_idx[c]
            self.target_list += target_lists[c]
        self.nb_examples = len(self.target_list)
    
    def __iter__(self):
        return iter(self.target_list)
    
    def __len__(self):
        return self.nb_examples


# Used for generating images from one class only
class mySampler(Sampler):
    def __init__(self, data_source, class_idx):
        self.data_source = data_source
        self.class_idx = class_idx
        nb_examples = 0
        mylist = []
        for i, (data, label) in enumerate(self.data_source):
            if label == self.class_idx:
                nb_examples += 1
                mylist.append(i)
        self.mylist = mylist
        self.nb_examples = nb_examples
    
    def __iter__(self):
        return iter(self.mylist)
    
    def __len__(self):
        return self.nb_examples


def equal_split_func(incorr_idx, corr_idx, seed=None):
    if seed is not None:
        np.random.seed(seed)
    else:
        print("Warning: No seed provided for equal_split")
    if len(incorr_idx) < len(corr_idx):
        corr_idx  = np.random.choice(corr_idx, len(incorr_idx), replace=False)
    elif len(incorr_idx) > len(corr_idx):
        # print(f"Warning: More incorrect examples than correct ones for class {class_idx}")
        incorr_idx  = np.random.choice(incorr_idx, len(corr_idx), replace=False)
    return incorr_idx, corr_idx


def class_split_func(results, incorr_idx, corr_idx, class_idx):
    candidates = np.where(results[0, :, 1] == class_idx)[0]
    incorr_idx = np.intersect1d(candidates, incorr_idx)
    corr_idx = np.intersect1d(candidates, corr_idx)
    return incorr_idx, corr_idx


def get_corr_idx(result_path, equal_split, seed=None, class_idx=None, equal_split_first=False):
    results = np.load(result_path, mmap_mode="r")
    test_angles = np.sort(np.unique(results[:,:,-1]))
    idx0 = np.where(test_angles==0)[0][0]
    # columns = ["idx", "label", "prediction", "confidence", "angle"]
    incorr_idx = np.where(results[idx0,:,1] != results[idx0,:,2])[0]
    corr_idx = np.where(results[idx0,:,1] == results[idx0,:,2])[0]

    if equal_split_first:
        if equal_split:
            incorr_idx, corr_idx = equal_split_func(incorr_idx, corr_idx, seed)
        if class_idx is not None:
            incorr_idx, corr_idx = class_split_func(results, incorr_idx, corr_idx, class_idx)
        if equal_split:
            incorr_idx, corr_idx = equal_split_func(incorr_idx, corr_idx, seed)
    else:
        if class_idx is not None:
            incorr_idx, corr_idx = class_split_func(results, incorr_idx, corr_idx, class_idx)
        if equal_split:
            incorr_idx, corr_idx = equal_split_func(incorr_idx, corr_idx, seed)
    return incorr_idx, corr_idx, test_angles


# Used for generating images from one class only
class incorrectSampler(Sampler):
    def __init__(self, data_source, result_path, correct, equal_split=True, class_idx=None, seed=None):
        self.data_source = data_source
        self.correct = correct
        incorr_idx, corr_idx, test_angles = get_corr_idx(result_path, equal_split, seed, class_idx)
        self.test_angles = test_angles
        self.idx = corr_idx if correct else incorr_idx

    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)