# functions that assist training CNN models

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
    def __init__(self, data_source, batch_size, nb_classes=10, shuffle=True):
        self.data_source = data_source
        target_lists = collections.defaultdict(list)
        for i, (data, label) in enumerate(self.data_source):
            target_lists[label].append(i)
        self.target_lists = target_lists
        self.nb_examples = len(data_source)
        self.shuffle = shuffle
        self.cur_class = 0
        self.nb_classes = nb_classes
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        count = 0
        idices = collections.defaultdict(int)
        while count < self.nb_examples:
            batch.append(self.target_lists[self.cur_class][idices[self.cur_class]])
            idices[self.cur_class] += 1
            if idices[self.cur_class] >= len(self.target_lists[self.cur_class]):
                if self.shuffle:
                    np.random.shuffle(self.target_lists[self.cur_class])
                yield batch
                batch = []
                self.cur_class = (self.cur_class + 1) % self.nb_classes
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                self.cur_class = (self.cur_class + 1) % self.nb_classes
            count += 1
    
    def __len__(self):
        return self.nb_examples
