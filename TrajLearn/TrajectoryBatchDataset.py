import os
import json
import random
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset


class TrajectoryBatchDataset(IterableDataset):
    def __init__(self, dataset_directory, type='train', delimiter=' ', validation_ratio=0.1, test_ratio=0.2):
        self.dataset_directory = dataset_directory

        full_data = [np.array([int(j) for j in i.split(delimiter)]) for i in pd.read_csv(
            os.path.join(dataset_directory, 'data.txt'), header=None)[0]]

        self.number_of_trajectories = len(full_data)
        self.vocab_size = sum(1 for _ in open(
            os.path.join(dataset_directory, 'vocab.txt')))

        if type == 'train':
            self.data = full_data[:-int(self.number_of_trajectories *
                                        (validation_ratio + test_ratio))]
        elif type == 'val':
            self.data = full_data[-int(self.number_of_trajectories * (
                validation_ratio + test_ratio)): -int(self.number_of_trajectories * test_ratio)]
        elif type == 'test':
            self.data = full_data[-int(self.number_of_trajectories * test_ratio):]
        else:
            raise ValueError('Invalid type')

        self.dataX = []
        self.dataY = []
        self.type = type

    def create_batches(self, batch_size, observe, predict=1, shift=True, shuffle=True, drop_last=False):

        if isinstance(observe, int):
            observe = [observe]
        self.observe = observe
        if isinstance(predict, int):
            predict = [predict] * len(observe)
        self.predict = predict
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        for trajectory in self.data:
            for j in range(len(self.observe)):
                for i in range(0, len(trajectory) - self.observe[j] - self.predict[j] + 1):
                    self.dataX.append(trajectory[i:i+self.observe[j]])
                    if self.type == 'test' or not self.shift:
                        self.dataY.append(
                            trajectory[i+self.observe[j]:i+self.observe[j]+self.predict[j]])
                    else:
                        self.dataY.append(
                            trajectory[i+self.predict[j]:i+self.observe[j]+self.predict[j]])

        # Group indices of same size together
        size_to_indices = defaultdict(list)
        for i in range(len(self.dataX)):
            size_to_indices[len(self.dataX[i])].append(i)

        # Prepare the list of batches and shuffle it
        batches = []
        for size_indices in size_to_indices.values():
            for i in range(0, len(size_indices), self.batch_size):
                batch = size_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        self.batches = batches

    def get_neighbors(self):
        with open(os.path.join(self.dataset_directory, 'neighbors.json'), 'r') as neighbors_file:
            neighbors = json.load(neighbors_file)
            neighbors = {int(k): v + [0] for k, v in neighbors.items()}
            neighbors[0] = []
        return neighbors

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch_indices = self.batches[index]
        return torch.LongTensor(np.stack([self.dataX[i] for i in batch_indices])), torch.LongTensor(np.stack([self.dataY[i] for i in batch_indices]))

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()

        # if worker_info is None:
        #     batches = self.batches
        # else:
        #     n_workers = worker_info.num_workers
        #     n_data = len(self.batches)
        #     chunk_size = n_data // n_workers

        #     chunk_start = chunk_size * worker_info.id
        #     batches = self.batches[chunk_start: chunk_start + chunk_size]

        for batch_indices in self.batches:
            yield torch.LongTensor(np.stack([self.dataX[i] for i in batch_indices])), torch.LongTensor(np.stack([self.dataY[i] for i in batch_indices]))
