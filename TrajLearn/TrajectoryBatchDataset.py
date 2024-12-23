import os
import json
import random
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset


class TrajectoryBatchDataset(IterableDataset):
    def __init__(self, dataset_directory, dataset_type='train', delimiter=' ', validation_ratio=0.1, test_ratio=0.2):
        self.dataset_directory = dataset_directory

        full_data = [np.array([int(j) for j in i.strip().split(delimiter)]) for i in pd.read_csv(
            os.path.join(dataset_directory, 'data.txt'), header=None)[0]]

        self.number_of_trajectories = len(full_data)
        self.vocab_size = sum(1 for _ in open(
            os.path.join(dataset_directory, 'vocab.txt'), encoding='utf-8'))

        if dataset_type == 'train':
            self.data = full_data[:-int(self.number_of_trajectories *
                                        (validation_ratio + test_ratio))]
        elif dataset_type == 'val':
            self.data = full_data[-int(self.number_of_trajectories * (
                validation_ratio + test_ratio)): -int(self.number_of_trajectories * test_ratio)]
        elif dataset_type == 'test':
            self.data = full_data[-int(self.number_of_trajectories * test_ratio):]
        else:
            raise ValueError('Invalid type')

        self.dataX = []
        self.dataY = []
        self.batches = []
        self.dataset_type = dataset_type

    def create_batches(self, batch_size, observe, predict=1, shuffle=True, drop_last=False):

        if isinstance(observe, int):
            observe = [observe]
        if isinstance(predict, int):
            predict = [predict] * len(observe)

        for trajectory in self.data:
            for j, observe_length in enumerate(observe):
                for i in range(0, len(trajectory) - observe_length - predict[j] + 1):
                    self.dataX.append(trajectory[i:i+observe_length])
                    self.dataY.append(
                            trajectory[i+observe_length:i+observe_length+predict[j]])

        # Group indices of same size together
        size_to_indices = defaultdict(list)
        for i, x in enumerate(self.dataX):
            size_to_indices[len(x)].append(i)

        # Prepare the list of batches and shuffle it
        batches = []
        for size_indices in size_to_indices.values():
            for i in range(0, len(size_indices), batch_size):
                batch = size_indices[i:i+batch_size]
                if len(batch) == batch_size or not drop_last:
                    batches.append(batch)

        if shuffle:
            random.shuffle(batches)

        self.batches = batches

    def get_neighbors(self):
        with open(os.path.join(self.dataset_directory, 'neighbors.json'), encoding='utf-8') as neighbors_file:
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
        # for i in range(len(self.dataX)):
        #     yield torch.LongTensor(self.dataX[i]), torch.LongTensor(self.dataY[i])

        for batch_indices in self.batches:
            yield torch.LongTensor(np.stack([self.dataX[i] for i in batch_indices])), torch.LongTensor(np.stack([self.dataY[i] for i in batch_indices]))
