# Implementation of AnnealedSampler, SquareRootSampler and UniqueBatchSampler.


import random
import numpy as np
from torch.utils.data import Sampler


class AnnealedSampler:
    def __init__(self, dataset_sizes, total_epochs, max_decay_rate=0.8):
        """
        dataset_sizes: list of sizes
        total_epochs: total training epochs
        """
        self.sizes = np.array(dataset_sizes)
        self.total_epochs = total_epochs
        self.max_decay_rate = max_decay_rate
        
    def get_probabilities(self, current_epoch):
        # for epoch starts from 0, α starts at 1 and decays toward (1 - max_decay_rate)
        alpha = 1.0 - self.max_decay_rate * (current_epoch / (self.total_epochs - 1))
        
        # P_i ∝ N_i^α
        probs = np.power(self.sizes, alpha)
        return probs / probs.sum()


class SquareRootSampler:
    def __init__(self, dataset_sizes):
        self.sizes = np.array(dataset_sizes)
        sqrt_sizes = np.sqrt(self.sizes)
        self.probs = sqrt_sizes / sqrt_sizes.sum()

    def get_probabilities(self):
        return self.probs


class UniqueBatchSampler(Sampler):
    """
    Ensure no duplicates in a batch, for MNRL or MNRL with hard negatives.
    When using this sampler, do not set batch_size, shuffle, or sampler in DataLoader.
    """
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        random.shuffle(self.indices)
        batch = []
        seen_texts = set()

        for idx in self.indices:
            # anchor, pos, _ = self.dataset[idx]  # (anchor, pos, hard_neg)
            anchor, pos, *_ = self.dataset[idx]
            
            # check if anchor or positive is already in current batch;
            # skipped idx won't be sampled anymore in the same epoch
            if anchor in seen_texts or pos in seen_texts:
                continue
                
            batch.append(idx)
            seen_texts.add(anchor)
            seen_texts.add(pos)
            
            # if batch is full, yield it and reset
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                seen_texts = set()
        
        # yield remaining samples
        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        # note that the actual len may be less than this
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    