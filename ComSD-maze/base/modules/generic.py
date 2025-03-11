# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
import torch.nn as nn


class OneHotEmbedding(nn.Module):
    """ Embedding layer that converts integers to one-hot vectors """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.eye(self.num_classes, requires_grad=False)[x]


class CategoricalWithoutReplacement:
    def __init__(self, max_val, min_val=0, exclusion_list=()):
        """
        Categorical distribution which samples without replacement
        :param max_val: largest value for the sampling process (excluded)
        :param min_val: smallest value for the sampling process (included)
        :param exclusion_list: which values to exclude from the sampling process
        """
        self.values = np.array([v for v in range(min_val, max_val) if v not in exclusion_list])
        self.n = self.values.size
        
        self.initial_samples_idx = 0
        self.initial_samples = np.array([[v] for v in range(min_val, max_val) if v not in exclusion_list] * 2)
        np.random.shuffle(self.initial_samples)
        
    def sample(self, sample_shape, replace=False):
        if isinstance(sample_shape, int) or len(sample_shape) == 1:
            # if self.initial_samples_idx >= self.initial_samples.size:
            #     np.random.shuffle(self.initial_samples)
            #     self.initial_samples_idx = 0                                
            # samples = self.initial_samples[self.initial_samples_idx]
            # self.initial_samples_idx += 1
            
            samples = np.random.choice(self.values, size=sample_shape, replace=replace)
        elif len(sample_shape) == 2:
            bsize, size = sample_shape
            samples = np.stack([np.random.choice(self.values, size=size, replace=replace) for _ in range(bsize)])
        else:
            raise ValueError("This distributions only supports sampling tensors of up to order 2")
        return torch.from_numpy(samples).detach()

    def get_all(self, batch_size=0):
        if batch_size == 0:
            return self.values
        else:
            return torch.from_numpy(np.stack([self.values for _ in range(batch_size)])).detach()
