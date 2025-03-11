# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, input_size, epsilon=0.01, zero_mean=False, momentum=None, extra_dims=0):
        super(Normalizer, self).__init__()

        self.input_size = int(input_size)
        self.epsilon = float(epsilon)
        self.zero_mean = bool(zero_mean)
        self.momentum = 1 if momentum is None else min(1, max(0, momentum))
        self.extra_dims = int(extra_dims)

        self.register_buffer('running_sum', torch.zeros(self.input_size))
        self.register_buffer('running_sumsq', torch.zeros(self.input_size) + self.epsilon)
        self.register_buffer('count', torch.zeros(1) + self.epsilon)

    @property
    def mean(self):
        if self.zero_mean:
            return torch.zeros_like(self.running_sum).view(1, self.input_size)
        m = self.running_sum / self.count
        return m.view(1, self.input_size).detach()

    @property
    def std(self):
        var = (self.running_sumsq / self.count) - torch.pow(self.mean, 2)
        var = var.masked_fill(var < self.epsilon, self.epsilon)
        std = torch.pow(var, 0.5)
        return std.view(1, self.input_size).detach()

    def split(self, x):
        if self.extra_dims == 0:
            return x, None
        return x[:, :-self.extra_dims], x[:, -self.extra_dims:]

    def join(self, x1, x2):
        if self.extra_dims == 0:
            return x1
        return torch.cat([x1, x2], dim=1)

    def update(self, x):
        self.running_sum *= self.momentum
        self.running_sumsq *= self.momentum
        self.count *= self.momentum

        x = x.view(-1, self.input_size)
        self.running_sum += x.sum(dim=0).detach()
        self.running_sumsq += torch.pow(x, 2).sum(dim=0).detach()
        self.count += x.shape[0]

    def forward(self, x):
        x1, z2 = self.split(x)
        z1 = (x1 - self.mean) / self.std
        if self.training:
            self.update(x1)
        z1 = torch.clamp(z1, -5.0, 5.0)
        return self.join(z1, z2)

    def denormalize(self, z):
        z1, x2 = self.split(z)
        x1 = z1 * self.std + self.mean
        return self.join(x1, x2)


class DatasetNormalizer(nn.Module):
    def __init__(self, input_size, epsilon=0.01, zero_mean=False):
        super(DatasetNormalizer, self).__init__()

        self.input_size = input_size
        self.epsilon = epsilon ** 2  # for consistency with Normalizer
        self.zero_mean = bool(zero_mean)

        self.register_buffer('mean_buffer', torch.zeros(self.input_size))
        self.register_buffer('std_buffer', torch.full((self.input_size,), epsilon))

    def update(self, dataset=None, mean=None, std=None):
        if dataset is None:
            assert std is not None
            std = std.masked_fill(std < self.epsilon, self.epsilon)
            if self.zero_mean:
                mean = torch.zeros_like(std)
            else:
                assert mean is not None
        else:
            assert mean is None
            assert std is None
            std = dataset.std(dim=0)
            mean = dataset.mean(dim=0) if not self.zero_mean else torch.zeros_like(std)

        self.mean_buffer = mean
        self.std_buffer = std

    @property
    def mean(self):
        return self.mean_buffer.detach()

    @property
    def std(self):
        return self.std_buffer.detach()

    def forward(self, x):
        z = (x - self.mean) / self.std
        return torch.clamp(z, -5.0, 5.0)

    def denormalize(self, x):
        return x * self.std + self.mean
