# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from abc import ABC, abstractmethod


class DensityModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def novelty(self, *args, **kwargs):
        return torch.zeros(10)
