# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule


class Discriminator(nn.Module, IntrinsicMotivationModule):
    def __init__(self, n, state_size, hidden_size, num_layers=4, normalize_inputs=False,
                 input_key='next_state', input_size=None):
        super().__init__()

        self.n = n
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.input_key = str(input_key)
        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_normalizer = Normalizer(self.state_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=self.state_size, output_size=self.n, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer)

        self.softmax = nn.Softmax(dim=1)

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def log_approx_posterior(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)
        p = self.softmax(x)
        p_skills = p[torch.arange(0, p.shape[0]), batch['skill']]
        return torch.log(p_skills)

    def surprisal(self, batch):
        return self.log_approx_posterior(batch) - float(np.log(1.0 / self.n))

    def skill_assignment(self, batch):
        return torch.argmax(self.layers(batch[self.input_key]), dim=1)

    def forward(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)
        return self.loss(x, batch['skill']).mean()
