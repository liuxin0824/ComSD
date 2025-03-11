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


class SkillDynamics(nn.Module, IntrinsicMotivationModule):
    def __init__(self, n, state_size, hidden_size, num_layers=4, skill_preprocessing_fn=lambda x: x,
                 normalize_inputs=False):
        super().__init__()

        self.n = n
        self.state_size = int(state_size)
        assert num_layers >= 2
        self.num_layers = int(num_layers)
        self.normalize_inputs = bool(normalize_inputs)
        self.skill_preprocessing_fn = skill_preprocessing_fn

        self._make_normalizer_module()  # in practice, this will normalize states (outputs) instead of skills (inputs)
        self.layers = create_nn(input_size=self.input_size, output_size=self.output_size, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=nn.Sequential())

        self.mse_loss = nn.MSELoss(reduction='none')

    @property
    def input_size(self):
        return self.n

    @property
    def output_size(self):
        return self.state_size

    @property
    def normalizes_inputs(self):
        return self.normalizer is not None

    def _make_normalizer_module(self):
        self.normalizer = Normalizer(self.state_size) if self.normalize_inputs else None

    def update_normalizer(self, **kwargs):
        if self.normalizes_inputs:
            self.normalizer.update(**kwargs)

    def compute_logprob_under_latent(self, batch, z=None, **kwargs):
        """ Compute p(s|z) for an arbitrary z """
        s = batch["next_state"]
        if z is None:
            z = batch["skill"]
        z = self.skill_preprocessing_fn(z)
        s_ = self.layers(z)
        if self.normalizes_inputs:
            s_ = self.normalizer.denormalize(s_)
        logprob = -1. * self.mse_loss(s, s_).sum(dim=1)
        return logprob

    def compute_logprob(self, batch, **kwargs):
        """ Compute p(s|z) for the skill used to collect s """
        return self.compute_logprob_under_latent(batch, z=None)

    def forward(self, batch):
        loss = - self.compute_logprob(batch)
        self.update_normalizer(x=batch["next_state"])
        return loss.mean()

    def surprisal(self, batch):
        with torch.no_grad():
            log_q_s_z = self.compute_logprob(batch).detach()  # \log q_{\phi}(s|z)
        sum_q_s_z_i = torch.zeros_like(log_q_s_z)  # \sum_{i=1}^{N} q_{\phi}(s|z_i)
        for z_i in range(self.n):
            skill = torch.full_like(batch["skill"], z_i, dtype=torch.long)
            with torch.no_grad():
                sum_q_s_z_i += torch.exp(self.compute_logprob_under_latent(batch, z=skill).detach())  # q_{\phi}(s|z_i)
        # r(s,z) = \log q_{\phi}(s|z) - \log \frac{1}{N} \sum_{i=1}^{N} q_{\phi}(s|z_i), \, z_i \sim p(z)
        r = log_q_s_z + float(np.log(self.n)) - torch.log(sum_q_s_z_i)
        return r
