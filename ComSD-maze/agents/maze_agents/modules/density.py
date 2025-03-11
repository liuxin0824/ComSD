# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.distributions import Beta
from dist_train.utils.helpers import create_nn
from base.modules.density import DensityModule
from base.modules.normalization import Normalizer
from base.modules.vector_quantization.embeddings import VQEmbedding


class BaseVAEDensity(nn.Module, DensityModule):
    def __init__(self, num_skills, state_size, hidden_size, code_size,
                 num_layers=4, normalize_inputs=False, skill_preprocessing_fn=lambda x: x,
                 input_key='next_state', input_size=None):
        super().__init__()

        self.num_skills = int(num_skills)
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.code_size = int(code_size)
        self.normalize_inputs = bool(normalize_inputs)
        self.skill_preprocessing_fn = skill_preprocessing_fn
        self.input_key = str(input_key)

        self._make_normalizer_module()

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        self.encoder = create_nn(input_size=self.input_size, output_size=self.encoder_output_size,
                                 hidden_size=hidden_size, num_layers=self.num_layers,
                                 input_normalizer=self.normalizer if self.normalizes_inputs else nn.Sequential())

        self.decoder = create_nn(input_size=self.code_size, output_size=self.input_size,
                                 hidden_size=hidden_size, num_layers=self.num_layers)

        self.mse_loss = nn.MSELoss(reduction='none')

    @property
    def input_size(self):
        return self.state_size + self.num_skills

    @property
    def encoder_output_size(self):
        return NotImplementedError

    @property
    def normalizes_inputs(self):
        return self.normalizer is not None

    def _make_normalizer_module(self):
        raise NotImplementedError

    def compute_logprob(self, batch, **kwargs):
        raise NotImplementedError

    def novelty(self, batch, **kwargs):
        with torch.no_grad():
            return -self.compute_logprob(batch, **kwargs).detach()

    def update_normalizer(self, **kwargs):
        if self.normalizes_inputs:
            self.normalizer.update(**kwargs)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)

    def forward(self, batch):
        raise NotImplementedError


class VAEDensity(BaseVAEDensity):
    def __init__(self, num_skills, state_size, hidden_size, code_size, beta=0.5, loss_mult=1., **kwargs):
        super().__init__(num_skills=num_skills, state_size=state_size, hidden_size=hidden_size, code_size=code_size,
                         **kwargs)

        self.beta = float(beta)
        self.loss_mult = float(loss_mult)

    def _make_normalizer_module(self):
        self.normalizer = Normalizer(self.state_size, extra_dims=self.num_skills) if self.normalize_inputs else None

    @property
    def encoder_output_size(self):
        return self.code_size * 2

    def compute_logprob(self, batch, sample=True, with_moments=False, sum_logprob=True):
        s, z = batch[self.input_key], self.skill_preprocessing_fn(batch['skill'])
        x = torch.cat([s, z], dim=1)
        mu_and_logvar = self.encoder(x)
        mu, logvar = mu_and_logvar[:, :self.code_size], mu_and_logvar[:, self.code_size:]
        std = (0.5 * logvar).exp()
        if sample:
            normal = torch.distributions.Normal(mu, std)
            code = normal.rsample()  # sample with the reparameterization trick
        else:
            code = mu + torch.ones_like(mu) * std
        x_ = self.decoder(code)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_)
        if sum_logprob:
            logprob = logprob.sum(dim=1)
        if with_moments:
            return logprob, mu, logvar
        else:
            return logprob

    def novelty(self, batch, sample=True):
        with torch.no_grad():
            return -self.compute_logprob(batch, sample=sample, with_moments=False).detach()

    def forward(self, batch):
        logprob, mu, logvar = self.compute_logprob(batch, sample=True, with_moments=True, sum_logprob=False)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss = self.beta * kle.mean() - logprob.mean()
        loss *= self.loss_mult
        return loss


class VQVAEDensity(BaseVAEDensity):
    def __init__(self, num_skills, state_size, hidden_size, codebook_size, code_size, beta=0.25, **kwargs):
        super().__init__(num_skills=num_skills, state_size=state_size, hidden_size=hidden_size, code_size=code_size,
                         **kwargs)
        self.codebook_size = int(codebook_size)
        self.beta = float(beta)

        self.apply(self.weights_init)

        self.vq = VQEmbedding(self.codebook_size, self.code_size, self.beta)

    @property
    def encoder_output_size(self):
        return self.code_size

    def _make_normalizer_module(self):
        self.normalizer = Normalizer(self.input_size) if self.normalize_inputs else None

    @classmethod
    def weights_init(cls, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)

    def compute_logprob(self, batch, with_codes=False):
        s, z = batch[self.input_key], self.skill_preprocessing_fn(batch['skill'])
        x = torch.cat([s, z], dim=1)
        z_e_x = self.encoder(x)
        z_q_x, selected_codes = self.vq.straight_through(z_e_x)
        x_ = self.decoder(z_q_x)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        if with_codes:
            return logprob, z_e_x, selected_codes
        else:
            return logprob

    def get_centroids(self, batch):
        z_idx = batch['skill']
        z_q_x = torch.index_select(self.vq.embedding.weight.detach(), dim=0, index=z_idx)
        centroids = self.decoder(z_q_x)
        if self.normalizes_inputs:
            centroids = self.normalizer.denormalize(centroids)
        return centroids

    def novelty(self, batch, **kwargs):
        with torch.no_grad():
            return -self.compute_logprob(batch, with_codes=False).detach()

    def forward(self, batch):
        logprob, z_e_x, selected_codes = self.compute_logprob(batch, with_codes=True)
        loss = self.vq(z_e_x, selected_codes) - logprob
        return loss.mean()
