# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from base.learners.skill_discovery.base import BaseSkillDiscoveryLearner


class BaseSMMLearner(BaseSkillDiscoveryLearner):
    AGENT_TYPE = 'SMM'

    def __init__(self, skill_n, **kwargs):
        self.skill_n = int(skill_n)

        # At least trigger the default usage for im and density modules
        if 'im_params' not in kwargs:
            kwargs['im_params'] = {}
        if 'density_params' not in kwargs:
            kwargs['density_params'] = {}
        super().__init__(**kwargs)
        self.im_type = 'reverse_mi'
        self.density_type = 'vae'

    def relabel_episode(self):
        super().relabel_episode()

        # Add density model reward
        self._add_density_reward()

    def relabel_batch(self, batch):
        batch = super().relabel_batch(batch)

        # Compute reward from density model
        with torch.no_grad():
            new_density_rew = self.density.novelty(batch)

        # Make sure that weights for density rewards are not None
        density_nu = self.density_nu if self.density_nu is not None else 0.

        # Detach density rewards from computation graph
        new_density_rew = new_density_rew.detach()

        batch['reward'] = batch['reward'] + density_nu * new_density_rew
        batch['density_model_reward'] = new_density_rew

        return batch

    def _compute_novelty(self, batched_episode):
        return self.density.novelty(batched_episode)

    def _add_density_reward(self):
        if self.density is not None:
            for ep in self._compress_me:
                batched_episode = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
                novelty = self._compute_novelty(batched_episode)

                if self.density_scale:
                    self.train()
                    _ = self._density_bn(novelty.view(-1, 1))
                    self.eval()
                    novelty = novelty / torch.sqrt(self._density_bn.running_var[0])

                for e, s in zip(ep, novelty):
                    e['reward'] += (self.density_nu * s.detach())
                    e['density_model_reward'] = s.detach()

    def get_density_loss(self, batch):
        return self.density(batch)
