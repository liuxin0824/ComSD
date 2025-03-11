# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from base.modules.generic import CategoricalWithoutReplacement
from base.actors.skill_discovery import BaseSkillDiscoveryAgent


class StochasticAgent(BaseSkillDiscoveryAgent):
    def __init__(self, skill_n, **kwargs):
        self.skill_n = int(skill_n)

        super().__init__(**kwargs)

        self.skill_dist = CategoricalWithoutReplacement(self.skill_n)

    def _make_modules(self, policy, skill_embedding):
        self.policy = policy
        self.skill_embedding = skill_embedding

    def preprocess_skill(self, curr_skill):
        assert curr_skill is not None
        return self.skill_embedding(curr_skill)

    def sample_skill(self):
        return self.skill_dist.sample(sample_shape=(1,)).view([])

    @property
    def rollout(self):
        states = torch.stack([e['state'] for e in self.episode] + [self.episode[-1]['next_state']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]
