# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer


class Critic(nn.Module):
    def __init__(self, env, hidden_size, use_antigoal=False, a_range=None, state_size=None, goal_size=None,
                 action_size=None, num_layers=4, normalize_inputs=False,
                 hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
        super().__init__()
        self.use_antigoal = use_antigoal

        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_size = self.state_size + self.goal_size + self.action_size
        if self.use_antigoal:
            input_size += self.goal_size

        input_normalizer = Normalizer(input_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=input_size, output_size=1, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer,
                                hidden_init_fn=hidden_init_fn, b_init_value=b_init_value, last_fc_init_w=last_fc_init_w)

    def q_no_grad(self, s, a, g, ag=None):
        for p in self.parameters():
            p.requires_grad = False

        q = self(s, a, g, ag)

        for p in self.parameters():
            p.requires_grad = True

        return q

    def forward(self, s, a, g, ag=None):
        """Produce an action"""
        if self.use_antigoal:
            return self.layers(torch.cat([s, a, g, ag], dim=1)).view(-1)
        else:
            return self.layers(torch.cat([s, a, g], dim=1)).view(-1)


class Value(nn.Module):
    def __init__(self, env, hidden_size, use_antigoal=False, a_range=None, state_size=None, goal_size=None,
                 num_layers=4, normalize_inputs=False, hidden_init_fn=None, b_init_value=None, last_fc_init_w=None,
                 antigoal_size=None):
        super().__init__()
        self.use_antigoal = use_antigoal

        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.antigoal_size = env.goal_size if antigoal_size is None else antigoal_size

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_size = self.state_size + self.goal_size
        if self.use_antigoal:
            input_size += self.antigoal_size

        input_normalizer = Normalizer(input_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=input_size, output_size=1, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer,
                                hidden_init_fn=hidden_init_fn, b_init_value=b_init_value, last_fc_init_w=last_fc_init_w)

    def forward(self, s, g, ag=None):
        if self.use_antigoal:
            return self.layers(torch.cat([s, g, ag], dim=1)).view(-1)
        else:
            return self.layers(torch.cat([s, g], dim=1)).view(-1)