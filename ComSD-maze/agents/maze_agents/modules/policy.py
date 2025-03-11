# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.distributions import Beta
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer


class Policy(nn.Module):
    def __init__(self, env, hidden_size, a_range=None, state_size=None, goal_size=None, action_size=None):
        super().__init__()
        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        input_size = self.state_size + self.goal_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )

    def forward(self, s, g):
        """Produce an action"""
        return torch.tanh(self.layers(torch.cat([s, g], dim=1)) * 0.005) * self.a_range


class StochasticPolicy(nn.Module):
    def __init__(self, env, hidden_size, a_range=None, state_size=None, goal_size=None, action_size=None,
                 num_layers=4, normalize_inputs=False):
        super().__init__()
        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_size = self.state_size + self.goal_size

        input_normalizer = Normalizer(input_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=input_size, output_size=self.action_size * 2, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer,
                                final_activation_fn=nn.Softplus)

    def action_stats(self, s, g):
        x = torch.cat([s, g], dim=1) if g is not None else s
        action_stats = self.layers(x) + 1.05 #+ 1e-6
        return action_stats[:, :self.action_size], action_stats[:, self.action_size:]

    def scale_action(self, logit):
        # Scale to [-1, 1]
        logit = 2 * (logit - 0.5)
        # Scale to the action range
        action = logit * self.a_range
        return action

    def action_mode(self, s, g):
        c0, c1 = self.action_stats(s, g)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        return self.scale_action(action_mode)

    def forward(self, s, g, greedy=False, action_logit=None):
        """Produce an action"""
        c0, c1 = self.action_stats(s, g)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        m = Beta(c0, c1)

        # Sample.
        if action_logit is None:
            if greedy:
                action_logit = action_mode
            else:
                action_logit = m.sample()

            n_ent = -m.entropy().mean()
            lprobs = m.log_prob(action_logit)
            action = self.scale_action(action_logit)
            return action, action_logit, lprobs, n_ent

        # Evaluate the action previously taken
        else:
            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit)
            action = self.scale_action(action_logit)
            return lprobs, n_ent, action


class ReparamTrickPolicy(nn.Module):
    """ Gaussian policy which makes uses of the reparameterization trick to backprop gradients from a critic """
    def __init__(self, env, hidden_size, a_range=None, state_size=None, goal_size=None, action_size=None,
                 num_layers=4, normalize_inputs=False, min_logstd=-20, max_logstd=2,
                 hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
        super().__init__()
        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

        input_size = self.state_size + self.goal_size

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_normalizer = Normalizer(input_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=input_size, output_size=self.action_size * 2, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer,
                                hidden_init_fn=hidden_init_fn, b_init_value=b_init_value, last_fc_init_w=last_fc_init_w)

    def action_stats(self, s, g):
        x = torch.cat([s, g], dim=1) if g is not None else s
        action_stats = self.layers(x)
        mean = action_stats[:, :self.action_size]
        log_std = action_stats[:, self.action_size:]
        log_std = torch.clamp(log_std, self.min_logstd, self.max_logstd)
        std = log_std.exp()
        return mean, std

    def scale_action(self, logit):
        # Scale to the action range
        action = logit * self.a_range
        return action

    def forward(self, s, g=None, greedy=False, action_logit=None):
        mean, std = self.action_stats(s, g)
        m = torch.distributions.Normal(mean, std)

        # Sample.
        if action_logit is None:
            if greedy:
                action_logit_unbounded = mean
            else:
                action_logit_unbounded = m.rsample()  # for the reparameterization trick
            action_logit = torch.tanh(action_logit_unbounded)

            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit_unbounded) - torch.log(1 - action_logit.pow(2) + 1e-6)
            action = self.scale_action(action_logit)
            return action, action_logit_unbounded, lprobs, n_ent

        # Evaluate the action previously taken
        else:
            action_logit_unbounded = action_logit
            action_logit = torch.tanh(action_logit_unbounded)
            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit_unbounded) - torch.log(1 - action_logit.pow(2) + 1e-6)
            action = self.scale_action(action_logit)
            return lprobs, n_ent, action
