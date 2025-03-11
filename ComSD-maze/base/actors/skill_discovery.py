# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from .base import BaseActor


class BaseSkillDiscoveryAgent(BaseActor):
    def __init__(self, **kwargs):
        self.curr_skill = None

        self.batch_keys = [
            'state', 'next_state', 'skill',
            'action', 'n_ent', 'log_prob', 'action_logit',
            'reward', 'terminal', 'complete',
        ]
        self.no_squeeze_list = []

        super().__init__(**kwargs)

    def preprocess_skill(self, curr_skill):
        """ Add here any processing needed after the skill is sampled and before feeding it to the policy """
        raise NotImplementedError

    def sample_skill(self):
        raise NotImplementedError

    def reset_skill(self, skill=None):
        self.curr_skill = self.sample_skill()
        if skill is not None:
            self.curr_skill = self.curr_skill * 0 + skill

    def reset(self, skill=None, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.episode = []
        self.reset_skill(skill)

    def collect_transitions(self, num_transitions, reset_dict={}, do_eval=False):
        self.episode = []
        for _ in range(num_transitions):
            if self.env.is_done:
                self.env.reset(**reset_dict)
                self.reset_skill()
            self.step(do_eval)

    def step(self, do_eval=False):
        s = self.env.state
        z = self.preprocess_skill(self.curr_skill)
        a, logit, log_prob, n_ent = self.policy(s.view(1, -1), z.view(1, -1), greedy=do_eval)
        a = a.view(-1)
        logit = logit.view(-1)
        log_prob = log_prob.sum()

        self.env.step(a)
        complete = self.env.is_complete if hasattr(self.env, 'is_complete') else self.env.is_success
        complete = float(complete) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        r = torch.zeros(1)
        env_rew = self.env.reward * torch.ones(1)
        discriminator_rew = torch.zeros(1)

        self.episode.append({
            'state': s,
            'skill': self.curr_skill.detach(),
            'action': a,
            'action_logit': logit,
            'log_prob': log_prob.view([]),
            'n_ent': n_ent.view([]),
            'next_state': s_next,
            'terminal': terminal.view([]),
            'complete': complete.view([]),
            'env_reward': env_rew.view([]),
            'im_reward': discriminator_rew.view([]),  # to be filled during relabeling
            'reward': r.view([]),  # to be filled during relabeling
        })


class BaseSMMAgent(BaseSkillDiscoveryAgent):

    def step(self, do_eval=False):
        super().step(do_eval=do_eval)
        density_rew = torch.zeros(1)
        self.episode[-1]['density_model_reward'] = density_rew.view([])
