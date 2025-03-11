# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT


import torch
from base.learners.distance import BaseDistanceLearner, BaseSiblingRivalryLearner


class BaseEDLLearner(BaseDistanceLearner):
    AGENT_TYPE = 'EDL'

    def __init__(self, env_reward=False, hidden_size=128, num_layers=4, normalize_inputs=False, **kwargs):
        self.env_reward = bool(env_reward)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.normalize_inputs = bool(normalize_inputs)

        super().__init__(**kwargs)

        self.ep_summary_keys = ['success', 'dist_to_goal',
                                'cumulative_rew', 'cumulative_im_rew', 'cumulative_density_rew']

    def preprocess_skill(self, z, **kwargs):
        return self.agent.preprocess_skill(z, **kwargs)

    def sample_skill(self):
        return self.agent.sample_skill()

    def _compute_surprisal(self, batched_episode):
        return self.im.surprisal(batched_episode)

    def _add_im_reward(self):
        if self.im is not None:
            for ep in self._compress_me:
                batched_episode = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
                surprisals = self._compute_surprisal(batched_episode)

                if self.im_scale:
                    self.train()
                    _ = self._im_bn(surprisals.view(-1, 1))
                    self.eval()
                    surprisals = surprisals / torch.sqrt(self._im_bn.running_var[0])

                for e, s in zip(ep, surprisals):
                    e['reward'] += (self.im_nu * s.detach())
                    e['im_reward'] = s.detach()

    def get_im_loss(self, batch):
        return self.im(batch)

    def fill_summary(self, *values):
        manual_summary = [float(self.was_success),
                          float(self.dist_to_goal),
                          float(sum([e['reward'] for e in self.agent.episode])),
                          float(sum([e.get('im_reward', 0.) for e in self.agent.episode])),
                          float(sum([e.get('density_model_reward', 0.) for e in self.agent.episode]))]

        for v in values:
            manual_summary.append(v.item())

        self._ep_summary = manual_summary


class BaseEDLSiblingRivalryLearner(BaseSiblingRivalryLearner, BaseEDLLearner):
    AGENT_TYPE = 'EDL+SR'

    def __init__(self, env_reward=False, hidden_size=128, num_layers=4, normalize_inputs=False, **kwargs):
        self.env_reward = bool(env_reward)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.normalize_inputs = bool(normalize_inputs)

        super().__init__(**kwargs)

        self.ep_summary_keys = ['success', 'dist_to_goal', 'dist_to_antigoal',
                                'cumulative_rew', 'cumulative_im_rew', 'cumulative_density_rew']

    def play_episode(self, reset_dict=None, do_eval=False, **kwargs):
        self._reset_ep_stats()

        if reset_dict is None:
            reset_dict = {}

        for agent in self.agents:
            agent.play_episode(reset_dict, do_eval)
            reset_dict = agent.env.sibling_reset
            reset_dict['skill'] = agent.curr_skill
        self.relabel_episode()

    def fill_summary(self, *values):
        manual_summary = [float(self.was_success),
                          float(self.avg_dist_to_goal),
                          float(self.avg_dist_to_antigoal),
                          float(sum([e['reward'] for e in self.agent.episode])),
                          float(sum([e.get('im_reward', 0.) for e in self.agent.episode])),
                          float(sum([e.get('density_model_reward', 0.) for e in self.agent.episode]))]

        for v in values:
            manual_summary.append(v.item())

        self._ep_summary = manual_summary
