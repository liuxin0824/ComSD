# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from agents.maze_agents.toy_maze.env import Env
from base.modules.generic import OneHotEmbedding
from agents.maze_agents.modules.density import VAEDensity
from base.learners.skill_discovery.smm import BaseSMMLearner
from .reverse_mi import StochasticAgent as BaseStochasticAgent
from base.modules.skill_discovery.reverse_mi import Discriminator
from agents.maze_agents.modules import ReparamTrickPolicy, Value, Critic


class StochasticAgent(BaseStochasticAgent):

    def step(self, do_eval=False):
        super().step(do_eval=do_eval)
        density_rew = torch.zeros(1)
        self.episode[-1]['density_model_reward'] = density_rew.view([])


class SMMLearner(BaseSMMLearner):
    def create_env(self):
        return Env(**self.env_params)

    def _make_agent(self):
        return StochasticAgent(skill_n=self.skill_n, env=self.create_env(), policy=self.policy,
                               skill_embedding=self.skill_emb)

    def _make_agent_modules(self):
        self._make_skill_embedding()
        kwargs = dict(env=self._dummy_env, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      goal_size=self.skill_n, normalize_inputs=self.normalize_inputs)
        self.policy = ReparamTrickPolicy(**kwargs)
        self.v_module = Value(use_antigoal=False, **kwargs)
        self.v_target = Value(use_antigoal=False, **kwargs)
        self.q1 = Critic(use_antigoal=False, **kwargs)
        self.q2 = Critic(use_antigoal=False, **kwargs)

    def _make_skill_embedding(self):
        self.skill_emb = OneHotEmbedding(self.skill_n)

    def _make_im_modules(self):
        return Discriminator(self.skill_n, self._dummy_env.state_size,
                             num_layers=self.num_layers, hidden_size=self.hidden_size,
                             normalize_inputs=self.normalize_inputs, **self.im_kwargs)

    def _make_density_modules(self):
        self.density = VAEDensity(num_skills=self.skill_n, state_size=self._dummy_env.state_size,
                                  hidden_size=self.hidden_size, num_layers=self.num_layers,
                                  skill_preprocessing_fn=self.preprocess_skill,
                                  **self.density_kwargs)

    def get_density_loss(self, batch):
        return self.density(batch)
