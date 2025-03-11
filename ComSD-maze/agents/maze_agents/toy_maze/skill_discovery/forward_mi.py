# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .base import StochasticAgent
from agents.maze_agents.toy_maze.env import Env
from base.modules.generic import OneHotEmbedding
from agents.maze_agents.modules import StochasticPolicy, Value
from base.modules.skill_discovery.forward_mi import SkillDynamics
from base.learners.skill_discovery.forward_mi import BaseForwardMILearner


class ForwardMILearner(BaseForwardMILearner):
    def create_env(self):
        return Env(**self.env_params)

    def _make_agent(self):
        return StochasticAgent(skill_n=self.skill_n, env=self.create_env(), policy=self.policy,
                               skill_embedding=self.skill_emb)

    def _make_agent_modules(self):
        self._make_skill_embedding()
        kwargs = dict(env=self._dummy_env, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      goal_size=self.skill_n, normalize_inputs=self.normalize_inputs)
        self.policy = StochasticPolicy(**kwargs)
        self.v_module = Value(use_antigoal=False, **kwargs)

    def _make_skill_embedding(self):
        self.skill_emb = OneHotEmbedding(self.skill_n)

    def _make_im_modules(self):
        return SkillDynamics(self.skill_n, self._dummy_env.state_size,
                             num_layers=self.num_layers, hidden_size=self.hidden_size,
                             skill_preprocessing_fn=self.preprocess_skill, normalize_inputs=self.normalize_inputs)
