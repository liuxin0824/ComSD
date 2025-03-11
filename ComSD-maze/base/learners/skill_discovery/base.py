# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from ..base import BaseLearner


class BaseSkillDiscoveryLearner(BaseLearner):

    def __init__(self, env_reward=False, hidden_size=128, num_layers=4, normalize_inputs=False, **kwargs):
        self.env_reward = bool(env_reward)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.normalize_inputs = bool(normalize_inputs)

        super().__init__(**kwargs)

        self.ep_summary_keys = ["cumulative_rew", "cumulative_im_rew", "cumulative_density_rew"]

    def fill_summary(self, *values):
        self._ep_summary = [float(sum([e['reward'] for e in self.agent.episode])),
                            float(sum([e.get('im_reward', 0.) for e in self.agent.episode])),
                            float(sum([e.get('density_model_reward', 0.) for e in self.agent.episode]))]
        self._ep_summary += [v.item() for v in values]

    def relabel_episode(self):
        # self._compress_me = []

        # for e in self.agent.episode:
        #     # Optionally take into account extrinsic reward
        #     r = e['env_reward'] * float(self.env_reward)
        #     e['reward'] = r
        # self._compress_me.append(self.agent.episode)
        assert len(self._compress_me[0]) == 50 * 50 and len(self._compress_me) == 1
        # Add discriminator reward
        self._add_im_reward()

    def relabel_batch(self, batch):
        # Compute intrinsic rewards
        with torch.no_grad():
            new_im_rew = self.im.surprisal(batch)
            if self.density is not None:
                new_density_rew = self.density.novelty(batch)
            else:
                new_density_rew = torch.zeros_like(new_im_rew)

        # Make sure that weights for intrinsic rewards are not None
        im_nu = self.im_nu if self.im_nu is not None else 0.
        density_nu = self.density_nu if self.density_nu is not None else 0.

        # Detach intrinsic rewards from computation graph
        new_im_rew = new_im_rew.detach()
        new_density_rew = new_density_rew.detach()

        # Optionally take into account extrinsic reward
        r = batch['env_reward'] * float(self.env_reward)
        # Add intrinsic rewards
        r += im_nu * new_im_rew + density_nu * new_density_rew

        batch['reward'] = r
        batch['im_reward'] = new_im_rew
        batch['density_model_reward'] = new_density_rew

        return batch

    def _compute_surprisal(self, batched_episode):
        return self.im.surprisal(batched_episode)

    def _add_im_reward(self):
        if self.im is not None:
            for ep in self._compress_me:
                batched_episode = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
                with torch.no_grad():
                    surprisals = self._compute_surprisal(batched_episode)

                if self.im_scale:
                    self.train()
                    _ = self._im_bn(surprisals.view(-1, 1))
                    self.eval()
                    surprisals = surprisals / torch.sqrt(self._im_bn.running_var[0])

                for e, s in zip(ep, surprisals):
                    e['reward'] += (self.im_nu * s.detach())
                    e['im_reward'] = s.detach()

    def preprocess_skill(self, z, **kwargs):
        return self.agent.preprocess_skill(z, **kwargs)

    def get_values(self, batch):
        return self.v_module(
            batch['state'],
            self.preprocess_skill(batch['skill']),
        )

    def get_terminal_values(self, batch):
        return self.v_module(
            batch['next_state'][-1:],
            self.preprocess_skill(batch['skill'][-1:])
        )

    def get_policy_lprobs_and_nents(self, batch):
        log_prob, n_ent, _ = self.policy(
            batch['state'],
            self.preprocess_skill(batch['skill']),
            action_logit=batch['action_logit']
        )
        return log_prob.sum(dim=1), n_ent

    def get_im_loss(self, batch):
        return self.im(batch)

    def soft_update(self):
        module_pairs = [
            dict(source=self.v_module, target=self.v_target),
        ]
        for pair in module_pairs:
            for p, p_targ in zip(pair['source'].parameters(), pair['target'].parameters()):
                p_targ.data *= self.polyak
                p_targ.data += (1 - self.polyak) * p.data

    def _get_q_module(self, q_i):
        q_i = q_i if q_i is not None else 1
        assert q_i in [1, 2]
        return [self.q1, self.q2][q_i - 1]

    def get_action_qs(self, batch, q_i=None):
        return self.get_curr_qs(batch, new_actions=None, q_i=q_i)

    def get_policy_loss_and_actions(self, batch):
        policy_actions, logprobs = self.sample_policy_actions_and_lprobs(batch)
        p_obj = self.q1.q_no_grad(batch['state'], policy_actions, self.preprocess_skill(batch['skill']))
        if hasattr(self, 'alpha'):  # for SAC
            p_obj -= self.alpha * logprobs
        p_losses = -p_obj  # flip sign to turn the maximization objective into a loss function to minimize
        p_loss = p_losses.mean()
        return p_loss, policy_actions

    def get_curr_qs(self, batch, new_actions=None, q_i=None):
        """
        Compute Q_i(s,a). Use new_actions to override the actions in the batch (e.g. for SAC).
        q_i selects the index of the Q-function.
        """
        action = new_actions if new_actions is not None else batch['action']
        return self._get_q_module(q_i)(
            batch['state'],
            action,
            self.preprocess_skill(batch['skill'])
        )

    def get_next_vs(self, batch):
        return self.v_target(
            batch['next_state'],
            self.preprocess_skill(batch['skill']),
        )

    def sample_policy_actions_and_lprobs(self, batch):  # For SAC; we need to sample new actions when updating V
        """ Sample new actions. Returns (actions, logprobs) tuple. """
        action, action_logit, lprobs, n_ent = self.policy(
            batch['state'],
            self.preprocess_skill(batch['skill'])
        )
        return action, lprobs.sum(dim=1)
