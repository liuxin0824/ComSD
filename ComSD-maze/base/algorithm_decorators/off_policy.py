# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from base.learners.base import BaseLearner


def dqn_decorator(partial_agent_class):
    assert issubclass(partial_agent_class, BaseLearner)

    class NewClass(partial_agent_class):
        def __init__(self, epsilon=0.0, polyak=0.95, **kwargs):
            self.epsilon = epsilon
            self.polyak = polyak

            super().__init__(**kwargs)

        def forward(self, mini_batch):
            self.train()

            # Get the q target
            q_next = self.get_next_qs(mini_batch)

            # Bootstrap from early terminal means bootstrap from terminal value when episode didn't complete
            if self.bootstrap_from_early_terminal:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['complete']) * self.gamma * q_next)
            else:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['terminal']) * self.gamma * q_next)

            # Get the Q values associated with the observed transitions
            q = self.get_action_qs(mini_batch)

            # Loss for the q_module
            q_loss = torch.pow(q - q_targ.detach(), 2).mean()

            p_loss = torch.zeros_like(q_loss)
            n_ent = torch.zeros_like(q_loss)

            self.fill_summary(mini_batch['reward'].mean(), q.mean(), q_loss, p_loss, n_ent)

            if self.im is not None:
                q_loss += (self.im_lambda * self.get_im_loss(mini_batch))

            self.eval()

            return q_loss

    return NewClass


def ddpg_decorator(partial_agent_class):
    assert issubclass(partial_agent_class, BaseLearner)

    class NewClass(partial_agent_class):
        def __init__(self, noise=0.0, epsilon=0.0, action_l2_lambda=0.0, polyak=0.95, **kwargs):
            self.noise = float(noise)
            assert 0 <= self.noise

            self.epsilon = float(epsilon)
            assert 0 <= self.epsilon <= 1.0

            self.action_l2_lambda = float(action_l2_lambda)
            assert 0 <= self.action_l2_lambda

            self.polyak = polyak
            assert 0 <= self.polyak <= 1.0

            super().__init__(**kwargs)

            self.ep_summary_keys += ["avg_batch_rew", "avg_q", "q_loss", "p_loss", "l2_loss"]

        def forward(self, mini_batch):
            # Get the q target
            q_next = self.get_next_qs(mini_batch)

            # Bootstrap from early terminal means bootstrap from terminal value when episode didn't complete
            if self.bootstrap_from_early_terminal:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['complete']) * self.gamma * q_next)
            else:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['terminal']) * self.gamma * q_next)
            # q_targ = torch.clamp(q_targ, *self._q_clamp)

            # Get the Q values associated with the observed transitions
            q = self.get_action_qs(mini_batch)

            # Loss for the q_module
            q_loss = torch.pow(q - q_targ.detach(), 2).mean()

            # We want to optimize the actions wrt their q value (without getting q module gradients)
            p_loss, policy_actions = self.get_policy_loss_and_actions(mini_batch)

            l2 = torch.mean(policy_actions ** 2)
            l2_loss = l2 * self.action_l2_lambda

            self.fill_summary(mini_batch['reward'].mean(), q.mean(), q_loss, p_loss, l2)

            loss = p_loss + q_loss + l2_loss

            if self.im is not None:
                loss += (self.im_lambda * self.get_im_loss(mini_batch))

            return loss

    return NewClass


def sac_decorator(partial_agent_class):
    """
    Decorator for Soft Actor-Critic (SAC). The learner needs the following components:
        - Estimator for V(s)
        - Target network for V(s), which will be updated with Polyak averaging
        - Two estimators for Q(s,a), Q_1 and Q_2.
            - Q_1 will be used to update the policy in a DDPG fashion
            - min(Q_1, Q_2) will be used to define targets for V(s)
            - Both Q_1 and Q_2 will use the same target, defined by q_t = r + V_targ(s')
        - Stochastic policy; gradients coming from the critic will be estimated with the reparameterization trick
    """
    assert issubclass(partial_agent_class, BaseLearner)

    class SACLearner(partial_agent_class):
        def __init__(self, alpha=0.1, polyak=0.95, **kwargs):
            self.alpha = float(alpha)
            assert 0 <= self.alpha

            self.polyak = polyak
            assert 0 <= self.polyak <= 1.0

            super().__init__(**kwargs)

            self.ep_summary_keys += ["avg_batch_rew", "avg_q1", "avg_q2", "q1_loss", "q2_loss", "p_loss", "v_loss"]

        def forward(self, mini_batch):
            # Sample actions from the current version of the policy
            new_actions, new_action_logprobs = self.sample_policy_actions_and_lprobs(mini_batch)

            # Get Q_i(s, new_actions) for i=1,2
            q1_new = self.get_curr_qs(mini_batch, new_actions=new_actions, q_i=1)
            q2_new = self.get_curr_qs(mini_batch, new_actions=new_actions, q_i=2)

            # Reduce overestimation bias by taking the minimum
            qmin = torch.min(q1_new, q2_new)

            # Compute V_targ(s')
            v_next = self.get_next_vs(mini_batch)

            # Define target for V(s)
            v_targ = qmin - self.alpha * new_action_logprobs

            # Get V(s) associated with the observed transitions
            v = self.get_values(mini_batch)

            # Loss for the v module
            v_loss = torch.pow(v - v_targ.detach(), 2).mean()

            # Bootstrap from early terminal means bootstrap from terminal value when episode didn't complete
            if self.bootstrap_from_early_terminal:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['complete']) * self.gamma * v_next)
            else:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['terminal']) * self.gamma * v_next)
            # q_targ = torch.clamp(q_targ, *self._q_clamp)

            # Get the Q values associated with the observed transitions
            q1 = self.get_action_qs(mini_batch, q_i=1)
            q2 = self.get_action_qs(mini_batch, q_i=2)

            # Loss for the q modules (the target is the same for both)
            q1_loss = torch.pow(q1 - q_targ.detach(), 2).mean()
            q2_loss = torch.pow(q2 - q_targ.detach(), 2).mean()

            # We want to optimize the actions wrt their q value (without getting q module gradients)
            p_loss, policy_actions = self.get_policy_loss_and_actions(mini_batch)

            self.fill_summary(mini_batch['reward'].mean(), q1.mean(), q2.mean(), q1_loss, q2_loss, p_loss, v_loss)

            loss = p_loss + q1_loss + q2_loss + v_loss

            return loss

    return SACLearner
