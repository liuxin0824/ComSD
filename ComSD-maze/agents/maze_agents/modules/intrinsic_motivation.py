# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from base.modules.intrinsic_motivation import IntrinsicMotivationModule


class IntrinsicCuriosityModule(nn.Module, IntrinsicMotivationModule):
    def __init__(self, env, hidden_size, state_size=None, action_size=None):
        super().__init__()

        self.state_size = env.state_size if state_size is None else state_size
        self.action_size = env.action_size if action_size is None else action_size

        self.state_embedding_layers = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.inverse_model_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )

        self.forward_model_layers = nn.Sequential(
            nn.Linear(self.action_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def normalize(x):
        return x / torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True))

    def surprisal(self, episode_batch):
        """Compute surprisal for intrinsic motivation"""
        state = episode_batch['state']
        next_state = episode_batch['next_state']
        action = episode_batch['action']
        state_emb = self.normalize(self.state_embedding_layers(state))
        next_state_emb = self.normalize(self.state_embedding_layers(next_state))
        next_state_emb_hat = self.normalize(self.forward_model_layers(torch.cat([state_emb, action], dim=1)))
        return torch.mean(torch.pow(next_state_emb_hat - next_state_emb, 2), dim=1)

    def forward(self, mini_batch):
        """Compute terms for intrinsic motivation via surprisal (inlcuding losses and surprise)"""
        state = mini_batch['state']
        next_state = mini_batch['next_state']
        action = mini_batch['action']
        state_emb = self.normalize(self.state_embedding_layers(state))
        next_state_emb = self.normalize(self.state_embedding_layers(next_state))

        action_hat = self.inverse_model_layers(torch.cat([state_emb, next_state_emb], dim=1))
        inv_loss = torch.mean(torch.pow(action_hat - action, 2))

        next_state_emb_hat = self.normalize(self.forward_model_layers(torch.cat([state_emb, action], dim=1)))
        fwd_loss = torch.mean(torch.pow(next_state_emb_hat - next_state_emb.detach(), 2))

        return inv_loss + fwd_loss


class RandomNetworkDistillation(nn.Module, IntrinsicMotivationModule):
    def __init__(self, env, hidden_size, state_size=None):
        super().__init__()

        self.state_size = env.state_size if state_size is None else state_size

        self.random_network = nn.Sequential(
            nn.Linear(self.state_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
        )

        self.distillation_network = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def normalize(x):
        return x / torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True))

    def surprisal(self, episode_batch):
        """Compute surprisal for intrinsic motivation"""
        next_state = episode_batch['next_state']
        r_state_emb = self.normalize(self.random_network(next_state))
        d_state_emb = self.normalize(self.distillation_network(next_state))
        return torch.mean(torch.pow(r_state_emb - d_state_emb, 2), dim=1).detach()

    def forward(self, mini_batch):
        """Compute losses for intrinsic motivation via surprisal (inlcuding losses and surprise)"""
        next_state = mini_batch['next_state']
        r_state_emb = self.normalize(self.random_network(next_state)).detach()
        d_state_emb = self.normalize(self.distillation_network(next_state))
        return torch.mean(torch.pow(r_state_emb - d_state_emb, 2))
