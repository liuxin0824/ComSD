# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import time
import torch
import numpy as np
import torch.distributed as dist
from dist_train.utils.shared_optim import SharedAdam as Adam
from dist_train.workers.utils import create_worker_logger, ReplayBuffer
from agents import agent_classes


def _save_buffer(exp_dir, curr_epoch, replay_buffer):
    """ Replay buffer saving logic, which is the same for most managers. """
    tstart = time.time()
    save_dir = os.path.join(exp_dir, '{:04d}_replay_buffer'.format(curr_epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    replay_buffer.save_buffer(save_dir)
    total_time = time.time() - tstart
    print("\nSaved replay buffer at epoch {} (took {:.2f} seconds)\n".format(curr_epoch, total_time), flush=True)


class BaseOffPolicyManager:
    def __init__(self, rank, config, settings):
        self.rank = rank
        self.config = config
        self.settings = settings

        # For logging
        # self.tag = os.environ.get('EXP_TAG', self.settings.tag)
        # self.exp_dir = os.path.join(BASE_DIR, self.tag)
        config_path = self.settings.config_path
        exp_name = config_path.split('/')[-1][:-5]
        self.exp_dir = os.path.join(self.settings.log_dir, exp_name)
        self.logger = create_worker_logger(rank, self.exp_dir)

        self.model_path = os.path.join(self.exp_dir, 'model.pth.tar')
        self.optim_path = os.path.join(self.exp_dir, 'optim.pth.tar')
        self.aux_optim_path = os.path.join(self.exp_dir, 'aux_optim.pth.tar')

        # Instantiate a copy of the model and place it on this worker's GPU
        agent_class = agent_classes(self.config['agent_type'], self.config['learner_type'], self.config['train_type'])
        self.agent_model = agent_class(**self.config['agent_params'])

        if os.path.isfile(self.model_path):
            self.agent_model.load_checkpoint(self.model_path)

        for parameter in self.agent_model.state_dict().values():
            dist.broadcast(parameter.data, src=0)

        self.optim = Adam(self.agent_model.parameters(), lr=config['learning_rate'])
        if os.path.isfile(self.optim_path):
            self.optim.load_state_dict(torch.load(self.optim_path))

        if self.agent_model.im is not None or self.agent_model.density is not None:
            self.aux_optim = Adam(self.agent_model.get_aux_optim_params())
            if os.path.isfile(self.aux_optim_path):
                self.aux_optim.load_state_dict(torch.load(self.aux_optim_path))
        else:
            self.aux_optim = None

        # Initialize the buffer
        self.replay_buffer = ReplayBuffer(self.agent_model, self.config)
        self._group_ready = False

        # Initialize trackers/counters
        self.time_keeper = {
            'n_rounds': 0,
            'ep_save': 0,
            'last_ep_num': int(self.agent_model.train_steps.data.item()),
            'idle_ep_count': 0,
        }
        self.epoch_buffer = []

        self.curr_epoch = 0

        self.eval_stats = {}

    @property
    def sample_type(self):
        raise NotImplementedError

    @staticmethod
    def condense_loss(loss_):
        if isinstance(loss_, torch.Tensor):
            return loss_.mean()
        if isinstance(loss_, (list, tuple)):
            net_loss = 0.
            for sub_loss in loss_:
                net_loss += sub_loss.mean()
            return net_loss
        else:
            raise TypeError

    def group_is_ready(self):
        if self._group_ready:
            return True
        if self.replay_buffer.size >= self.replay_buffer.min_size:
            status = torch.ones(1)
        else:
            status = torch.zeros(1)
        dist.all_reduce(status)
        if status.item() >= dist.get_world_size():
            self._group_ready = True
            return True
        else:
            return False

    def checkpoint(self):
        self.agent_model.save_checkpoint(self.model_path)
        torch.save(self.optim, self.optim_path)
        if self.aux_optim is not None:
            torch.save(self.aux_optim, self.aux_optim_path)

        if self.config.get('save_buffer', False) and self.curr_epoch % self.config.get('save_buffer_freq', 1) == 0:
            _save_buffer(exp_dir=self.exp_dir, curr_epoch=self.curr_epoch, replay_buffer=self.replay_buffer)

        if self.settings.keep_checkpoints:
            checkpoint_path = os.path.join(self.exp_dir, '{:04d}_model.pth.tar'.format(self.curr_epoch))
            self.agent_model.save_checkpoint(checkpoint_path)

        # self.agent_model.save_checkpoint(os.path.join(self.exp_dir, '{:04d}_model.pth.tar'.format(self.curr_epoch)))
        # torch.save(self.optim.state_dict(), os.path.join(self.exp_dir, '{:04d}_optim.pth.tar'.format(self.curr_epoch)))

        n_episodes_played = int(self.agent_model.train_steps.data.item())

        optimization_steps = 0
        # for v in optim.state.values():
        #     optimization_steps = max(optimization_steps, int(v['step']))
        for pg in self.optim.param_groups:
            for p in pg['params']:
                optimization_steps = max(optimization_steps, int(self.optim.state[p]['step'][0]))

        print(
            '\nCHECKPOINT REACHED  --  Epochs = {}/{}  --  N Episodes = {}  --  N Optimizations = {}\n'.format(
                self.curr_epoch, int(self.settings.dur), n_episodes_played, optimization_steps
            ),
            flush=True
        )

    def eval_wrapper(self):
        raise NotImplementedError

    def log_eval_results(self, stats, episodes):
        # Save the rollout logs
        # hist_name = 'hist_{}.json'.format(self.rank)
        # with open(os.path.join(self.exp_dir, hist_name), 'a') as save_file:
        #     for log in self.epoch_buffer:
        #         save_file.write(json.dumps(log))
        #     save_file.close()
        self.epoch_buffer = []

        # Save this crop of episodes
        if self.rank == 0:
            dstr = 'eval_{:04d}'.format(self.curr_epoch)
            # c_path = os.path.join(BASE_DIR, self.tag, dstr + '.json')
            c_path = os.path.join(self.exp_dir, dstr + '.json')
            with open(c_path, 'wt') as f:
                json.dump(episodes, f)

        # Save the stats dictionary
        self.eval_stats[str(self.curr_epoch)] = np.array(stats)
        np.savez(os.path.join(self.exp_dir, 'stats_{:02d}.npz'.format(self.rank)), **self.eval_stats)

        # Print out some summary stats
        mean_stats = np.array(stats).mean(axis=0)
        n = len(mean_stats)
        f_str = 'E{:04d} Eval Stats, rank {:02d}:  ' + (', '.join(['{:+07.3f}'] * n))
        print(f_str.format(self.curr_epoch, self.rank, *mean_stats), flush=True)

        # Produce a summary
        total_stats = torch.from_numpy(np.array(stats))
        dist.all_reduce(total_stats)

        if self.rank == 0:
            net_mean = total_stats / dist.get_world_size()
            net_mean = net_mean.mean(dim=0).data.numpy()

            f_str = '\nE{:04d} Eval Stats, AVERAGE:  ' + (', '.join(['{:+07.3f}'] * n))
            print(f_str.format(self.curr_epoch, *net_mean), flush=True)

    def do_cycle_rollouts(self):
        raise NotImplementedError

    def do_update(self):
        batch = self.replay_buffer.make_batch()
        self.optim.zero_grad()
        loss = self.condense_loss(self.agent_model(batch))
        loss.backward()
        for p in self.agent_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data)
                p.grad.data /= dist.get_world_size()
        self.optim.step()

    def do_aux_update(self):
        if not hasattr(self, "aux_optim") or self.aux_optim is None:
            return
        # Prepare batch; all passes are done in inference (eval) mode
        self.agent_model.eval()
        batch = self.replay_buffer.make_batch()

        # Update the model
        self.agent_model.train()
        self.aux_optim.zero_grad()
        loss = self.condense_loss(self.agent_model.forward_aux(batch))
        loss.backward()
        for p in self.agent_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data)
                p.grad.data /= dist.get_world_size()
        self.aux_optim.step()

    def do_cycle_updates(self):
        raise NotImplementedError

    def do_cycle(self):
        raise NotImplementedError

    def init_epoch(self):
        return

    def do_epoch(self):
        self.curr_epoch += 1

        self.init_epoch()

        for _ in range(self.config['cycles_per_epoch']):
            self.do_cycle()

        stats, episodes = self.eval_wrapper()
        self.log_eval_results(stats, episodes)

        if self.rank == 0:
            self.checkpoint()
            self.replay_buffer.profile(0.001)

        dist.barrier()

        for gp in self.optim.param_groups:
            gp['lr'] *= self.config.get("epoch_lr_decay", 1.0)

        if self.aux_optim is not None:
            for gp in self.aux_optim.param_groups:
                gp['lr'] *= self.config.get("epoch_lr_decay", 1.0)


class EpisodicOffPolicyManager(BaseOffPolicyManager):
    @property
    def sample_type(self):
        return "Episodes"

    def rollout_wrapper(self, c_ep_counter):
        raise NotImplementedError

    def do_cycle_rollouts(self):
        cycle_ep_counter = torch.zeros(1)

        for _ in range(self.config["rollouts_per_cycle"]):
            # Run an episode! (wrapper handles logging and saving internally)
            self.rollout_wrapper(cycle_ep_counter)

        dist.all_reduce(cycle_ep_counter)
        self.agent_model.train_steps += cycle_ep_counter.item()

    def do_cycle_updates(self):
        for v in self.agent_model.state_dict().values():
            dist.broadcast(v.data, src=0)

        if self.group_is_ready():
            for _ in range(self.config["updates_per_cycle"]):
                self.do_aux_update()
            for _ in range(self.config["updates_per_cycle"]):
                self.do_update()

            for v in self.agent_model.state_dict().values():
                dist.broadcast(v.data, src=0)

    def do_cycle(self):
        self.agent_model.eval()
        self.do_cycle_rollouts()

        self.agent_model.train()
        self.do_cycle_updates()

        self.agent_model.soft_update()


class OffPolicyManager(BaseOffPolicyManager):
    def __init__(self, rank, config, settings):
        super().__init__(rank, config, settings)
        self.agent_model.agent.reset()

    @property
    def sample_type(self):
        return "Timesteps"

    def env_transitions_wrapper(self, c_step_counter, num_transitions):
        raise NotImplementedError

    def do_cycle_rollouts(self):
        cycle_step_counter = torch.zeros(1)

        # Collect transitions (wrapper handles logging and saving internally)
        self.env_transitions_wrapper(cycle_step_counter, self.config["env_steps_per_cycle"])

        dist.all_reduce(cycle_step_counter)
        self.agent_model.train_steps += cycle_step_counter.item()

    def do_cycle_updates(self):
        for v in self.agent_model.state_dict().values():
            dist.broadcast(v.data, src=0)

        if self.group_is_ready():
            for _ in range(self.config["gradient_steps_per_cycle"]):
                self.do_aux_update()
            for _ in range(self.config["gradient_steps_per_cycle"]):
                self.do_update()

            for v in self.agent_model.state_dict().values():
                dist.broadcast(v.data, src=0)

            self.agent_model.soft_update()
        else:
            print("\nThe group is not ready. Skipping update...\n")

    def do_cycle(self):
        self.agent_model.eval()
        self.do_cycle_rollouts()

        self.agent_model.train()
        self.do_cycle_updates()


class OnPolicyManager:
    def __init__(self, rank, config, settings):
        self.rank = rank
        self.config = config
        self.settings = settings

        # For logging
        # self.tag = os.environ.get('EXP_TAG', self.settings.tag)
        # self.exp_dir = os.path.join(BASE_DIR, self.tag)
        config_path = self.settings.config_path
        exp_name = config_path.split('/')[-1][:-5]
        self.exp_dir = os.path.join(self.settings.log_dir, exp_name)
        self.logger = create_worker_logger(rank, self.exp_dir)

        self.model_path = os.path.join(self.exp_dir, 'model.pth.tar')
        self.optim_path = os.path.join(self.exp_dir, 'optim.pth.tar')
        self.aux_optim_path = os.path.join(self.exp_dir, 'aux_optim.pth.tar')

        # Instantiate a copy of the model and place it on this worker's GPU
        agent_class = agent_classes(self.config['agent_type'], self.config['learner_type'], self.config['train_type'])
        self.agent_model = agent_class(**self.config['agent_params'])

        if os.path.isfile(self.model_path):
            self.agent_model.load_checkpoint(self.model_path)

        for parameter in self.agent_model.state_dict().values():
            dist.broadcast(parameter.data, src=0)

        self.optim = Adam(self.agent_model.parameters(), lr=config['learning_rate'])
        if os.path.isfile(self.optim_path):
            self.optim.load_state_dict(torch.load(self.optim_path))

        if self.agent_model.im is not None or self.agent_model.density is not None:
            self.aux_optim = Adam(self.agent_model.get_aux_optim_params())
            if os.path.isfile(self.aux_optim_path):
                self.aux_optim.load_state_dict(torch.load(self.aux_optim_path))
        else:
            self.aux_optim = None

        # Initialize trackers/counters
        self.time_keeper = {
            'n_rounds': 0,
            'ep_save': 0,
            'last_ep_num': int(self.agent_model.train_steps.data.item()),
            'idle_ep_count': 0,
        }

        self.curr_epoch = 0

        self.eval_stats = {}

    @staticmethod
    def condense_loss(loss_):
        if isinstance(loss_, torch.Tensor):
            return loss_.mean()
        if isinstance(loss_, (list, tuple)):
            net_loss = 0.
            for sub_loss in loss_:
                net_loss += sub_loss.mean()
            return net_loss
        else:
            raise TypeError

    # Set up saving
    def checkpoint(self):
        self.agent_model.save_checkpoint(self.model_path)
        torch.save(self.optim, self.optim_path)
        if self.aux_optim is not None:
            torch.save(self.aux_optim, self.aux_optim_path)

        if self.settings.keep_checkpoints:
            checkpoint_path = os.path.join(self.exp_dir, '{:04d}_model.pth.tar'.format(self.curr_epoch))
            self.agent_model.save_checkpoint(checkpoint_path)

        n_episodes_played = int(self.agent_model.train_steps.data.item())

        optimization_steps = 0
        for pg in self.optim.param_groups:
            for p in pg['params']:
                optimization_steps = max(optimization_steps, int(self.optim.state[p]['step'][0]))

        print(
            '\nCHECKPOINT REACHED  --  Epochs = {}/{}  --  N Episodes = {}  --  N Optimizations = {}\n'.format(
                self.curr_epoch, int(self.settings.dur), n_episodes_played, optimization_steps
            ),
            flush=True
        )

    def rollout_wrapper(self, c_ep_counter):
        raise NotImplementedError

    def eval_wrapper(self):
        raise NotImplementedError

    def log_eval_results(self, stats, episodes):
        # Save this crop of episodes
        if self.rank == 0:
            dstr = 'eval_{:04d}'.format(self.curr_epoch)
            # c_path = os.path.join(BASE_DIR, self.tag, dstr + '.json')
            c_path = os.path.join(self.exp_dir, dstr + '.json')
            with open(c_path, 'wt') as f:
                json.dump(episodes, f)

        # Save the stats dictionary
        self.eval_stats[str(self.curr_epoch)] = np.array(stats)
        np.savez(os.path.join(self.exp_dir, 'stats_{:02d}.npz'.format(self.rank)), **self.eval_stats)

        # Print out some summary stats
        mean_stats = np.array(stats).mean(axis=0)
        n = len(mean_stats)
        f_str = 'E{:04d} Eval Stats, rank {:02d}:  ' + (', '.join(['{:+07.3f}'] * n))
        print(f_str.format(self.curr_epoch, self.rank, *mean_stats), flush=True)

        # Produce a summary
        total_stats = torch.from_numpy(np.array(stats))
        dist.all_reduce(total_stats)

        if self.rank == 0:
            net_mean = total_stats / dist.get_world_size()
            net_mean = net_mean.mean(dim=0).data.numpy()

            f_str = '\nE{:04d} Eval Stats, AVERAGE:  ' + (', '.join(['{:+07.3f}'] * n))
            print(f_str.format(self.curr_epoch, *net_mean), flush=True)

    def update_wrapper(self):
        self.optim.zero_grad()

        loss = 0
        cycle_ep_counter = torch.zeros(1)
        for _ in range(self.config["rollouts_per_update"]):
            this_loss = self.rollout_wrapper(cycle_ep_counter)
            loss += this_loss / self.config["rollouts_per_update"]

        loss.backward()
        for p in self.agent_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data)
                p.grad.data /= dist.get_world_size()
        self.optim.step()

        for v in self.agent_model.state_dict().values():
            dist.broadcast(v.data, src=0)

        dist.all_reduce(cycle_ep_counter)
        self.agent_model.train_steps += cycle_ep_counter.item()

    def do_cycle(self):
        for _ in range(self.config["updates_per_cycle"]):
            self.update_wrapper()

    def init_epoch(self):
        return

    def do_epoch(self):
        self.curr_epoch += 1

        self.init_epoch()

        for _ in range(self.config['cycles_per_epoch']):
            self.do_cycle()

        stats, episodes = self.eval_wrapper()
        self.log_eval_results(stats, episodes)

        if self.rank == 0:
            self.checkpoint()

        dist.barrier()

        for gp in self.optim.param_groups:
            gp['lr'] *= self.config.get("epoch_lr_decay", 1.0)


class PPOManager(OnPolicyManager):
    def do_cycle(self):
        for _ in range(self.config["rollouts_per_cycle"]):
            self.update_wrapper()

    def update_wrapper(self):
        cycle_ep_counter = torch.zeros(1)

        self.rollout_wrapper(cycle_ep_counter)

        if self.aux_optim is not None:
            for u in range(self.config["update_epochs_per_rollout"]):
                for mini_batch in self.agent_model.make_epoch_mini_batches(normalize_advantage=False):
                    self.aux_optim.zero_grad()
                    loss = self.agent_model.forward_aux(mini_batch)
                    loss.backward()
                    for p in self.agent_model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data)
                            p.grad.data /= dist.get_world_size()
                    # _ = clip_grad_norm_(self.agent_model.parameters(), max_norm=0.5)
                    self.aux_optim.step()
            # We collect new data with reward coming from the updated density model. This is slower (but easier to
            # implement) than relabeling previous samples. Note that we only count the rollouts used for updating the
            # policy when reporting data efficiency.
            cycle_ep_counter = torch.zeros(1)
            self.rollout_wrapper(cycle_ep_counter)

        if self.config.get("norm_advantage", False):
            self.agent_model.distributed_advantage_normalization()

        for u in range(self.config["update_epochs_per_rollout"]):
            for mini_batch in self.agent_model.make_epoch_mini_batches(normalize_advantage=False):
                self.optim.zero_grad()
                loss = self.agent_model(mini_batch)
                loss.backward()
                for p in self.agent_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data)
                        p.grad.data /= dist.get_world_size()
                # _ = clip_grad_norm_(self.agent_model.parameters(), max_norm=0.5)
                self.optim.step()

        for v in self.agent_model.state_dict().values():
            dist.broadcast(v.data, src=0)

        dist.all_reduce(cycle_ep_counter)
        self.agent_model.train_steps += cycle_ep_counter.item()


