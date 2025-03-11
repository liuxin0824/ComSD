# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT


import os
import json
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from agents import agent_classes
from collections import OrderedDict
from tqdm import tqdm, tqdm_notebook


# Find logdir. This can be set through ROOT_DIR (which should point to 'edl/') or will otherwise use a default value.
root_dir = os.environ.get("ROOT_DIR", os.path.join(str(pathlib.Path(__file__).parent.absolute().resolve()), ".."))
EXPERIMENT_DIR = os.path.join(root_dir, "logs")


def make_from_config(config_path):
    config = json.load(open(config_path))
    ac = agent_classes(config['agent_type'], config['learner_type'], config['train_type'])
    return ac(**config['agent_params'])


class EvalStats:
    def __init__(self, name, ax_height=3.3):
        self.name = name
        self.ax_height = ax_height

        self.raw_stat_dict = OrderedDict()
        self.epoch_ranks = {}
        self.load()

    def load(self):
        exp_dir = os.path.join(EXPERIMENT_DIR, self.name)
        stat_files = sorted([f for f in os.listdir(exp_dir) if 'stats_' in f])
        for f in stat_files:
            try:
                stat_dict = np.load(os.path.join(exp_dir, f))
            except:
                print('Bad time loading. Try again soon.')
                return
            for epoch, stat_array in stat_dict.items():
                epoch = int(epoch)
                if epoch not in self.raw_stat_dict:
                    self.raw_stat_dict[epoch] = []
                self.raw_stat_dict[epoch].append(stat_array)

    @property
    def stat_dict(self):
        stat_dict = OrderedDict()
        keys = sorted([k for k in self.raw_stat_dict.keys()])
        for k in keys:
            stat_dict[k] = np.concatenate(self.raw_stat_dict[k])
        return stat_dict

    @property
    def averages(self):
        return np.stack([v.mean(axis=0) for v in self.stat_dict.values()])

    def plot_stat(self, stat_idx, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(16, self.ax_height))
        y = self.averages[:, stat_idx]
        x = np.arange(len(y)) + 1
        ax.plot(x, y, 'o-')
        ax.grid(b=True)
        return ax

    def plot_all(self, axes=None):
        ys = self.averages
        n = ys.shape[1]
        if axes is None:
            _, axes = plt.subplots(n, 1, figsize=(16, self.ax_height * n), sharex=True)
        for i, ax in enumerate(axes):
            y = self.averages[:, i]
            x = np.arange(len(y)) + 1
            ax.plot(x, y, 'o-')
            ax.grid(b=True)
        return axes


class EvalSaves:
    def __init__(self, name, ax_height=3.3, notebook_mode=False):
        self.name = name
        self.ax_height = ax_height

        self.raw_dict = OrderedDict()
        self.tqdm = tqdm_notebook if notebook_mode else tqdm
        self.load()

        self._epc = int(self.epochs)
        self._itr = 0

    def load(self):
        exp_dir = os.path.join(EXPERIMENT_DIR, self.name)
        eval_files = sorted([f for f in os.listdir(exp_dir) if 'eval_' in f])
        add_me = []
        for f in eval_files:
            _, epoch = f.split('_')
            epoch = epoch.rsplit('.', 1)[0]
            epoch = int(epoch)
            if epoch not in self.raw_dict:
                add_me.append([epoch, f])

        if not add_me:
            print('No new eval files found')
            return

        for epoch, f in self.tqdm(add_me):
            epoch_eps = json.load(open(os.path.join(exp_dir, f)))
            epoch_dict = {}
            for itr, ep in epoch_eps.items():
                itr = int(itr)
                better_ep = {}
                for k in ep[0].keys():
                    better_ep[k] = np.array([t[k] for t in ep])
                epoch_dict[itr] = better_ep
            self.raw_dict[epoch] = epoch_dict
        self.set_episode(epoch=-1, itr=0)

    @property
    def epochs(self):
        return max([k for k in self.raw_dict.keys()])

    def set_episode(self, epoch=None, itr=None):
        if epoch is not None:
            if epoch == -1:
                self._epc = int(self.epochs)
            else:
                assert epoch in self.raw_dict
                self._epc = int(epoch)
        if itr is not None:
            assert itr in self.raw_dict[self._epc]
            self._itr = int(itr)

    @property
    def episode(self):
        return self.raw_dict[self._epc][self._itr]

    def get_field(self, field):
        return self.episode[field]

    def get_epoch_field(self, field):
        orig_itr = int(self._itr)
        fields = []
        sorted_keys = sorted([k for k in self.raw_dict[self._epc].keys()])
        for k in sorted_keys:
            self.set_episode(itr=k)
            fields.append(self.get_field(field))
        self.set_episode(itr=orig_itr)
        return fields

    def plot_field(self, field, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(16, self.ax_height))
        ax.plot(self.get_field(field), **kwargs)
        ax.set_title('E{}, I{}'.format(self._epc, self._itr))
        ax.set_ylabel(field, fontsize=15)
        ax.grid(b=True)
        return ax


class Experiment:
    def __init__(self, exp_name, ax_height=3.3, load_stats=True, load_evals=True, load_model=True, notebook_mode=False):
        self.exp_name = str(exp_name)
        self._load_stats = bool(load_stats)
        self._load_evals = bool(load_evals)
        self._load_model = bool(load_model)
        self.ax_height = float(ax_height)

        self.tqdm = tqdm_notebook if notebook_mode else tqdm
        self.exp_dir = os.path.join(EXPERIMENT_DIR, self.exp_name)

        # Load the agent
        if self._load_model:
            self.learner = make_from_config(os.path.join(self.exp_dir, 'config.json'))
            self.learner.load_state_dict(torch.load(os.path.join(self.exp_dir, 'model.pth.tar')))
        else:
            self.learner = None

        # Initialize the eval stats tracker
        if self._load_stats:
            self.stats = EvalStats(self.exp_name, self.ax_height)
        else:
            self.stats = None

        # Initialize the evaluation episodes tracker
        if self._load_evals:
            self.evals = EvalSaves(self.exp_name, self.ax_height, notebook_mode=notebook_mode)
        else:
            self.evals = None

    @property
    def name(self):
        return str(self.exp_name)

    def load(self, epoch=None):
        if self._load_model:
            if epoch is None:
                model_path = os.path.join(self.exp_dir, 'model.pth.tar')
            else:
                epoch = int(epoch)
                model_path = os.path.join(self.exp_dir, '{:04d}_model.pth.tar'.format(epoch))
                assert os.path.isfile(model_path)
            self.learner.load_state_dict(torch.load(model_path))

        if self._load_stats:
            self.stats.load()

        if self._load_evals:
            self.evals.load()

    def valid_load_epochs(self):
        checkpoints = [int(f.split('_')[0]) for f in os.listdir(self.exp_dir) if f.endswith('_model.pth.tar')]
        return sorted(checkpoints) + [None]

    def test(self, n):
        if not self._load_model:
            return
        outcomes = []
        for _ in self.tqdm(range(n)):
            self.learner.play_episode(do_eval=True)
            # outcomes.append(self.learner.agent.episode[-1]['reward'] == 0)
            outcomes.append(sum([e['reward'] for e in self.learner.agent.episode]))
        return np.mean(outcomes)

    def set_episode(self, epoch=None, itr=None):
        if not self._load_evals:
            return
        self.evals.set_episode(epoch, itr)

    def add_model(self, load_checkpoint=True):
        if self._load_model:
            return
        self._load_model = True
        self.learner = make_from_config(os.path.join(self.exp_dir, 'config.json'))
        if load_checkpoint:
            self.learner.load_state_dict(torch.load(os.path.join(self.exp_dir, 'model.pth.tar')))

    def add_stats(self):
        if self._load_stats:
            return
        self._load_stats = True
        self.stats = EvalStats(self.exp_name, self.ax_height)

    def add_evals(self):
        if self._load_evals:
            return
        self._load_evals = True
        self.evals = EvalSaves(self.exp_name, self.ax_height)

    @property
    def epochs(self):
        return self.evals.epochs if self._load_evals else None

    @property
    def episode(self):
        return self.evals.episode if self._load_evals else None

    @property
    def stat_dict(self):
        return self.stats.stat_dict if self._load_stats else None

    @property
    def averages(self):
        return self.stats.averages if self._load_stats else None

    @property
    def summary_keys(self):
        return self.learner.ep_summary_keys if self.learner is not None else ()

    def get_field(self, field):
        if not self._load_evals:
            return
        return self.episode[field]

    def plot_field(self, *args, **kwargs):
        if not self._load_evals:
            return
        self.evals.plot_field(*args, **kwargs)

    def plot_stat(self, *args, **kwargs):
        if not self._load_stats:
            return
        self.stats.plot_stat(*args, **kwargs)

    def plot_all(self, *args, **kwargs):
        if not self._load_stats:
            return
        self.stats.plot_all(*args, **kwargs)

    def get_config(self):
        return json.load(open(os.path.join(self.exp_dir, 'config.json')))

    def print_config(self):
        print(json.dumps(self.get_config(), indent=2))
