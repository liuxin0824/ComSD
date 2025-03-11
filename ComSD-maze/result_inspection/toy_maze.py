# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT


import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from dist_train.workers.utils import ReplayBuffer
from .experiment import Experiment, EXPERIMENT_DIR
from agents.maze_agents.toy_maze.skill_discovery.edl import VQVAEDiscriminator
import seaborn as sns
import numpy as np 

NUM_TRAJECTORIES = 20
TRAJECTORY_KWARGS = dict(alpha=0.2, linewidth=2)

SAVEFIG_KWARGS = dict(bbox_inches='tight', transparent=True)

ENV_LIMS = dict(
    square_a=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_bottleneck=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
    square_corridor=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_corridor2=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_tree=dict(xlim=(-6.55, 6.55), ylim=(-6.55, 0.55), x=(-6.5, 6.5), y=(-6.5, 0.5))
)


def load_exp_data(exp_name, notebook_mode=True):
    exp = Experiment(exp_name, notebook_mode=notebook_mode)
    agent = exp.learner.agent
    if agent.skill_n <= 10:
        cmap = plt.get_cmap('tab10')
    elif 10 < agent.skill_n <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('plasma', agent.skill_n)
    return exp, cmap


def config_subplot(ax, maze_type=None, title=None, extra_lim=0., fontsize=14, exp=None):
    if maze_type is None and exp is not None:
        maze_type = exp.learner.agent.env.maze_type

    if maze_type is not None:
        env_config = ENV_LIMS[maze_type]
        ax.set_xlim(env_config["xlim"][0] - extra_lim, env_config["xlim"][1] + extra_lim)
        ax.set_ylim(env_config["ylim"][0] - extra_lim, env_config["ylim"][1] + extra_lim)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)


def play_episode(agent, skill, do_eval, reset_dict={}):
    agent.reset(**reset_dict)
    agent.curr_skill = agent.curr_skill * 0 + skill
    while not agent.env.is_done:
        agent.step(do_eval)


def _plot_all_skills(exp, cmap, ax=None, reset_dict=None, alpha=1., linewidth=1.):
    agent = exp.learner.agent
    agent.env.maze.plot(ax)

    if reset_dict is None:
        reset_dict = agent.env.sibling_reset  # fix s_0 across trajectories and skills
    cclr = sns.color_palette("Spectral",agent.skill_n)
    for skill_idx in range(agent.skill_n):
        # Collect rollout
        play_episode(agent, skill_idx, do_eval=False, reset_dict=reset_dict)
        # Plot trajectory    
        ax.plot(*agent.rollout, label="Skill #{}".format(skill_idx), color=cmap(skill_idx) if agent.skill_n <= 20 else cclr[skill_idx], alpha=alpha,
                linewidth=linewidth, zorder=10)
    # Mark initial state with a dot
    ax.plot(agent.rollout[0][0], agent.rollout[1][0], marker='o', markersize=8, color='black', zorder=11)


def plot_all_skills(exp, cmap, ax=None, reset_dict=None, notebook_mode=True, desc=None, figsize=(5, 5), **kwargs):
    desc = desc or "Trajectories"
    tqdm_ = tqdm_notebook if notebook_mode else tqdm

    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        return_ax = False

    for _ in tqdm_(range(NUM_TRAJECTORIES), desc=desc, disable=False, leave=True, total=NUM_TRAJECTORIES):
        _plot_all_skills(exp, cmap, ax, reset_dict=reset_dict, **TRAJECTORY_KWARGS)

    config_subplot(ax, exp=exp, **kwargs)

    if return_ax:
        return ax


def load_smm_buffer(exp_name, epoch=None, notebook_mode=True):
    exp, _ = load_exp_data(exp_name, notebook_mode=notebook_mode)

    valid_epochs = sorted([int(d.split("_")[0]) for d in os.listdir(exp.exp_dir) if "replay_buffer" in d])

    if epoch is None:
        epoch = valid_epochs[-1]

    assert epoch in valid_epochs, "Replay buffer for epoch {} not found. Found epochs: {}".format(epoch, valid_epochs)

    # Load buffer
    config = exp.get_config()
    config['load_buffer'] = True
    config['buffer_path'] = os.path.join(exp.exp_dir, "{:04d}_replay_buffer".format(epoch))
    buffer = ReplayBuffer(None, config, verbose_load=True)

    # Get all training samples from the buffer
    batch_size = buffer.batch_size
    buffer.batch_size = int(buffer.size)
    dataset = buffer.make_batch(normalize=False)['next_state']
    buffer.batch_size = batch_size

    return exp, dataset


def visualize_smm_samples(exp_name, epoch=None, ax=None, sample_frac=1., figsize=(5, 5), **kwargs):
    exp, dataset = load_smm_buffer(exp_name, epoch)
    env = exp.learner.agent.env

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    env.maze.plot(ax)
    config_subplot(ax, exp=exp, **kwargs)

    num_samples = int(sample_frac * dataset.shape[0])
    _ = ax.scatter(dataset[:num_samples, 0], dataset[:num_samples, 1], s=3, marker='o')


def load_vqvae(exp_name, verbose=False):
    exp_dir = os.path.join(EXPERIMENT_DIR, exp_name)

    # Load config
    config = json.load(open(os.path.join(exp_dir, "config.json")))
    if verbose:
        print(json.dumps(config, indent=2))

    # Load model
    model = VQVAEDiscriminator(state_size=2, **config['vae_args'])
    model.load_state_dict(torch.load(os.path.join(exp_dir, "model.pth.tar")))
    model.eval()

    # Load train loss
    loss = json.loads(json.load(open(os.path.join(exp_dir, "loss.json"))))

    return model, config, loss
