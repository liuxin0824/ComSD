import torch
import numpy as np
import argparse
import sys
from tqdm import tqdm
from result_inspection.experiment import Experiment, EXPERIMENT_DIR
sys.path.append("../")

# bash
# python Entropy.py --algo cic_mi 
# python Entropy.py --algo contrastive_mi_0.5 
# python Entropy.py --algo contrastive_mi_0.01 
# python Entropy.py --algo reverse_mi
# python Entropy.py --algo forward_mi

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--algo', type=str, default='contrastive_mi')         # cic_mi, contrastive_mi_0.5
parser.add_argument('--env', type=str, default='square_maze')     
# knn parameters
parser.add_argument('--knn_k', type=int, default='16')     
parser.add_argument('--knn_avg', type=bool, default=True)     
parser.add_argument('--rms', type=bool, default=True)     
parser.add_argument('--knn_clip', type=float, default=0.0005)     
args = parser.parse_args()

env = args.env
algo = args.algo

def load_exp_data(exp_name, notebook_mode=True):
    exp = Experiment(exp_name, notebook_mode=notebook_mode)
    agent = exp.learner.agent
    return exp

exp = load_exp_data(f"{args.env}/{args.algo}", notebook_mode=False)
agent = exp.learner.agent


def play_episode(agent, skill, do_eval, reset_dict={}):
    agent.reset(**reset_dict)
    agent.curr_skill = agent.curr_skill * 0 + skill
    while not agent.env.is_done:
        agent.step(do_eval)


def obtain_algo_data(NUM_TRAJECTORIES=10):
    agent = exp.learner.agent
    reset_dict = agent.env.sibling_reset

    states_list = []
    for i in tqdm(range(NUM_TRAJECTORIES)):
        for skill_idx in range(agent.skill_n):
            # Collect rollout
            play_episode(agent, skill_idx, do_eval=False, reset_dict=reset_dict)
            states_list.append(np.hstack([agent.rollout[0], agent.rollout[1]]).reshape(-1))  # (102,)

    return np.array(states_list)


def compute_knn_entropy(source, target, knn_avg=True):
    b1, b2 = source.shape[0], target.shape[0]
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
    reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward))  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)

    return reward.mean()


if __name__ == '__main__':
    data = obtain_algo_data()
    print("collected data:", data.shape)
    data = torch.from_numpy(data)

    entropy = compute_knn_entropy(source=data, target=data)
    print("algo:", algo, ", entropy:", entropy)

