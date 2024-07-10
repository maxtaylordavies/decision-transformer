from typing import Sequence

import numpy as np

from .graph_env import GraphEnv

def make_empty_dataset():
    return {
        "obss": [],
        "acts": [],
        "rewards": [],
        "dones": [],
        "infos": [],
    }

def extend_dataset(dataset, new_data):
    for key in dataset:
        if isinstance(new_data[key], Sequence):
            dataset[key].extend(new_data[key])
        else:
            dataset[key].append(new_data[key])

def dataset_to_numpy(dataset):
    return {
        key: np.array(dataset[key]) if key != "info" else dataset[key] for key in dataset
    }

def compute_returns_to_go(dataset, gamma=1.0):
    rtgs = np.zeros(len(dataset["rewards"]))

    ep_ends = np.nonzero(dataset["dones"])[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    assert(len(ep_starts) == len(ep_ends))

    for start, end in zip(ep_starts, ep_ends):
        rtg = 0
        for i in range(end, start - 1, -1):
            rtg = dataset["rewards"][i] + (gamma * rtg)
            rtgs[i] = rtg

    return rtgs

def collect_random_rollout(env, seed=0, max_length=100):
    rollout = make_empty_dataset()

    obs, info = env.reset(seed)
    t, done = 0, False
    while not done:
        action = env.np_random.choice(env.action_space)
        new_obs, r, done, info = env.step(action)
        done = done or t >= max_length
        extend_dataset(rollout, {"obss": obs, "acts": action, "rewards": r, "dones": done, "infos": info})
        obs, t = new_obs, t + 1

    return rollout

def generate_random_dataset(n, p, num_graphs, rollouts_per_graph, seed=0, max_length=100):
    dataset = make_empty_dataset()
    for i in range(num_graphs):
        env, seed_i = GraphEnv(n, p), seed + i
        for j in range(rollouts_per_graph):
            rollout = collect_random_rollout(env, seed=seed_i, max_length=max_length)
            extend_dataset(dataset, rollout)

    dataset = dataset_to_numpy(dataset)
    dataset["rtgs"] = compute_returns_to_go(dataset)
    return dataset
