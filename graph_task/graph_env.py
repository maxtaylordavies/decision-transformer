from networkx import Graph, gnp_random_graph
import numpy as np


class GraphEnv:
    n: int
    p: float
    graph: Graph
    start_node: int
    curr_node: int
    goal_node: int
    np_random: np.random.Generator

    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p

    def reset(self, seed: int):
        self.np_random = np.random.default_rng(seed)
        self.graph = gnp_random_graph(self.n, self.p, seed=seed)
        self.start_node = self.np_random.integers(0, self.n)
        self.goal_node = self.np_random.choice([i for i in range(self.n) if i != self.start_node])
        self.curr_node = self.start_node
        return self.curr_node, self.get_info()

    def step(self, action: int):
        assert action >= 0 and action < self.n
        if action in self.graph[self.curr_node]:
            self.curr_node = action

        r, done, info = -1, False, self.get_info()
        if self.curr_node == self.goal_node:
            r, done = 0, True

        return self.curr_node, r, done, info

    def get_info(self):
        return {
            "start_node": self.start_node,
            "curr_node": self.curr_node,
            "goal_node": self.goal_node,
        }

    @property
    def action_space(self):
        return self.n

    @property
    def observation_space(self):
        return self.n

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges
