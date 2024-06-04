import torch
from sbi.utils.torchutils import *
import time

class DynamicGraph(object):
    def __init__(self, name="dynamic_graph",
                 alpha_lower=0.0,
                 alpha_upper=1.0,
                 beta_lower=0.0,
                 beta_upper=1.0,
                 graph_size=20,
                 theta_dim=2,
                 x_dim=400,
                 prior_start=None):
        self.name = name
        self.theta_dim = theta_dim
        self.graph_size = graph_size
        self.x_dim = x_dim
        self.prior = [Uniform(torch.tensor([alpha_lower]), torch.tensor([alpha_upper])),
                      Uniform(torch.tensor([beta_lower]), torch.tensor([beta_upper]))]

        assert self.x_dim == self.graph_size * self.graph_size, "x_dim must be equal to graph_size * graph_size"

    def sample_theta(self, size):
        alpha = self.prior[0].sample(size).reshape(-1, 1)
        beta = self.prior[1].sample(size).reshape(-1, 1)

        return torch.cat([alpha, beta], dim=1)

    def __call__(self, thetas):
        alpha = thetas[0]
        beta = thetas[1]
        N = self.graph_size

        y = torch.zeros(N, N)

        for j in range(N):
            i = 0
            while i < j:
                prob = alpha / (alpha + beta)
                outcome = int(torch.rand(1) < prob)
                y[i, j] = outcome
                y[j, i] = outcome
                i += 1

        for j in range(N):
            i = 0
            while i < j:
                if y[i, j] == 0:
                    prob = alpha
                else:
                    prob = 1 - beta
                outcome = int(torch.rand(1) < prob)
                y[i, j] = outcome
                y[j, i] = outcome
                i += 1

        return y


if __name__ == "__main__":
    # simulator = DynamicGraph()
    # thetas = torch.tensor([0.5, 0.9])
    # data = simulator(thetas)
    # print(data)

    import numpy as np
    import matplotlib.pyplot as plt


    def random_walk_1d(diffusion_rate, max_distance):
        position = 0
        steps = 0
        while abs(position) < max_distance:
            step = np.random.choice([-1, 1]) * diffusion_rate
            position += step
            steps += 1
        return steps


    # params
    diffusion_rate = 0.1
    max_distance = 100

    # model
    st = time.time()
    steps_needed = random_walk_1d(diffusion_rate, max_distance)
    et = time.time()
    print(f"Time taken: {et - st:.2f}s")
    print(f"Steps needed to reach {max_distance}:", steps_needed)

    positions = [0]
    for _ in range(steps_needed):
        positions.append(positions[-1] + np.random.choice([-1, 1]) * diffusion_rate)
    plt.plot(positions)
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.title('Random Walk 1D')
    plt.grid(True)
    plt.show()