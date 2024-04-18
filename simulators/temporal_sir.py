import torch
import time
from sbi.utils.torchutils import *


class TemporalSIR(object):
    def __init__(self,
                 name="temporal_sir",
                 population=1000,
                 prior_lower=None,
                 prior_upper=None,
                 theta_dim=2,
                 x_dim=12,
                 num_bins=10):

        if prior_lower is None:
            prior_lower = [0.1, 0.1]
        if prior_upper is None:
            prior_upper = [1.0, 1.0]

        self.name = name
        self.N = population
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.num_bins = num_bins
        self.prior = Uniform(torch.tensor(prior_lower), torch.tensor(prior_upper))
                      # Uniform(prior_lower[1] * torch.ones(1), prior_upper[1] * torch.ones(1))]

    def __call__(self, thetas):
        beta = thetas[0]
        gamma = thetas[1]

        I = torch.tensor(1.0)  # infected individuals
        S = torch.tensor(self.N - 1.0)  # susceptible individuals
        t = torch.tensor(0.0)  # time
        times = [t.item()]
        types = [1]  # 1 for infection, 2 for removal

        while I > 0:
            rate = (beta / self.N) * I * S + gamma * I
            t += torch.distributions.Exponential(1 / rate).sample()
            times.append(t.item())

            if torch.rand(1) < (beta * S) / ((beta * S) + self.N * gamma):
                I += 1
                S -= 1
                types.append(1)
            else:
                I -= 1
                types.append(2)

        # removal_times = [times[i] - min([times[j] for j in range(len(types)) if types[j] == 2]) for i in
        #                  range(len(times))
        #                  if types[i] == 2]
        T = times[-1]
        removal_times = torch.tensor([times[i] for i in range(len(times)) if types[i] == 2])

        removals_per_bin = torch.histc(removal_times, bins=self.num_bins, min=0, max=T)
        final_size = self.N - S.item()

        x = torch.cat([removals_per_bin, torch.tensor([final_size]), torch.tensor([T])])

        return x


if __name__ == "__main__":
    simulator = TemporalSIR()
    N = 10
    # beta = torch.linspace(0.01, 0.5, N)
    beta = 0.9
    gamma = 0.5
    # gamma = torch.linspace(0.01, 0.5, N)

    result = simulator([beta, gamma])


