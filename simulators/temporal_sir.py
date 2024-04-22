import torch
from sbi.utils.torchutils import *
import torch.distributions as dist


class TemporalSIR(object):
    def __init__(self,
                 name="temporal_sir",
                 population=1000,
                 beta_lower=0.1,
                 beta_upper=1.0,
                 gamma_lower=0.1,
                 gamma_upper=1.0,
                 theta_dim=2,
                 x_dim=12,
                 num_bins=10,
                 prior_start=None):

        self.name = name
        self.N = population
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.num_bins = num_bins
        self.prior = [Uniform(torch.tensor([beta_lower]), torch.tensor([beta_upper])),
                      Uniform(torch.tensor([gamma_lower]), torch.tensor([gamma_upper]))]
        # dist.Gamma(concentration=torch.tensor([gamma_concentration]), rate=torch.tensor([gamma_rate]))

    def sample_theta(self, size):
        beta = self.prior[0].sample(size).reshape(-1, 1)
        gamma = self.prior[1].sample(size).reshape(-1, 1)

        return torch.cat([beta, gamma], dim=1)

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


