import numpy as np
import os
from scipy.special import gammaln
import torch


class CMPGM(object):
    def __init__(self,
                 name="cmpgm",
                 dim=4,
                 theta_ij=0.1,
                 theta_0j=0.99,
                 inference=None,
                 k_mixture=None,
                 k=None):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.dim = dim

        self.tri = np.triu_indices(dim)
        self.dim_total = dim + int(dim * (dim - 1) / 2) + dim

        self.diag_idx = (np.eye(dim, dtype=bool))[self.tri[0], self.tri[1]]
        self.offd_idx = (~np.eye(dim, dtype=bool))[self.tri[0], self.tri[1]]

        # Fix the interaction parameters (theta_ij) and dispersion parameters (theta_0j)
        self.theta_ij = theta_ij
        self.theta_0j = theta_0j

        self.theta_true = np.load(os.path.join(f'{self.path}/../data/cmpgm_theta_obs_dim{self.dim}.npy'))
        self.x_obs = np.load(os.path.join(f'{self.path}/../data/cmpgm_x_obs_dim{self.dim}.npy'))


    def _uloglikelihood(self, param, x):
        xa = np.abs(x)
        xx = np.outer(xa, xa)
        np.fill_diagonal(xx, xa)
        t0 = xx[self.tri[0], self.tri[1]]

        t1 = np.dot(t0[self.diag_idx], param)

        t2 = np.dot(t0[self.offd_idx], np.ones_like(t0[self.offd_idx]) * self.theta_ij)

        t3 = np.dot(gammaln(xa + 1), np.ones_like(gammaln(xa + 1)) * self.theta_0j)

        return t1 - t2 - t3

    def poisson_sample(self, lam, n):
        d = len(lam)
        samples = np.zeros((n, d), dtype=int)

        for i in range(n):
            for j in range(d):
                L = np.exp(-lam[j])
                k = 0
                p = 1.0
                while p > L:
                    k += 1
                    u = np.random.uniform(0, 1)
                    p *= u
                samples[i, j] = k - 1
        return samples

    def __call__(self, param, num_samples):
        proposal_samples = self.poisson_sample(param, n=num_samples)  # hand-crafted implementation

        weights = np.zeros(num_samples)

        for i in range(num_samples):
            # Calculate the log-likelihood for the target distribution
            logl_target = self._uloglikelihood(param, proposal_samples[i])

            # Calculate the log-likelihood for the proposal distribution
            logl_proposal = np.sum(proposal_samples[i] * np.log(param) - param - gammaln(proposal_samples[i] + 1))

            # Weight is the exponent of the difference between the target and proposal log-likelihoods
            weights[i] = np.exp(logl_target - logl_proposal)

        total_weight = np.sum(weights)
        weights /= total_weight

        # Resample according to the normalized weights
        resampled_indices = np.random.choice(num_samples, size=num_samples, replace=True, p=weights)
        resampled_samples = proposal_samples[resampled_indices]

        marginal_means = np.mean(resampled_samples, axis=0)

        # Calculate marginal variances for each dimension
        marginal_variances = np.var(resampled_samples, axis=0)

        return np.concatenate([marginal_means, marginal_variances], axis=0)


class CMPGM_MCMC():
    def __init__(self, dim):
        super(CMPGM_MCMC, self).__init__()
        self.dim = dim
        self.tri = torch.triu_indices(dim, dim)
        self.dim_range = range(dim)
        self.dim_total = dim + int(dim * (dim - 1) / 2) + dim

        self.diag_idx_0 = (torch.eye(dim).bool())[self.tri[0], self.tri[1]]
        self.offd_idx_0 = (~torch.eye(dim).bool())[self.tri[0], self.tri[1]]
        self.addt_idx_0 = torch.zeros(dim).bool()
        self.diag_idx = torch.cat((self.diag_idx_0, self.addt_idx_0))
        self.offd_idx = torch.cat((self.offd_idx_0, self.addt_idx_0))
        self.addt_idx = torch.cat(((self.diag_idx_0 * False), (~self.addt_idx_0)))

        self.initp = torch.distributions.Poisson(2 * torch.ones(self.dim))
        self.unifp = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.jumpsize = 3

    def sample(self, param, num_sample, num_burnin):
        state_old = self.initp.sample().float()
        chain = torch.zeros(num_burnin + num_sample, self.dim).float()

        for i in range(num_burnin + num_sample):
            move = (2.0 * torch.randint(0, 2, (self.dim,)) - 1.0) * torch.randint(1, self.jumpsize + 1, (self.dim,))
            state_new = state_old + move

            logl_new = self._uloglikelihood(state_new, param)
            logl_old = self._uloglikelihood(state_old, param)

            if self.unifp.sample().log() < min(0, logl_new - logl_old):
                state_old = state_new
            chain[i] = state_old.abs()

        return chain[num_burnin:, ]

    def _uloglikelihood(self, x, param):
        ad = 2 ** ((x != 0).float().sum())
        xa = x.abs()
        xt = xa.unsqueeze(-1)
        xx = xt @ xt.t()
        xx[self.dim_range, self.dim_range] = xa

        t0 = xx[self.tri[0], self.tri[1]]
        t1 = t0[self.diag_idx_0] @ param[self.diag_idx]
        t2 = t0[self.offd_idx_0] @ (param[self.offd_idx] ** 2)
        t3 = torch.lgamma(xa + 1) @ (param[self.addt_idx] ** 2)

        return t1 - t2 - t3 - torch.log(ad)
