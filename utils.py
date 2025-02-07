import torch
import numpy as np
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import scipy.spatial.distance as distance


# Acceptance probability function for Gamma simulator's linear cost. Penalty function is g(z) = z^k
def A_theta_Gamma(theta, k, alpha, beta, prior_start):
    return (alpha * prior_start + beta)**k / (alpha * theta + beta)**k


def cost_linear(theta, alpha, beta):
    return alpha * theta + beta


def MMD_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    z = np.concatenate((x, y), axis=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * np.sum(kxx) - (2 / (m * n)) * np.sum(kxy) + (1 / n ** 2) * np.sum(kyy)


def MMD_weighted(x, y, w, lengthscale):
    #     """ Optimally weighted squared MMD estimate between samples x_i ~ P and y_i ~ Q
    #     """

    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()
        w = np.array(w, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    xy = np.concatenate((x, y), axis=0)

    K = kernel_matrix(xy, xy, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    # first sum
    sum1 = np.matmul(np.matmul(w.transpose(), kxx), w)

    # second sum
    sum2 = np.sum(np.matmul(w.transpose(), kxy))

    # third sum
    sum3 = (1 / n ** 2) * np.sum(kyy)

    return sum1 - (2 / (n)) * sum2 + sum3

def median_heuristic(y):
    a = distance.cdist(y, y, 'sqeuclidean')
    return np.sqrt(np.median(a / 2))


# Function to compute the kernel Gram matrix
def kernel_matrix(x, y, l):
    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    return np.exp(-(1 / (2 * l ** 2)) * distance.cdist(x, y, 'sqeuclidean'))


def gamma_sampler(theta, n=500):
    data = torch.zeros(2)
    # split theta into integer and decimal parts
    i_theta = theta // torch.ones(1)
    p = int(torch.sum(i_theta))

    # sample uniforms
    unif = torch.rand(n, p)

    # generate samples
    # split theta into integer and decimal parts
    i_theta, d_theta = theta // torch.ones(1), theta % torch.ones(1)
    i_theta = i_theta.int()

    # initialise
    utild = torch.zeros([n, 1])
    x = torch.zeros((n, 2))

    # logs on uniforms
    logunif = torch.log(unif)

    # Repeat if we get inf when taking log
    while torch.isinf(logunif).sum() > 0:
        unif = torch.rand(n, p)
        logunif = torch.log(unif)

    # get \tilde{u}
    j = 0
    for i in range(1):
        s = torch.zeros(n)
        if i_theta[i] != 0:
            for k in range(i_theta[i]):
                s += logunif[:, k + j]
            utild[:, i] = -s
        if d_theta[i] != 0:
            utild[:, i] += torch.tensor(np.random.gamma(d_theta[i], 1, n))
        j += int(i_theta[i])

    data[0] = utild[:, 0].mean()
    data[1] = utild[:, 0].std()

    return data


class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        if train_x is None:
            train_x = torch.zeros(1)
        if train_y is None:
            train_y = torch.zeros(1)

        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def calc_acc_prob(model, likelihood, theta, prior_start, k):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        cost = likelihood(model(theta)).mean

        lower_cost = likelihood(model(prior_start)).mean
    return (lower_cost ** k) / (cost**k)


def calc_acc_prob_linear(a_1, a_2, b, theta, prior_start, k):
    # cost = a_1 * theta[:, 0] + a_2 * theta[:, 1] + b
    # lower_cost = a_1 * prior_start[:, 0] + a_2 * prior_start[:, 1] + b
    cost = cost_linear_step(a_1, a_2, b, theta)
    lower_cost = cost_linear_step(a_1, a_2, b, prior_start)
    return (lower_cost ** k) / (cost**k)


def cost_linear_step(a_1, a_2, b, theta):
    cost = a_1 * theta[:, 0] + a_2 * theta[:, 1] + b
    return cost if cost > 0.03 else 0.03

def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def rejection_ABC(s_obs, param, sumStats, epsilon_distance):
    M = param.shape[0]

    norm_factor = mad(sumStats, axis=0)

    norm_sumStats = sumStats / norm_factor
    norm_s_obs = s_obs / norm_factor

    distance = np.linalg.norm(norm_sumStats - norm_s_obs, axis=1)

    d_ = np.sort(distance)

    indices = np.where(distance < epsilon_distance)[0]

    posterior_samples = param[indices, :]

    return posterior_samples, indices


if __name__ == "__main__":
    theta = torch.tensor([2.0])
    data = gamma_sampler(theta)
    print(data)