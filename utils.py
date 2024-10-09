import torch
import numpy as np
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import scipy.spatial.distance as distance


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


def update_plot_style():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        'font.family': 'times',
        'font.size': 14.0,
        'lines.linewidth': 2,
        'lines.antialiased': True,
        'axes.facecolor': 'fdfdfd',
        'axes.edgecolor': '777777',
        'axes.linewidth': 1,
        'axes.titlesize': 'medium',
        'axes.labelsize': 'medium',
        'axes.axisbelow': True,
        'xtick.major.size': 0,  # major tick size in points
        'xtick.minor.size': 0,  # minor tick size in points
        'xtick.major.pad': 6,  # distance to major tick label in points
        'xtick.minor.pad': 6,  # distance to the minor tick label in points
        'xtick.color': '333333',  # color of the tick labels
        'xtick.labelsize': 'medium',  # fontsize of the tick labels
        'xtick.direction': 'in',  # direction: in or out
        'ytick.major.size': 0,  # major tick size in points
        'ytick.minor.size': 0,  # minor tick size in points
        'ytick.major.pad': 6,  # distance to major tick label in points
        'ytick.minor.pad': 6,  # distance to the minor tick label in points
        'ytick.color': '333333',  # color of the tick labels
        'ytick.labelsize': 'medium',  # fontsize of the tick labels
        'ytick.direction': 'in',  # direction: in or out
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 1,
        'legend.fancybox': True,
        'legend.fontsize': 'Small',
        'figure.figsize': (2.5, 2.5),
        'figure.facecolor': '1.0',
        'figure.edgecolor': '0.5',
        'hatch.linewidth': 0.1,
        'text.usetex': True
    })

    plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'


def get_color_map():
    color_map = {'green': '#009E60', 'orange': '#C04000',
                 'blue': 'C0', 'black': '#3A3B3C',
                 'purple': '#843B62', 'red': '#C41E3A'}
    return color_map


if __name__ == "__main__":
    theta = torch.tensor([2.0])
    data = gamma_sampler(theta)
    print(data)