import torch
import numpy as np


def MMD_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    m = x.shape[0]
    n = y.shape[0]

    z = torch.cat((x, y), dim=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)


def median_heuristic(y):
    a = torch.cdist(y, y)**2
    return torch.sqrt(torch.median(a / 2))


def kernel_matrix(x, y, l):
    d = torch.cdist(x, y)**2

    kernel = torch.exp(-(1 / (2 * l ** 2)) * d)

    return kernel

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


if __name__ == "__main__":
    theta = torch.tensor([2.0])
    data = gamma_sampler(theta)
    print(data)