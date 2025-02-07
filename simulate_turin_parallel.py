import torch
import numpy as np
from sbi import utils as utils


import utils as u

import warnings
warnings.filterwarnings("ignore")

import time


device = torch.device("cpu")


def TurinModel(theta, B=4e9, Ns=801, N=50, tau0=0, output="moments"):
    G0 = theta[0].to(device)
    T = theta[1].to(device)
    lambda_0 = theta[2].to(device)
    sigma2_N = theta[3].to(device)

    nRx = N

    delta_f = B / (Ns - 1)  # Frequency step size
    t_max = 1 / delta_f

    tau = torch.linspace(0, t_max, Ns)

    H = torch.zeros((nRx, Ns), dtype=torch.cfloat)
    mu_poisson = lambda_0 * t_max  # Mean of Poisson process

    for jR in range(nRx):

        n_points = int(torch.poisson(mu_poisson))  # Number of delay points sampled from Poisson process

        delays = (torch.rand(n_points) * t_max).to(device)  # Delays sampled from a 1-dimensional Poisson point process

        delays = torch.sort(delays)[0]

        alpha = torch.zeros(n_points,
                            dtype=torch.cfloat).to(
            device)  # Initialising vector of gains of length equal to the number of delay points

        sigma2 = G0 * torch.exp(-delays / T) / lambda_0 * B

        for l in range(n_points):
            if delays[l] < tau0:
                alpha[l] = 0
            else:
                std = torch.sqrt(sigma2[l] / 2)
                alpha[l] = torch.normal(0, std) + torch.normal(0, std) * 1j

        H[jR, :] = torch.matmul(torch.exp(-1j * 2 * torch.pi * delta_f * (torch.ger(torch.arange(Ns), delays))), alpha)

    # Noise power by setting SNR
    Noise = torch.zeros((nRx, Ns), dtype=torch.cfloat).to(device)

    for j in range(nRx):
        normal = torch.distributions.normal.Normal(0, torch.sqrt(sigma2_N / 2))
        Noise[j, :] = normal.sample([Ns]) + normal.sample([Ns]) * 1j

    # Received signal in frequency domain

    Y = H + Noise

    if output == "moments":
        temporal_moments = temporalMomentsGeneral(Y)
        return temporal_moments
    elif output == "data":
        y = torch.zeros(Y.shape, dtype=torch.cfloat).to(device)
        p = torch.zeros(Y.shape).to(device).to(device)
        lens = len(Y[:, 0])

        for i in range(lens):
            y[i, :] = torch.fft.ifft(Y[i, :])

            p[i, :] = torch.abs(y[i, :]) ** 2

        return 10 * torch.log10(p)


def temporalMomentsGeneral(Y, K=3, B=4e9):
    N, Ns = Y.shape

    delta_f = B / (Ns - 1)
    t_max = 1 / delta_f

    tau = torch.linspace(0, t_max, Ns)
    out = torch.zeros((N, K), dtype=torch.float64)

    for k in range(K):
        for i in range(N):
            y = torch.fft.ifft(Y[i, :])
            out[i, k] = torch.trapz(tau ** (k) * (torch.abs(y) ** 2), tau)

    x_stat = torch.zeros((6,))
    x_stat[0:3] = torch.mean(torch.log(out), axis=0)
    x_stat[3:6] = torch.std(torch.log(out), axis=0)

    return x_stat


def main():
    num_dim = 4
    prior_start = torch.tensor([1e-9, 1e-9, 1e9, 1e-10])
    prior_end = torch.tensor([1e-8, 1e-8, 1e10, 1e-9])

    prior = utils.BoxUniform(low=prior_start * torch.ones(num_dim), high=prior_end * torch.ones(num_dim))

    num_sim = 200

    x_turin_npe = torch.zeros(size=(num_sim, 6))

    st = time.time()

    prior_samples = prior.sample((num_sim,))

    for i in range(num_sim):
        x_turin_npe[i, :] = TurinModel(prior_samples[i, :])
    et = time.time()

    time_npe = et - st

    np.save("time_npe.npy", time_npe)
    print("NPE Time taken: ", time_npe)


    k = 2
    alpha = torch.tensor(np.load("data/turin_cost_alpha.npy"))
    beta = torch.tensor(np.load("data/turin_cost_beta.npy"))

    x_turin_ca = torch.zeros(size=(num_sim, 6))
    theta_tilde = torch.zeros(num_sim, num_dim)

    count = 0

    st = time.time()

    while count < num_sim:
        param_value = prior.sample()
        if u.A_theta_Gamma(param_value[2], k, alpha=alpha, beta=beta, prior_start=prior_start[2]) > torch.rand(1):
            theta_tilde[count, :] = param_value
            count += 1

    for i in range(num_sim):
        x_turin_ca[i, :] = TurinModel(theta_tilde[i, :])
    et = time.time()
    time_canpe = et - st
    np.save("time_canpe.npy", time_canpe)
    print("CA-NPE k=2 Time taken: ", time_canpe)

    # Mixture
    k = np.array([0, 1, 2, 3])  # Exponent of the penaly function g(z) = z^k

    num_mixtures = k.size
    num_per_mixture = int(round(num_sim / num_mixtures))

    x_turin_ca = torch.zeros(size=(num_sim, 6))
    theta_tilde = torch.zeros(num_sim, num_dim)
    w_u = torch.zeros(num_sim, )

    count = 0

    st = time.time()

    for ind in range(k.size):

        while count < ((ind + 1) * num_per_mixture):
            param_value = prior.sample()
            if u.A_theta_Gamma(param_value[2], k[ind], alpha=alpha, beta=beta, prior_start=prior_start[2]) > torch.rand(
                    1):
                theta_tilde[count, :] = param_value
                w_u[count] = u.cost_linear(theta_tilde[count, 2], alpha, beta) ** k[ind]  # self-normalised importance weights
                count += 1

    for i in range(num_sim):
        x_turin_ca[i, :] = TurinModel(theta_tilde[i, :])

    et = time.time()
    time_canpe_mixture = et - st
    np.save("time_canpe_mixture.npy", time_canpe_mixture)
    print("CA-NPE mixture Time taken: ", time_canpe_mixture)


if __name__ == "__main__":
    main()