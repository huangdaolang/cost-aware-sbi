import time
import torch
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import utils
import os


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def train(cfg):
    if cfg.device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    cost_abc = []
    mmd_abc = []
    cost_caabc = []
    mmd_caabc = []

    N = cfg.N
    m = 500

    dims = [4, 6, 8, 10]

    eps = [4, 10, 13.8, 12.6]

    for i, d in enumerate(dims):
        model = hydra.utils.instantiate(cfg.simulator, dim=d)

        # get reference posterior samples
        theta_ref = np.load(os.path.join(f'data/cmpgm_thetas_ref_dim{d}.npy'))
        x_ref = np.load(os.path.join(f'data/cmpgm_xs_ref_dim{d}.npy'))
        posterior_samples_ref, _ = utils.rejection_ABC(model.x_obs, theta_ref, x_ref, eps[i])

        # ABC
        st_abc = time.time()
        theta_abc = np.random.rand(N, d) * 10
        x_abc = np.vstack([model(theta, num_samples=m) for theta in theta_abc])
        et_abc = time.time()
        cost_abc_ = et_abc - st_abc
        cost_abc.append(cost_abc_)
        print("ABC cost: ", cost_abc_)

        posterior_samples_abc, _ = utils.rejection_ABC(model.x_obs, theta_abc, x_abc, eps[i])

        mmd_abc_ = utils.MMD_unweighted(torch.tensor(posterior_samples_abc), torch.tensor(posterior_samples_ref),
                                        lengthscale=utils.median_heuristic(
                                            torch.tensor(posterior_samples_ref))).numpy()
        mmd_abc.append(mmd_abc_)
        print("MMD ABC: ", mmd_abc_)


        # CA-ABC
        theta_values = np.array([0.01, 10])
        theta_values = np.repeat(theta_values[:, np.newaxis], d, axis=1)

        sample_times = []
        for step in range(2):
            start_time = time.time()
            samples = model(theta_values[step], num_samples=10000)
            elapsed_time = time.time() - start_time
            sample_times.append(elapsed_time)

        # determine the cost function
        alpha = (sample_times[-1] - sample_times[0]) / (np.sum(theta_values[-1]) - np.sum(theta_values[0]))
        beta = -alpha * np.sum(theta_values[0]) + sample_times[0]


        st_caabc = time.time()
        theta_caabc = np.empty([N, d])
        theta_low = np.ones([1, d]) * 1.0
        count = 0
        num_mixtures = len(cfg.simulator.k_mixture)
        num_per_mixture = int(round(N / num_mixtures))

        for ind in range(num_mixtures):
            while count < ((ind + 1) * num_per_mixture):
                theta_ = np.random.rand(1, d) * 10
                # print(calc_acc_prob(cost_linear, alpha, beta, theta_, theta_low, k_mixture[ind]))
                if calc_acc_prob(cost_linear, alpha, beta, theta_, theta_low, cfg.simulator.k_mixture[ind]) > np.random.rand(1):
                    theta_caabc[count] = theta_.reshape(-1)
                    count += 1
        x_caabc = np.vstack([model(theta, num_samples=m) for theta in theta_caabc])
        et_caabc = time.time()
        cost_caabc_ = et_caabc - st_caabc
        cost_caabc.append(cost_caabc_)
        print("CA-ABC cost: ", cost_caabc_)

        posterior_samples_caabc, _ = utils.rejection_ABC(model.x_obs, theta_caabc, x_caabc, eps[i])

        mmd_caabc_ = utils.MMD_unweighted(torch.tensor(posterior_samples_caabc), torch.tensor(posterior_samples_ref),
                                          lengthscale=utils.median_heuristic(
                                              torch.tensor(posterior_samples_ref))).numpy()
        mmd_caabc.append(mmd_caabc_)
        print("MMD CA-ABC: ", mmd_caabc_)

    if cfg.checkpoint:
        ckpt = {
            "cost_abc": cost_abc,
            "mmd_abc": mmd_abc,
            "cost_caabc": cost_caabc,
            "mmd_caabc": mmd_caabc,
        }

        output_directory = get_original_cwd()
        print(f"{output_directory=}")

        checkpoint_path = os.path.join(output_directory, "sims", model.name, "mixture", str(cfg.seed))

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = os.path.join(checkpoint_path, "ckpt.tar")
        torch.save(ckpt, checkpoint_path)


def cost_linear(theta, alpha, beta):
    value = alpha * np.sum(theta, axis=1) + beta
    return value


def calc_acc_prob(cost_linear, alpha, beta, theta, theta_low, k):
    cost = cost_linear(theta, alpha, beta)
    lower_cost = cost_linear(theta_low, alpha, beta)
    return (lower_cost ** k) / (cost ** k)


if __name__ == "__main__":
    train()
