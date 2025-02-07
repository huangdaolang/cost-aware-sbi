from cost_aware_snpe_c import CostAwareSNPE_C
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.torchutils import *
from sbi.utils import process_prior
import gpytorch

import time
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import utils
import os


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def train(cfg):
    if cfg.device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    simulator = hydra.utils.instantiate(cfg.simulator)
    prior = simulator.prior

    prior, *_ = process_prior(prior)

    N = cfg.N

    st_npe = time.time()
    theta_npe = simulator.sample_theta([N]).reshape(-1, cfg.simulator.theta_dim)
    x_npe = torch.empty([N, cfg.simulator.x_dim])

    for i in range(N):
        x_npe[i, :] = simulator(theta_npe[i])
    et_npe = time.time()
    cost_npe = et_npe - st_npe
    print("NPE cost: ", cost_npe)

    obs_x = torch.load("data/" + simulator.name + "_obs_x.pt")
    obs_theta = torch.load("data/" + simulator.name + "_obs_theta.pt")

    inference = SNPE_C(prior=prior)
    density_estimator = inference.append_simulations(theta_npe, x_npe).train()
    posterior = inference.build_posterior(density_estimator)

    posterior_samples = posterior.sample((1000,), x=obs_x)

    # Load linear cost function for temporal sir
    # a_1 = torch.tensor(np.load("data/temp_sir_linear_a1.npy"))
    # a_2 = torch.tensor(np.load("data/temp_sir_linear_a2.npy"))
    # b = torch.tensor(np.load("data/temp_sir_linear_b.npy"))
    a_2 = torch.tensor([-0.066272676])
    a_1 = torch.tensor([0.04224225])
    b = torch.tensor([0.07])

    # params of lowest cost
    prior_start = torch.tensor(cfg.simulator.prior_start).reshape(-1, cfg.simulator.theta_dim)

    theta_canpe = torch.zeros([N, cfg.simulator.theta_dim])
    x_canpe = torch.empty([N, cfg.simulator.x_dim])
    w_canpe = torch.zeros([N])
    k_indicator_canpe = torch.zeros([N], dtype=torch.int)
    count = 0

    st_sim = time.time()

    if cfg.mixture is True:
        num_mixtures = len(cfg.simulator.k_mixture)
        num_per_mixture = int(round(N / num_mixtures))

        for ind in range(num_mixtures):
            while count < ((ind + 1) * num_per_mixture):
                theta_ = simulator.sample_theta([1]).reshape(-1, cfg.simulator.theta_dim)
                if utils.calc_acc_prob_linear(a_1, a_2, b, theta_, prior_start, cfg.simulator.k_mixture[ind]) > torch.rand(1):
                    theta_canpe[count] = theta_.reshape(-1)
                    w_canpe[count] = (a_1*theta_canpe[count][0] + a_2*theta_canpe[count][1] + b).detach() ** cfg.simulator.k_mixture[ind]
                    k_indicator_canpe[count] = ind
                    count += 1
    else:
        while count < N:
            theta_ = simulator.sample_theta([1]).reshape(-1, cfg.simulator.theta_dim)
            if utils.calc_acc_prob_linear(a_1, a_2, b, theta_, prior_start, cfg.k) > torch.rand(1):
                theta_canpe[count] = theta_.reshape(-1)
                count += 1
        w_canpe = (a_1*theta_canpe[:, 0] + a_2*theta_canpe[:, 1] + b).detach() ** cfg.k
        k_indicator_canpe = torch.full((N,), 0)

    et_sim = time.time()
    print("Sampling time: ", et_sim - st_sim)



    for i in range(N):
        x_canpe[i, :] = simulator(theta_canpe[i])
    et_canpe = time.time()

    total_cost_canpe = et_canpe - st_sim
    print("CA-NPE cost", total_cost_canpe)

    inference_canpe = CostAwareSNPE_C(prior=prior)
    density_estimator_canpe = inference_canpe.append_simulations(theta_canpe, x_canpe).append_weights(w_canpe, k_indicator_canpe).train()
    posterior_canpe = inference_canpe.build_posterior(density_estimator_canpe)

    posterior_samples_canpe = posterior_canpe.sample((1000,), x=obs_x)

    if cfg.checkpoint:
        ckpt = {
            "posterior_npe": posterior,
            "posterior_samples_npe": posterior_samples,
            "cost_npe": cost_npe,
            "posterior_canpe": posterior_canpe,
            "posterior_samples_canpe": posterior_samples_canpe,
            "cost_canpe": total_cost_canpe,
            "true_theta": obs_theta,
        }

        output_directory = get_original_cwd()
        print(f"{output_directory=}")
        if cfg.mixture is True:
            checkpoint_path = os.path.join(output_directory, "sims", simulator.name, "linear_step", "multiple", str(cfg.seed))
        else:
            checkpoint_path = os.path.join(output_directory, "sims", simulator.name, "linear_step", str(cfg.k), str(cfg.seed))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = os.path.join(checkpoint_path, "ckpt.tar")
        torch.save(ckpt, checkpoint_path)


if __name__ == "__main__":
    train()
