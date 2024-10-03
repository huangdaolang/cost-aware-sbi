import torch
from cost_aware_snle_a import CostAwareSNLE_A
from cost_aware_snpe_c import CostAwareSNPE_C
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.inference.snle.snle_a import SNLE_A
from sbi.utils.torchutils import *
from sbi.utils import process_prior
import gpytorch

import time
import numpy as np
import hydra
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

    obs_x = simulator.x_obs
    obs_theta = simulator.theta_true

    x_npe, theta_npe = simulator.load_npe_data()

    if cfg.simulator.inference == "npe":
        inference = SNPE_C(prior=prior)
    elif cfg.simulator.inference == "nle":
        inference = SNLE_A(prior=prior)
    else:
        raise ValueError("Invalid inference method")

    density_estimator = inference.append_simulations(theta_npe, x_npe).train()
    posterior = inference.build_posterior(density_estimator)

    posterior_samples = posterior.sample((1000,), x=obs_x)

    # CA-NPE
    x_canpe, theta_canpe, w_canpe = simulator.load_ca_npe_data(cfg.mixture, cfg.k)
    if cfg.mixture:
        k_indicator_canpe = torch.cat([torch.full((2500,), i) for i in range(4)])
    else:
        k_indicator_canpe = torch.full((10000,), 0)

    if cfg.simulator.inference == "npe":
        inference_canpe = CostAwareSNPE_C(prior=prior)
    elif cfg.simulator.inference == "nle":
        inference_canpe = CostAwareSNLE_A(prior=prior)
    else:
        raise ValueError("Invalid inference method")

    density_estimator_canpe = inference_canpe.append_simulations(theta_canpe, x_canpe).append_weights(w_canpe, k_indicator_canpe).train()
    posterior_canpe = inference_canpe.build_posterior(density_estimator_canpe)

    posterior_samples_canpe = posterior_canpe.sample((1000,), x=obs_x)

    if cfg.checkpoint:
        ckpt = {
            "posterior_npe": posterior,
            "posterior_samples_npe": posterior_samples,
            "posterior_canpe": posterior_canpe,
            "posterior_samples_canpe": posterior_samples_canpe,
            "true_theta": obs_theta,
        }

        output_directory = get_original_cwd()
        print(f"{output_directory=}")
        if cfg.mixture is True:
            checkpoint_path = os.path.join(output_directory, "sims", simulator.name, "mixture", cfg.simulator.inference, str(cfg.seed))
        else:
            checkpoint_path = os.path.join(output_directory, "sims", simulator.name, str(cfg.k), cfg.simulator.inference, str(cfg.seed))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = os.path.join(checkpoint_path, "ckpt.tar")
        torch.save(ckpt, checkpoint_path)


if __name__ == "__main__":
    train()
