from cost_aware_snpe_c import CostAwareSNPE_C
from sbi.inference.snpe.snpe_c import SNPE_C
from sbi.utils.torchutils import *
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

    simulator = hydra.utils.instantiate(cfg.simulator)
    prior = simulator.prior

    N = cfg.N

    theta_npe = prior.sample([N]).reshape(-1, cfg.simulator.theta_dim)
    x_npe = torch.empty([N, cfg.simulator.x_dim])
    for i in range(N):
        x_npe[i, :] = simulator(theta_npe[i])

    obs_x = torch.load("data/" + simulator.name + "_obs_x.pt")
    obs_theta = torch.load("data/" + simulator.name + "_obs_theta.pt")

    inference = SNPE_C(prior=prior)
    density_estimator = inference.append_simulations(theta_npe, x_npe).train()
    posterior = inference.build_posterior(density_estimator)

    posterior_samples = posterior.sample((1000,), x=obs_x)

    if cfg.checkpoint:
        ckpt = {
            "posterior_samples": posterior_samples,
            "true_theta": obs_theta,
        }

        output_directory = get_original_cwd()
        print(f"{output_directory=}")
        checkpoint_path = os.path.join(output_directory, "sims", simulator.name, str(cfg.seed))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = os.path.join(checkpoint_path, "ckpt.tar")
        torch.save(ckpt, checkpoint_path)


if __name__ == "__main__":
    train()
