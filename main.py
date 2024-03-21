
import torch
from sbi import utils as utils

from cost_aware_snpe_c import CostAwareSNPE_C

prior = utils.BoxUniform(low=torch.ones(1), high=1000 * torch.ones(1))

# inference = CostAwareSNPE_C(prior=prior)
inference = CostAwareSNPE_C()

theta = torch.load("data/theta.pt")
x = torch.load("data/sim_data.pt")
obs_data = torch.load("data/obs_data.pt")
weights = torch.load("data/weights.pt")

density_estimator = inference.append_simulations(theta, x).append_weights(weights).train()