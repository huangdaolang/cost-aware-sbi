# Cost-aware Simulation-based Inference
This repository contains the code for the paper "[Cost-aware Simulation-based Inference](https://arxiv.org/abs/2410.07930)". Our work is built on top of [sbi](https://github.com/sbi-dev/sbi) package.

## Installation
```
git clone https://github.com/huangdaolang/cost-aware-sbi.git

cd cost-aware-sbi

conda create --name cost-aware-sbi python=3.9

conda activate cost-aware-sbi

pip install -r requirements.txt
```

## Reproducing the experiments
### Running the experiments

We use [Hydra](https://hydra.cc/) to manage the configurations. See `configs` for all configurations and defaults.

An example case to run Homogeneous SIR model is as follows:
```
python run_sir.py simulator=homogeneous_sir N=10000 seed=2024 k=1 mixture=False
```
It will run the Homogeneous SIR model with 10000 samples, seed 2024, k=1, and without training with multiple importance sampling.

In each notebook, we show how to create the cost function, and also the plots and statistics for the experiments.

## Citation

If you use this code in your research, please cite our paper:

```
@misc{bharti2024costawaresimulationbasedinference,
      title={Cost-aware Simulation-based Inference}, 
      author={Ayush Bharti and Daolang Huang and Samuel Kaski and François-Xavier Briol},
      year={2024},
      eprint={2410.07930},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.07930}, 
}
```