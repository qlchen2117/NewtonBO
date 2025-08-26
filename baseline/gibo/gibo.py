import os

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .src import config
from .src.loop import loop

configs = {
    "method": "gibo",
    "out_dir": './experiments/synthetic_experiments/test_experiment/gibo/',
    # Either choose max_iterations or max_objective_calls not None.
    "max_iterations": None,
    "max_objective_calls": None, #300
    # Manually set hyperparameters.
    "set_hypers": False,
    "only_set_noise_hyper": False,
    "optimizer_config": {
        "max_samples_per_iteration": "dim_search_space",
        "OptimizerTorch": "sgd",
        "optimizer4mll": 'scipy',
        "optimizer_torch_config": {"lr": 0.25},
        "lr_schedular": None,
        "Model": "derivative_gp",
        "model_config": {
            "prior_mean": 0.,
            "ard_num_dims": "dim_search_space",  # If not None, each input dimension gets its own separate lengthscale.
            "N_max": "variable",  # 5*dim_search_space
            "lengthscale_constraint": {"constraint": None, "kwargs": None},
            "lengthscale_hyperprior": {"prior": None,"kwargs": None},
            "outputscale_constraint": {"constraint": None,"kwargs": None},
            "outputscale_hyperprior": {"prior": None, "kwargs": None},
            "noise_constraint": {"constraint": None, "kwargs": None},
            "noise_hyperprior": {"prior": None, "kwargs": None},
        },
        "hyperparameter_config": {
            "optimize_hyperparameters": True,
            "hypers": {
                "covar_module.base_kernel.lengthscale": None,
                "covar_module.outputscale": None,
                "likelihood.noise": None,
            },
            "no_noise_optimization": False,
        },
        "optimize_acqf": "bga",
        "optimize_acqf_config": {"q": 1, "num_restarts": 5, "raw_samples": 64},
        # Either choose bounds or delta unequal None.
        "bounds": {"lower_bound": None, "upper_bound": None},
        "delta": 0.2,
        "epsilon_diff_acq_value": 0.1,
        "generate_initial_data": None,
        "standard_deviation_scaling": False,
        "normalize_gradient": True,
        "verbose": False,
    },
}

def gibo(objective, dim, n_iterations, max_time, n_init, batch_size, optimizer4mll, dtype=torch.double, device=torch.device("cpu")):
    def generate_initial_data(oracle):
        xInit = SobolEngine(dimension=dim, scramble=True).draw(n_init).to(dtype=dtype, device=device)
        yInit = torch.tensor([oracle(xx) for xx in xInit], dtype=dtype, device=device)
        return xInit, yInit
    # Translate config dictionary.
    cfg = config.insert(configs, config.insertion_config)

    cfg_dim = config.evaluate(cfg, dim_search_space=dim, factor_N_max=5,)
    cfg_dim["optimizer_config"]["optimizer4mll"] = optimizer4mll
    cfg_dim["optimizer_config"]['max_samples_per_iteration'] = 1
    cfg_dim["optimizer_config"]["optimize_acqf_config"]['q'] = batch_size
    cfg_dim["optimizer_config"]['generate_initial_data'] = generate_initial_data
    params, f_params, clock = loop(
        params_init=0.5 * torch.ones(dim, dtype=dtype, device=device),
        max_iterations=n_iterations,
        max_objective_calls=None,
        max_time=max_time,
        objective=objective,
        Optimizer=cfg_dim["method"],
        optimizer_config=cfg_dim["optimizer_config"],
        verbose=False,
    )

    # rewards = compute_rewards(params, objective)
    # print(f"Optimizer's max reward: {max(rewards)}")
    return params, f_params, np.array(clock)

if __name__ == '__main__':
    DIM = 36
    N_ITER = 10
    BATCH_SIZE = 5
    N_INIT = 10
    from botorch.test_functions import Ackley
    func = Ackley(dim=DIM)
    func.bounds[0, :].fill_(-5)
    func.bounds[1, :].fill_(10)
    lb, ub = func.bounds
    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return func(lb + (ub - lb) * x) * -1
    X, Y = gibo(objective=eval_objective, dim=DIM, n_iterations=N_ITER, n_init=N_INIT, batch_size=BATCH_SIZE)
    print(X.shape, Y.shape)
    import matplotlib.pyplot as plt
    plt.plot(np.minimum.accumulate(-1*Y))
    plt.show()