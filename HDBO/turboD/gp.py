###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

from math import sqrt, log
from functools import partial
import numpy as np
import torch
from torch.optim import Adam, SGD
from gpytorch.constraints.constraints import Interval, GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import LogNormalPrior
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_torch, fit_gpytorch_mll_scipy
from torch.utils.data import Dataset, DataLoader

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)
MIN_INFERRED_NOISE_LEVEL = 1e-4

optimAdam = partial(Adam, lr=0.05)


class Data(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.len = train_x.shape[0]
    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]
    def __len__(self):
        return self.len


# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        # base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        base_kernel = RBFKernel(ard_num_dims=ard_dims, lengthscale_constraint=lengthscale_constraint)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# Vanilla GP Model in [Hvarfner2024vanilla]
class VanillaGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, covar_module):
        super(VanillaGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



def train_gp(train_x, train_y, use_ard, num_steps, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


def train_gp_minibatch(train_x, train_y, use_ard, num_epochs, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = SGD([{"params": model.parameters()}], lr=0.025)
    dataset = Data(train_x, train_y)
    trainloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    for _ in range(num_epochs):
        for xx, yy in trainloader:
            optimizer.zero_grad()
            model.set_train_data(xx, yy, False)
            output = model(xx)
            loss = -mll(output, yy)
            loss.backward()
            optimizer.step()

    model.set_train_data(train_x, train_y, False)
    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


def train_vanillaGP(train_x, train_y, use_ard, num_steps, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    ard_dims = train_x.shape[-1] if use_ard else None

    # >>> kernel >>>
    lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_dims) * 0.5, scale=SQRT3)
    covar_module = RBFKernel(
        ard_num_dims=ard_dims,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=lengthscale_prior.mode)
    )
    # <<< kernel <<<

    # >>> Likelihood >>>
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, initial_value=noise_prior.mode)
    )
    # <<< Likelihood <<<

    model = SingleTaskGP(train_X=train_x, train_Y=train_y.view(-1, 1), likelihood=likelihood, covar_module=covar_module, outcome_transform=None, input_transform=None)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)

    # >>> Fit model >>>
    fit_gpytorch_mll_scipy(mll)
    # fit_gpytorch_mll_torch(mll, step_limit=num_steps, optimizer=optimAdam)
    # <<< Fit model <<<

    # Switch to eval mode
    model.eval()
    model.likelihood.eval()

    return model


def fine_tune(model, train_x, train_y, num_steps):

    model.set_train_data(train_x, train_y, strict=False)
    # Find optimal model hyperparameters
    model.train()
    model.likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    model.likelihood.eval()


def plot_gp(model, train_x, train_y, mu, sigma, func):
    from matplotlib import pyplot as plt
    from torch.quasirandom import SobolEngine

    dim = train_x.shape[-1]
    sobol = SobolEngine(dimension=dim, scramble=True)
    test_x = sobol.draw(n=100).to(train_x)
    test_y = (np.array([func(xx) for xx in test_x]) - mu.numpy()) / sigma.numpy()
    with torch.no_grad():
        posterior = model.likelihood(model(test_x))
    f_mean = posterior.mean
    lowB, upB = posterior.confidence_region()
    fig = plt.figure()
    # nrows = math.ceil(math.sqrt(dim))
    # >>> Plot the posterior
    for ii in range(1):
        ax = fig.add_subplot(1, 1, ii+1)
        xx = test_x[:, ii]
        indices = torch.argsort(xx)
        ax.plot(xx[indices], f_mean[indices], color = "blue")
        ax.fill_between(xx[indices], lowB[indices], upB[indices], color = "blue", alpha = 0.3,)

        # xx2 = train_x[:, ii]
        # indices = torch.argsort(xx2)
        ax.scatter(xx[indices], test_y[indices], color = "black", marker = "*", alpha = 0.5)
        ax.legend()
        ax.set_ylim(-1, 1)
        # ax.xlabel("x")
        # ax.ylabel("y")
    # <<< Plot the posterior
    plt.savefig("GP.png", bbox_inches='tight')


