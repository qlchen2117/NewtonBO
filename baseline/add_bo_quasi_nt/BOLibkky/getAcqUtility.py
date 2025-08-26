import torch
from torch import Tensor
from typing import Callable
from scipy.stats import norm
from torch.distributions.multivariate_normal import MultivariateNormal


def get_ucb_utility(x: Tensor, model: Callable, num_evals: int):
    """
    Args:
        X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
            design points each.
    Returns:
        A `(b)`-dim Tensor of acquisition function values at the given
        design points `X`.
    """
    # Prelims
    ndims = x.shape[-1]  # Expecting each x to be a row vector here.
    # Set beta_t. Using recommendation from Section 6 in Srinivas et al., ICML 2010
    t = torch.tensor(num_evals + 1)
    # Linear in dims, log in t
    beta_t = ndims * torch.log(2*t) / 5
    # Obtain mean and standard deviation
    mu, covar = model(x)  # shape(batch, 1)
    mvn = MultivariateNormal(mu, covar + 1e-6 * torch.eye(covar.shape[-1]).to(covar))
    samples = mvn.rsample([512])
    mean = samples.mean(dim=0)
    ucb_samples = mean + beta_t * (samples - mean).abs()
    return ucb_samples.max(dim=-1)[0].mean(dim=0)  # shape(batch)


def truncate_gaussian_mean(mu, sigma, trunc):
    """
    Computes the value E[max(0,x)] where x~N(mu, sigma**2)
    :param mu:
    :param sigma:
    :param trunc:
    :return trunc_mean:
    """
    y = mu - trunc
    var_zero_idxs = (sigma == 0)
    return var_zero_idxs * max(y, 0) + (~var_zero_idxs) * (y * norm.cdf(y/sigma) + sigma * norm.pdf(y/sigma))


def get_ei_utility(x, gp_func_h, trunc):
    """
    Expected Improvement Utility. Applies only to non-additive functions.
    :param x:
    :param gp_func_h:
    :param trunc:
    :return:
    """
    mu, sigma = gp_func_h(x)
    return truncate_gaussian_mean(mu, sigma, trunc)
