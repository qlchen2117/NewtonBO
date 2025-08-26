import torch
from torch import Tensor
from typing import List
from HDBO.add_bo_quasi_nt.addGPLibkky.computeIndGPOutput import compute_ind_gp_output
from HDBO.add_bo_quasi_nt.addGPLibkky.computeCombGPOutput import compute_comb_gp_output
# from HDBO.add_bo_quasi_nt.my_optim.optimize import optimize_acqf
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from HDBO.add_bo_quasi_nt.utils.plot3D import plot_acq
from .getAcqUtility import get_ucb_utility, get_ei_utility


class Utility:
    def __init__(self, model) -> None:
        self.model = model
    def forward(self, x: Tensor):
        return self.model(x)

def get_next_query_pt(train_x: Tensor, train_y: Tensor, params, decomposition: List[Tensor], bounds: Tensor, batch_size:int):
    """Obtain the next query point.
    Args:
        train_x: A 'num x ndims'-dim ndarray, training set.
        train_y: A 'num'-dim ndarray, testing set.
        params: Hyper parameters
        decomposition:
        bounds: A 'ndims x 2' tensor. bounds[:, 0] gives the lower bounds for x,
            and bounds[:, 1] gives the upper bounds for x.
    Returns:
        next_point, next_point_mean, next_point_std:
    """
    # So here's the plane here. We will define a utility function over each GP and
    # pick the point that maximizes this utility function.
    num_groups = len(decomposition)
    ndims = bounds.shape[0]
    sigma_sms, sigma_prs, mean_funcs, common_mean_func = params.sigma_sms, params.sigma_prs, [params.mean_funcs] * num_groups, params.common_mean_func
    ell, alpha = params.ell, params.alpha
    next_points = torch.zeros(batch_size, ndims).to(train_x)

    for k in range(num_groups):
        coord = decomposition[k]
        curr_bounds = bounds[coord]
        bw, scale, curr_mean_func = sigma_sms[k], sigma_prs[k], mean_funcs[k]

        def ucb(t):
            return get_ucb_utility(
                t, lambda x_test: compute_ind_gp_output(x_test, train_x, coord, ell, alpha, bw, scale, curr_mean_func),
                train_y.shape[0]
            )

        def ei(t):
            return get_ei_utility(
                t, lambda x_test: compute_ind_gp_output(x_test, train_x, coord, ell, alpha, bw, scale, curr_mean_func),
                torch.max(train_y)
            )
        if params.utility_func == 'UCB':
            utility = ucb
        elif params.utility_func == 'EI':
            # Only applies to non-additive models. (i.e. d = D, M = 1)
            utility = ei
        else:
            raise NotImplementedError

        # plot_acq(utility, bounds)
        # To maximize the acquisition function
        candidates, acq_value = optimize_acqf(utility, curr_bounds.T, q=batch_size, num_restarts=5, raw_samples=64)
        # Store next_point in relevant coordinates
        next_points[:, coord] = candidates
    # Finally return the mean and the standard deviation at next_point
    # Now the combined GP output
    next_point_mean, next_point_std, _ = compute_comb_gp_output(next_points, train_x, train_y, decomposition, ell, alpha, sigma_sms,
                                                                sigma_prs, mean_funcs, common_mean_func)
    return next_points, next_point_mean, next_point_std
