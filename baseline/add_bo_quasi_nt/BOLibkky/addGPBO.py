from math import log2
from numpy import ndarray
import numpy as np
import torch
from typing import Callable
import time

from HDBO.add_bo_quasi_nt.addGPLibkky.addGPRegression import add_gp_regression
from HDBO.add_bo_quasi_nt.addGPLibkky.addGPDecompMargLikelihood import add_gp_decomp_marg_likelihood
# from utils.sampleFromMultinomial import sample_from_multinomial
from HDBO.add_bo_quasi_nt.utils.projectToRectangle import project_to_rectangle
from HDBO.add_bo_quasi_nt.my_utils.sampling_torch import draw_sobol_samples
from .getNextQueryPoint import get_next_query_pt


def add_gp_bo(oracle: Callable, bounds: ndarray, num_iterations: int, max_time: int, n_init: int, batch_size: int, params):
    """ Implements Gaussian Process Bandits/ Bayesian Optimization using Additive Gaussian Processes.
    See ICML 2015 parper: "High Dimensional Bayesian Optimization and Bandits via Additive Models". K.Kandasamy,
    J.Schneider, B.Poczos
    Args:
        oracle: A function handle for the function you wish to optimize.
        bounds: A 'numDims x 2' ndarray specifying the lower and upper bound of each dimension.
        num_iterations: The number of GPB/BO iterations.
        params: A sturcture specifying the various hyper parameters for optimization. If you wish
            to use default settings, pass an empty struct. Otherwise, see demoCustomise.py to see how to set each hyper
            parameter. Also see below.
    Returns:
        maxVal: The maximum queried value of the function.
        maxPt: The queried point with the maximum value.
        boQueries: A matrix indicating the points at which the algorithm queried.
        boVals: A vector of the query values.
        history: A vector of the maximum value obtained up until that iteration.

    params should have a field called decompStrategy: It should be one of 'known', 'learn', 'random' and 'partialLearn'.
    'known': The decomposition is known and given in decomp. We will optimize according to this.
    'learn': The decomposition is unknown and should be learned.
    'random': Randomly pick a partition at each iteration.
    'partialLearn': Partially learn the decomposition at each iteration by trying out a few and picking the best.
    The default is partialLearn and is the best option if you don't know much about your function.
    """
    wallclocks = []
    # Prelims
    bounds = torch.from_numpy(bounds)
    ndims = bounds.shape[0]
    max_threshold_exceeds, num_iter_param_relearn = 5, 25
    decomp = params.decomp
    # The Decomposition
    if params.decomp_strategy == 'known':
        decomposition = decomp
        num_groups = len(decomposition)
        # do some diagnostics on the decomposition and print them out
        relevant_coords = torch.hstack(decomposition)
        num_relevant_coords = len(relevant_coords)
        if num_relevant_coords != len(torch.unique(relevant_coords)):
            raise Exception('The same coordinate cannot appear in different groups')
        print('# Groups: %d, %d/%d coordinates are relevant\n' % (num_groups, num_relevant_coords, ndims))
    elif hasattr(decomp, 'm'):
        # Now decomposition should have two fields d and m
        num_groups = decomp.m
    else:  # in this case the decomposition is given.
        num_groups = len(decomp)

    # Initialization points
    startT = time.monotonic()
    num_init_pts = n_init
    print('Obtaining %d Points for Initialization.\n' % num_init_pts)
    init_pts = draw_sobol_samples(bounds=bounds.T, n=num_init_pts, q=1).squeeze()  # shape(num_init, ndims)
    # init_pts = torch.rand(num_init_pts, ndims, dtype=bounds.dtype) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    """init_pts = torch.repeat_interleave(torch.arange(1, num_init_pts+1, dtype=bounds.dtype).reshape(-1, 1) / num_init_pts, ndims, dim=1)
    init_pts = init_pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]"""

    init_vals = torch.tensor([oracle(pt) for pt in init_pts], dtype=init_pts.dtype)
    wallclocks.extend([time.monotonic() - startT] * n_init)
    # use std to change some hyper-parameters.
    params.common_noise = params.common_noise * torch.std(init_vals, dim=0)
    params.sigma_pr_range = params.sigma_pr_range.to(init_vals) * torch.std(init_vals, dim=0)
    params.noises = params.noises.to(init_vals)

    # The Bandwidth
    # This BO algorithm will set the bandwidth via its own procedure
    al_bw_lb = params.al_bw_lb
    al_bw_ub = params.al_bw_ub
    # Set an initial bandwidth. This will change as the algorithm progresses
    al_curr_bw = al_bw_ub

    # Define the following before proceeding
    bo_queries = init_pts
    bo_vals = init_vals

    thresh_exceeded_counter = 0
    # print('Performing BO (dim = %d)\n' % ndims)
    bo_iter = 0
    while bo_iter < num_iterations and time.monotonic() - startT < max_time:
        # if (bo_iter+1) % num_iter_param_relearn == 0:
        #     print('Additive GP BO iter %d/%d. MaxVal: %0.4f CumReward: %0.4f\n'
        #           % (bo_iter, num_iterations, best_y, torch.sum(bo_vals)/(bo_iter + num_init_pts)))

        # Prelims
        # num_x = num_init_pts + bo_iter * batch_size
        """standardize to mean 0 and var 1"""
        train_y = (bo_vals - bo_vals.mean()) / bo_vals.std()  # shape(num)

        train_x = bo_queries

        # First redefine ranges for the GP bandwidth if needed
        if (not params.use_fixed_bandwidth) and \
                (bo_iter * batch_size % num_iter_param_relearn == 0 or 
                 thresh_exceeded_counter >= max_threshold_exceeds):
            if thresh_exceeded_counter >= max_threshold_exceeds:
                al_bw_ub = max(al_bw_lb, 0.9 * al_curr_bw)
                thresh_exceeded_counter = 0
                print('Threshold Exceeded %d times - Reducing BW\n' % max_threshold_exceeds)
            else:
                pass
            # Define the BW range for addGPMargLikelihood
            if al_bw_ub == al_bw_lb:
                params.fix_sm = True
                params.sigma_pr_ranges = al_bw_lb * torch.ones(num_groups)
            else:
                params.fix_sm = False
                # Use same bandwidth for now.
                params.use_same_sm = True
                params.sigma_sm_range = torch.tensor([al_bw_lb, al_bw_ub]).to(bo_queries)

            # Obtain the optimal GP parameters
            if params.decomp_strategy != 'stoch1':
                al_curr_bws, al_curr_scales, _, learned_decomp, marg_like_val \
                    = add_gp_decomp_marg_likelihood(train_x, train_y, decomp, params)
                al_curr_bw = al_curr_bws[0]  # modify to allow different bandwidths
            else:
                raise NotImplementedError

        # If stochastic pick a current GP
        if params.decomp_strategy != 'stoch1':
            params.sigma_sms = al_curr_bws
            params.sigma_prs = al_curr_scales
            curr_iter_decomp = learned_decomp
        else:
            raise NotImplementedError

        # Now build the GP
        add_gp_regression(train_x, train_y, curr_iter_decomp, params)
        # Now obtain the next point
        candidates, _, candidates_std = get_next_query_pt(train_x, train_y, params, curr_iter_decomp, bounds, batch_size)
        # If it is too close, perturb it a bit
        for candidate, cand_std in zip(candidates, candidates_std):
            if ((train_x-candidate)**2).sum(dim=-1).sqrt().min() / al_curr_bw < 1e-10:
                print("The candidate is too close to training set, perturb it a bit")
                while ((train_x-candidate)**2).sum(dim=-1).sqrt().min() / al_curr_bw < 1e-10:
                    candidate = project_to_rectangle(candidate + 0.1 * al_curr_bw * torch.randn(ndims).to(candidate), bounds)
            # Determine the current best point
            value = torch.tensor(oracle(candidate)).to(bo_vals)
            bo_queries = torch.cat((bo_queries, candidate.view(1, -1)))
            bo_vals = torch.cat((bo_vals, value.view(1)))
            wallclocks.append(time.monotonic() - startT)
            # Check if next_point_std is too small
            if cand_std < params.opt_pt_std_threshold:
                print("next_point_std is too small")
                thresh_exceeded_counter += 1
            else:
                thresh_exceeded_counter = 0
        bo_iter += 1
    return bo_queries.numpy(), bo_vals.numpy(), np.array(wallclocks)
