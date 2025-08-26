import cma
import torch
from torch.nn.init import trunc_normal_
import math
import time
from torch.distributions.normal import Normal
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.predictive_entropy_search  import qPredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize, Bounds
from .gp import train_vanillaGP
# from num_opt.grad_projection import gradientprojection


normaler = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

# def ei(x, y_target, model, xi=0.0):
#     # Expected Improvement (EI)
#     with torch.no_grad():
#         posterior = model.likelihood(model(x))
#     mu, stddev = posterior.mean, posterior.stddev
#     std = torch.maximum(stddev, torch.tensor(1e-6))
#     improve = y_target - xi - mu
#     scaled = improve / std
#     cdf, pdf = normaler.cdf(scaled), torch.exp(normaler.log_prob(scaled))
#     exploit = improve * cdf
#     explore = std * pdf
#     values = exploit + explore
#     return values


class TurboState:
    """Maintain the TuRBO state
    TuRBO needs to maintain a state, which includes the length of the trust region,
    success and failure counters, success and failure tolerance, etc. 

    In this tutorial we store the state in a dataclass and update the
    state of TuRBO after each batch evaluation. 

    **Note**: These settings assume that the domain has been scaled to $[0, 1]^d$ 
    and that the same batch size is used for each iteration.
    """
    def __init__(self, dim, batch_size, x_center, y_center) -> None:
        self.dim: int = dim
        self.batch_size: int = batch_size
        self.length: float = 0.4
        self.length_min: float = 0.1
        self.length_max: float = 0.8
        self.failure_counter: int = 0
        self.failure_tolerance: int = float("nan")  # Note: Post-initialized
        self.success_counter: int = 0
        self.success_tolerance: int = 3  # Note: The original paper uses 3
        self.best_value: float = -float("inf")
        self.restart_triggered: bool = False

        # self.failure_tolerance = math.ceil(
        #     max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        # )
        # self.failure_tolerance = max(5, self.dim)
        self.failure_tolerance = 3

        self.x_center = x_center
        self.best_value = y_center


def standardize(Y_turbo):
    mu, sigma = Y_turbo.mean(), Y_turbo.std()
    sigma = 1.0 if sigma < 1e-6 else sigma
    return (Y_turbo - mu) / sigma, mu, sigma

def update_state(state, X_next, Y_next):
    if Y_next.min() < state.best_value - 1e-3 * math.fabs(state.best_value):
        print("Success.")
        state.success_counter += 1
        state.failure_counter = 0
    else:
        print("Fail.")
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    idx = Y_next.argmin()
    if state.best_value > Y_next[idx].item():
        state.best_value = Y_next[idx].item()
        state.x_center = X_next[idx]

    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def solveSubProblem(fun, lb, ub):
    xRand = qmc.Sobol(d=len(lb), scramble=True).random_base2(m=10)
    xRand = xRand * (ub - lb) + lb  # scale
    yRand = np.array([fun(xx) for xx in xRand])
    indices = yRand.argsort()
    x0s, y0s = xRand[indices[:10]], yRand[indices[:10]]  # select k-smallest x
    sols, vals = np.zeros_like(x0s), np.zeros(x0s.shape[0])
    for ii, x0 in enumerate(x0s):
        # step = gradientprojection(x0, grad_np, hessian_np, lb, ub)
        res = minimize(fun, x0=x0, method='L-BFGS-B', bounds=Bounds(lb, ub))
        step = res.x
        sols[ii], vals[ii] = step, fun(step)
    return sols, vals, x0s, y0s


# from botorch.models import SingleTaskGP
# from gpytorch.constraints import Interval
# from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
# from gpytorch.mlls import ExactMarginalLogLikelihood

# def get_model(xTrain, train_Y, dim):
#     # indices = torch.LongTensor(random.sample(range(dim), low_dim))
#     lengthscale_constraint = Interval(0.005, 2.0)
#     covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim, lengthscale_constraint=lengthscale_constraint))
#     model = SingleTaskGP(
#         xTrain, train_Y,
#         covar_module=covar_module,
#         # likelihood=likelihood
#     )
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)
#     return model, mll


def turboD(eval_func, dim, n_init, total_trials, max_time, batch_size,
    restart_strategy='PES', num_steps_gp=50, tol=1e-7, dtype=torch.double, device=torch.device("cpu"), 
):
    startT = time.monotonic()
    X_turbo = torch.from_numpy(latin_hypercube(n_init, dim)).to(dtype=dtype, device=device)  # [n_init, dim]
    Y_turbo = torch.tensor(
        [eval_func(x) for x in X_turbo], dtype=dtype, device=device
    ).unsqueeze(-1)
    wallclocks = [time.monotonic() - startT] * n_init
    # >>> Initialize model >>>
    train_Y, _, _ = standardize(Y_turbo)
    model = train_vanillaGP(X_turbo, train_Y.view(-1), True, num_steps_gp)
    # <<< Initialize model <<<

    # from .gp import plot_gp
    # plot_gp(model, X_turbo, train_Y.view(-1), mu, sigma, eval_func)
    # exit()

    if restart_strategy == 'random':
        # >>> Greedily select initial points
        indices = torch.topk(Y_turbo.view(-1), batch_size, largest=False).indices
        x_centers = X_turbo[indices]
        y_centers = Y_turbo[indices]
        # <<< Greedily select initial points
    elif restart_strategy == 'PES':
        # # >>> Employ AF to select initial points >>>
        # X_rand = torch.from_numpy(latin_hypercube(10000, dim)).to(dtype=dtype, device=device)
        optimal_inputs, _ = get_optimal_samples(
            model, bounds=torch.stack((torch.zeros(dim), torch.ones(dim))).to(dtype=dtype, device=device),
            num_optima=12, maximize=False
        )
        # # qJES = qJointEntropySearch(model=model, optimal_inputs=optimal_inputs, optimal_outputs=optimal_outputs, estimation_type='LB', maximize=False)
        # # qMES = qMaxValueEntropy(model, candidate_set=torch.rand(1000, dim, dtype=dtype, device=device), maximize=False)
        AF = qPredictiveEntropySearch(model, optimal_inputs=optimal_inputs, maximize=False)
        # with torch.no_grad():
        #     af_rand = AF(X_rand.unsqueeze(-2))
        # # logEI = LogExpectedImprovement(model, train_Y.min().item(), maximize=False)
        # # ei_rand = logEI(X_rand.unsqueeze(-2))
        # top_inds = torch.topk(af_rand, batch_size).indices
        # x_centers = X_rand[top_inds, :]
        # y_centers =  torch.tensor(
        #     [eval_func(x) for x in x_centers], dtype=dtype, device=device
        # ).unsqueeze(-1)
        # X_turbo = torch.cat((X_turbo, x_centers), dim=0) # Append data
        # Y_turbo = torch.cat((Y_turbo, y_centers), dim=0)
        # # <<< Employ AF to select initial points

        ## >>> CMA-ES to select initial points >>>
        es = cma.CMAEvolutionStrategy(  # create the CMA-ES optimizer
            x0=np.ones(batch_size * dim) * .5,
            sigma0=0.2,
            inopts={"bounds": [0, 1], "popsize": 50, 'maxiter': 200},
        )
        with torch.no_grad():
            while not es.stop():
                xs = np.vstack(es.ask())
                xs_th = torch.from_numpy(xs).to(dtype=dtype, device=device).view(50*batch_size, dim)  # shape(popsize x batch_size, dim)
                ys = -AF(xs_th.unsqueeze(-2)).view(50, batch_size).cpu().numpy()  # shape(popsize, batch_size)
                es.tell(xs, ys.sum(1))  # return the result to the optimize
                # es.disp()
        x_centers = torch.from_numpy(es.best.x).to(dtype=dtype, device=device).view(batch_size, dim)  # shape(batch_size, dim)
        y_centers =  torch.tensor(
            [eval_func(x) for x in x_centers], dtype=dtype, device=device
        ).unsqueeze(-1)
        ## <<< CMA-ES to select initial points <<<
    else:
        raise NotImplementedError

    X_turbo = torch.cat((X_turbo, x_centers), dim=0) # Append data
    Y_turbo = torch.cat((Y_turbo, y_centers), dim=0)
    wallclocks.extend([time.monotonic() - startT]*batch_size)
    train_Y, _, _ = standardize(Y_turbo)
    model.set_train_data(X_turbo, train_Y.view(-1), strict=False)

    states = [TurboState(dim, batch_size, x_centers[i], y_centers[i].item()) for i in range(batch_size)]

    while Y_turbo.shape[0] < total_trials and time.monotonic() - startT < max_time:
        print(f"Trial {Y_turbo.shape[0]}")

        # def model_h(xx):
        #     model.eval()
        #     mvn = model.posterior(xx.unsqueeze(0).unsqueeze(0))
        #     # return mvn.rsample().sum()
        #     return mvn.mean.sum() #- .05 * mvn.variance.sqrt().sum()

        def mean_f(X):
            epsilon = torch.empty(100)
            trunc_normal_(epsilon, 0, 0.5, -1, 1)
            epsilon = epsilon.mean().abs().item()
            posterior = model.likelihood(model(X.unsqueeze(0)))
            # return posterior.mean.sum()
            # return posterior.mean.sum() - model.likelihood.noise_covar.noise.item() * posterior.stddev.sum()
            return posterior.mean.sum() - epsilon * posterior.stddev.sum()

        for i in range(batch_size):
            x_center = states[i].x_center
            grad = torch.autograd.functional.jacobian(mean_f, x_center)
            
            # # >>> adjust length scale >>>
            # # if train_Y.shape[0] < Y_turbo.shape[0]:  # check whether new data are added
            # #     train_Y = standardize(Y_turbo)
            # # while torch.norm(grad) < 1e-4 and num_steps >= 1:  # Check if any model is too complex
            # #     num_steps = int(num_steps * 0.9)
            # #     model = train_gp(train_x=X_turbo, train_y=train_Y.view(-1), use_ard=True, num_steps=num_steps, hypers={})
            # #     print("Re-select more coarse GP.")
            # #     grad = torch.autograd.functional.jacobian(mean_f, x_center)
            # # 'covar_module.base_kernel.raw_lengthscale'
            # while torch.norm(grad) < 1e-5:
            #     model.initialize(**{
            #         'mean_module.constant': model.mean_module.constant.detach(),
            #         "covar_module.outputscale": model.covar_module.outputscale.detach(),
            #         "covar_module.base_kernel.lengthscale": torch.minimum(model.covar_module.base_kernel.lengthscale.detach() * 2, torch.tensor(2.)),
            #         'likelihood.noise_covar.noise': model.likelihood.noise.detach()
            #     })
            #     grad = torch.autograd.functional.jacobian(mean_f, x_center)
            #     print("Adjust length scale")
            # # <<< adjust length scale <<<

            # Check if any TR needs to be restarted
            grad_norm = torch.norm(grad)
            if states[i].restart_triggered or grad_norm < 1e-5:
                print(f"TR{i} restart. Length {states[i].length}. Grad norm {grad_norm}")
                if restart_strategy == 'random':
                    # >>> Random restart >>>
                    X_rand = torch.from_numpy(latin_hypercube(n_init // batch_size, dim)).to(dtype=dtype, device=device)
                    Y_rand = torch.tensor(
                        [eval_func(x) for x in X_rand], dtype=dtype, device=device
                    ).unsqueeze(-1)
                    idx = Y_rand.view(-1).argmin()
                    x_center, y_center = X_rand[idx], Y_rand[idx].item()

                    X_turbo = torch.cat((X_turbo, X_rand), dim=0) # Append data
                    Y_turbo = torch.cat((Y_turbo, Y_rand), dim=0)
                    wallclocks.extend([time.monotonic() - startT] * X_rand.shape[0])
                    # <<< Random strategy <<<
                elif restart_strategy == 'PES':
                    # >>> employ Acq to restart >>>
                    # X_rand = torch.from_numpy(latin_hypercube(10000, dim)).to(dtype=dtype, device=device)
                    # ei_rand = ei(X_rand, train_Y.min().item(), model)
                    # idx = ei_rand.argmax()
                    # x_center = X_rand[idx]
                    # y_center = torch.tensor(eval_func(x_center))

                    # logEI = LogExpectedImprovement(model, train_Y.min().item(), maximize=False)
                    optimal_inputs, _ = get_optimal_samples(
                        model, bounds=torch.stack((torch.zeros(dim), torch.ones(dim))).to(dtype=dtype, device=device),
                        num_optima=12, maximize=False
                    )
                    # qJES = qJointEntropySearch(model=model, optimal_inputs=optimal_inputs, optimal_outputs=optimal_outputs, estimation_type='LB', maximize=False)
                    # qMES = qMaxValueEntropy(model, candidate_set=torch.rand(1000, dim, dtype=dtype, device=device), maximize=False)
                    AF = qPredictiveEntropySearch(model, optimal_inputs=optimal_inputs, maximize=False)
                    es = cma.CMAEvolutionStrategy(  # create the CMA-ES optimizer
                        x0=np.ones(dim) * .5,
                        sigma0=0.2,
                        inopts={"bounds": [0, 1], "popsize": 50, 'maxiter':200},
                    )

                    with torch.no_grad():
                        while not es.stop():
                            xs = es.ask()
                            xs_th = torch.from_numpy(np.vstack(xs)).to(dtype=dtype, device=device)
                            ys = -AF(xs_th.unsqueeze(-2)).cpu().numpy()
                            es.tell(xs, ys)  # return the result to the optimize
                            # es.disp()

                    print(f"AF value: {-es.result[1]}")
                    x_center = torch.from_numpy(es.best.x).to(dtype=dtype, device=device)
                    y_center = torch.tensor(eval_func(x_center))

                    X_turbo = torch.cat((X_turbo, x_center.view(1, -1)), dim=0) # Append data
                    Y_turbo = torch.cat((Y_turbo, y_center.view(1, 1)), dim=0)
                    wallclocks.append(time.monotonic() - startT)
                    # <<< employ Acq to restart <<<
                else:
                    raise NotImplementedError
                train_Y, _, _ = standardize(Y_turbo)
                model.set_train_data(X_turbo, train_Y.view(-1), strict=False)
                states[i] = TurboState(dim, batch_size, x_center, y_center)
            else:
                # ind = grad.abs() > 1e-8
                hessian = torch.autograd.functional.hessian(mean_f, x_center)
                # weights = torch.diag(hessian)
                # gradW = grad / weights  # grad @ diag(W)^-1
                # hessianW = (1/weights).unsqueeze(1) * hessian / weights  # diag(W)^-1 @ H @ diag(W)^-1
                # assert not torch.isnan(hessianW).any()
                # hessian_np, grad_np = hessianW.numpy(), gradW.numpy()
                hessian_np, grad_np = hessian.numpy(), grad.numpy()
                def mk(step: np.ndarray):
                    value = .5 * np.dot(step, hessian_np @ step) + np.dot(grad_np, step)
                    return value
                # minimize 1/2 p^T B p + g^T p, subject to lb <= p <= ub
                x_center_np = x_center.numpy()
                # ub = np.minimum(states[i].length, 1.-x_center_np-1e-5)
                # lb = np.maximum(-states[i].length, -x_center_np+1e-5)

                # >>> SCALING >>>
                weights = model.covar_module.lengthscale.cpu().detach().numpy().ravel()
                # weights = model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
                weights = weights / weights.mean()  # This will make the next line more stable
                weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
                ub = np.minimum(states[i].length * weights, 1.-x_center_np-1e-5)
                lb = np.maximum(-states[i].length * weights, -x_center_np+1e-5)
                # <<< SCALING <<<

                assert (lb < ub).all()

                ## >>> L-BFGS-B >>>
                # sols, vals, x0s, y0s = solveSubProblem(mk, lb, ub)
                # # dm = distance.cdist(sols + x_center_np, X_turbo.numpy(), 'euclidean').min(1)
                # ind = np.linalg.norm(sols, axis=1) < 1e-5  # candidate cannot too close to training set.

                # sols[ind], vals[ind] = x0s[ind], y0s[ind] # print("Too close to training set. Select an initial random point!")
                # ind = vals.argmin()
                # sol, val = sols[ind], vals[ind]
                ## <<< L-BFGS-B <<<

                ## >>> CMA-ES >>>
                sol, _ = cma.fmin2(mk, (lb+ub)/2, 0.5, {'bounds': [lb, ub], 'verbose': -9, 'maxfevals': 5000})
                val = mk(sol)
                ## <<< CMA-ES <<<

                if val > -tol: # Fail to find a solution
                    states[i].restart_triggered = True
                else:
                    # assert not torch.isnan(step).any()
                    sol = torch.from_numpy(sol).to(x_center)
                    # sol /= weights
                    x_next = torch.clamp(x_center + sol, 0., 1.)
                    y_next = torch.tensor(eval_func(x_next), dtype=dtype, device=device)
                    update_state(states[i], x_next.unsqueeze(0), y_next.view(1, 1))
                    # print(f"TR{i} length: {states[i].length}")
                    
                    X_turbo = torch.cat((X_turbo, x_next.unsqueeze(0)), dim=0)  # Append data
                    Y_turbo = torch.cat((Y_turbo, y_next.view(1, 1)), dim=0)
                    wallclocks.append(time.monotonic() - startT)
                    train_Y, _, _ = standardize(Y_turbo)
                    model.set_train_data(X_turbo, train_Y.view(-1), strict=False)


        model = train_vanillaGP(X_turbo, train_Y.view(-1), True, num_steps_gp)
        # fine_tune(model, X_turbo, train_Y.view(-1), 5)

    return X_turbo, Y_turbo, np.array(wallclocks)
