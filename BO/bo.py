""" Adapt from BoTorch-0.12 """ 
import time
from math import sqrt, log
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.predictive_entropy_search  import qPredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples

from botorch.acquisition.analytic import LogProbabilityOfImprovement, LogExpectedImprovement, UpperConfidenceBound

from gpytorch.kernels import RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Warp
from gpytorch.priors.torch_priors import LogNormalPrior

from botorch.exceptions import BadInitialCandidatesWarning
import warnings
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)
MIN_INFERRED_NOISE_LEVEL = 1e-4


def bo(eval_func, ndims, n_init, n_iterations, max_time, batch_size, acq, use_input_warp=False, optimizer='local_search'):
    startT = time.monotonic()
    train_X = SobolEngine(dimension=ndims, scramble=True).draw(n_init).to(dtype=dtype, device=device)
    origin_Y = torch.tensor(
        [eval_func(x) for x in train_X], dtype=dtype, device=device
    ).unsqueeze(-1)
    print(f"Best initial point: {origin_Y.max().item():.3f}")
    wallclocks = [time.monotonic() - startT] * n_init
    it = 0
    while it < n_iterations and time.monotonic() - startT < max_time:
        print(f"Iteration {it}...")
        train_Y = (origin_Y - origin_Y.mean()) / origin_Y.std()

        if use_input_warp:
            # initialize input_warping transformation
            warp_tf = Warp(
                indices=list(range(ndims)),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
            )
        else:
            warp_tf = None

        
        # noise_lb, noise_guess = 0.0008, 0.01
        # n_constr = GreaterThan(noise_lb)
        # n_prior  = LogNormalPrior(np.log(noise_guess), 0.5)
        # lik = GaussianLikelihood(noise_constraint = n_constr, noise_prior = n_prior)
        # gp = SingleTaskGP(
        #     train_X=X, train_Y=train_Y,
        #     input_transform=warp_tf,
        #     # covar_module=ScaleKernel(RBFKernel(ard_num_dims=ndims)),
        #     likelihood=lik
        # )
        # gp.likelihood.noise = max(1e-2, noise_lb)

        # >>> [Hvarfner2024vanilla] settings
        lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ndims) * 0.5, scale=SQRT3)
        covar_module = RBFKernel(
            ard_num_dims=ndims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=lengthscale_prior.mode)
        )

        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, initial_value=noise_prior.mode)
        )
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y, covar_module=covar_module, likelihood=likelihood, outcome_transform=None, input_transform=None)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        # <<< [Hvarfner2024vanilla] settings

        # >>> Fit model
        fit_gpytorch_mll(mll)
        # <<< Fit model

        # >>> Construct an acquisition function
        if acq == 'EI':
            acq_function = qExpectedImprovement(gp, train_Y.max())
        elif acq == 'UCB':
            iter  = max(1, X.shape[0] // batch_size)
            upsi  = 0.5
            delta = 0.01
            kappa = upsi * 2 * ((2.0 + ndims / 2.0) * np.log(iter) + np.log(np.pi**2 / delta))
            acq_function = qUpperConfidenceBound(gp, beta=kappa)
        elif acq == 'PI':
            from botorch.acquisition.analytic import ProbabilityOfImprovement
            acq_function = ProbabilityOfImprovement(gp, best_f=train_Y.max())
        elif acq == 'PES':
            optimal_inputs, _ = get_optimal_samples(
                gp, bounds=torch.stack((torch.zeros(ndims), torch.ones(ndims))).to(dtype=dtype, device=device),
                num_optima=12
            )
            acq_function = qPredictiveEntropySearch(gp, optimal_inputs=optimal_inputs)
        # <<< Construct an acquisition function

        def fn(x):
            X = torch.tensor(x, dtype=dtype, device=device).view(-1, 1, ndims)
            return acq_function(X).item() * -1

        # >>> Optimize the acquisition function
        if optimizer == 'local_search':
            X_next, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=torch.cat((torch.zeros(1, ndims), torch.ones(1, ndims))).to(dtype=dtype, device=device),
                q=batch_size, num_restarts=16, raw_samples=1024,
            )  # shape(batch_size, d_e)
        elif optimizer == 'differential_evolution':
            from scipy.optimize import differential_evolution
            result = differential_evolution(fn, [(0, 1)]*ndims)
            X_next = torch.tensor(result.x, dtype=dtype, device=device).view(-1, ndims)
        elif optimizer == 'CMA-ES':
            import cma
            es = cma.CMAEvolutionStrategy(  # create the CMA-ES optimizer
                x0=np.ones(batch_size * ndims) * .5,
                sigma0=0.2,
                inopts={"bounds": [0, 1], "popsize": 50, 'maxiter':100},
            )
            with torch.no_grad():
                while not es.stop():
                    xs = np.vstack(es.ask())
                    xs_th = torch.from_numpy(xs).to(dtype=dtype, device=device).view(50*batch_size, ndims)  # shape(popsize x batch_size, dim)
                    ys = -acq_function(xs_th.unsqueeze(-2)).view(50, batch_size).cpu().numpy()  # shape(popsize, batch_size)
                    es.tell(xs, ys.sum(1))  # return the result to the optimize
            X_next = torch.from_numpy(es.best.x).to(dtype=dtype, device=device).view(batch_size, ndims)  # shape(batch_size, dim)
        elif optimizer == 'direct':
            from scipy.optimize import direct, Bounds
            result = direct(fn, Bounds(np.zeros(ndims), np.ones(ndims)))
            X_next = torch.tensor(result.x, dtype=dtype, device=device).view(-1, ndims)
        elif optimizer == 'MACE':
            logPI = LogProbabilityOfImprovement(gp, best_f=train_Y.max())
            logEI = LogExpectedImprovement(gp, train_Y.max())

            iter  = max(1, train_X.shape[0] // batch_size)
            upsi  = 0.5
            delta = 0.01
            kappa = upsi * 2 * ((2.0 + ndims / 2.0) * np.log(iter) + np.log(np.pi**2 / delta))

            ucb = UpperConfidenceBound(gp, beta=kappa)
            X_next, _ = optimize_acqf_moo(logEI, ucb, logPI, ndims)
        else:
            raise NotImplementedError
        # <<< Optimize the acquisition function

        Y_next = torch.tensor(
            [eval_func(x) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        wallclocks.extend([time.monotonic() - startT] * batch_size)
        train_X = torch.cat((train_X, X_next), dim=0)
        origin_Y = torch.cat((origin_Y, Y_next), dim=0)
        it += 1
    return train_X, origin_Y, np.array(wallclocks)


def optimize_acqf_moo(logEI, ucb, logPI, ndims):
    from pymoo.core.problem import Problem

    class BOProblem(Problem):

        def __init__(self):
            super().__init__(n_var=ndims, n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            X = torch.tensor(x, dtype=dtype, device=device).view(-1, 1, ndims)
            with torch.no_grad():
                log_ei = logEI(X).view(-1, 1)
                ucb_ = ucb(X).view(-1, 1)
                log_pi = logPI(X).view(-1 ,1)
            out["F"] = torch.cat((log_ei, ucb_, log_pi), dim=1).cpu().numpy() * -1

    problem = BOProblem()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        eliminate_duplicates=True
    )

    from pymoo.termination import get_termination
    termination = get_termination("n_gen", 100)
    from pymoo.optimize import minimize

    res = minimize(problem, algorithm, termination)

    idx = np.random.choice(len(res.X), 1, replace = False)
    return torch.tensor(res.X[idx], dtype=dtype, device=device),\
        torch.tensor(res.F[idx], dtype=dtype, device=device)


def bo_map(eval_func, ndims, n_init, n_iterations, B, batch_size=1):
    B = B.to(dtype=dtype, device=device)
    X = SobolEngine(dimension=ndims, scramble=True, seed=0).draw(n_init).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [eval_func(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    print(f"Best initial point: {Y.max().item():.3f}")
    for i in range(n_iterations):
        train_X = X @ B
        train_Y = (Y - Y.mean()) / Y.std()
        # fit model
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        # Construct an acquisition function
        ei = qExpectedImprovement(gp, train_Y.max())
        # ucb = UpperConfidenceBound(gp, beta=0.2)
        # Optimize the acquisition function

        def acq_function(x):
            return ei(x @ B)

        X_next, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=torch.cat((torch.zeros(1, ndims), torch.ones(1, ndims))).to(dtype=dtype, device=device),
            q=batch_size, num_restarts=10, raw_samples=512,
        )  # shape(batch_size, d_e)

        Y_next = torch.tensor(
            [eval_func(x) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
    return X, Y


# 分析加入旋转矩阵对BO的影响
# if __name__ == "__main__":
#     BATCH_SIZE = 1
#     N_INIT = 10
#     N_ITERACTIONS = 100

#     from botorch.test_functions import Ackley
#     fun = Ackley(dim=10).to(dtype=dtype, device=device)
#     fun.bounds[0, :].fill_(-5)
#     fun.bounds[1, :].fill_(10)
#     DIM = fun.dim


#     def eval_objective(x):
#         """This is a helper function we use to unnormalize and evalaute a point"""
#         lb, ub = fun.bounds
#         return fun(lb + (ub - lb) * x) * -1
#     X, Y = bo(eval_func=eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE)
#     # X, Y = bo_map(eval_func=eval_objective, ndims=DIM, n_init=N_INIT, B=torch.randn(DIM, DIM),
#     #               n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE)
#     Y_np = -1 * Y.cpu()
#     import numpy as np
#     import matplotlib.pyplot as plt
#     fx = np.minimum.accumulate(Y_np)
#     plt.plot(fx, marker="", lw=3)

#     plt.plot([0, len(Y)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
#     plt.show()

# 分析维度升高对BO的影响
# if __name__ == "__main__":
#     BATCH_SIZE = 1
#     N_INIT = 10
#     N_ITERACTIONS = 100

#     from botorch.test_functions import Ackley
#     import numpy as np
#     import matplotlib.pyplot as plt
#     ax1 = plt.subplot(1,2,1)
#     ax2 = plt.subplot(1,2,2)
#     for i in range(1, 7):
#         DIM = i * 5
#         fun = Ackley(dim=DIM).to(dtype=dtype, device=device)
#         fun.bounds[0, :].fill_(-5)
#         fun.bounds[1, :].fill_(10)

#         def eval_objective(x):
#             """This is a helper function we use to unnormalize and evalaute a point"""
#             lb, ub = fun.bounds
#             return fun(lb + (ub - lb) * x) * -1
#         _, Y = bo(eval_func=eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE)
#         Y_np = -1 * Y.cpu()
#         fx = np.minimum.accumulate(Y_np)
#         ax1.plot(fx, marker="", lw=3, label=f"bo D{DIM}")

#         _, Y = random_search(eval_objective, DIM, N_INIT+N_ITERACTIONS)
#         Y_np = -1 * Y.cpu()
#         fx = np.minimum.accumulate(Y_np)
#         ax2.plot(fx, marker="", lw=3, label=f"RS D{DIM}")

#     ax1.plot([0, len(Y)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
#     ax1.set_title("Ackley")
#     ax1.grid(True)
#     ax1.set_xlabel("Number of evaluations")
#     ax1.set_ylabel("Best value found")
#     ax1.legend(loc="best")

#     ax2.plot([0, len(Y)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
#     ax2.set_title("Ackley")
#     ax2.grid(True)
#     ax2.set_xlabel("Number of evaluations")
#     ax2.set_ylabel("Best value found")
#     ax2.legend(loc="best")
#     plt.show()


if __name__ == '__main__':
    import torch
    import numpy as np
    from botorch.test_functions import Branin, Ackley
    N_EPOCH = 5
    DIM = 10
    EM_DIM = 2
    N_ITERACTIONS = 50
    N_INIT = 10
    BATCH_SIZE = 5
    TOTAL_TRIALS = N_INIT + N_ITERACTIONS * BATCH_SIZE

    func = Ackley(dim=EM_DIM)
    func.bounds[0, :].fill_(-5)
    func.bounds[1, :].fill_(10)

    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        x = x[:EM_DIM]
        lb, ub = func.bounds
        return func(lb + (ub - lb) * x) * -1

    Y_bo = np.empty((N_EPOCH, TOTAL_TRIALS))
    for i in range(N_EPOCH):
        _, Y = bo(eval_objective, ndims=DIM, n_init=N_INIT, n_iterations=N_ITERACTIONS, batch_size=BATCH_SIZE, acq='ucb')
        Y_np = Y.cpu().numpy() * -1
        Y_bo[i] = Y_np.squeeze()

    # Save Results
    np.save(f"{type(func).__name__}-bo.npy", Y_bo)

    # Read results
    Y_bo = np.mean(np.load(f"{type(func).__name__}-bo.npy"), axis=0)

    # Draw pictures
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.minimum.accumulate(Y_bo), label="BO")

    ax.plot([0, TOTAL_TRIALS], [func.optimal_value, func.optimal_value], "k--", lw=3)

    ax.grid(True)
    ax.set_title(f"{type(func).__name__}, D = {DIM}", fontsize=20)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    # ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=20)
    # ax.set_ylim([0, 8])
    ax.legend()
    plt.show()
