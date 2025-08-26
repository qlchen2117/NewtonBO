import numpy as np
from scipy.stats import qmc
import numpy as np
from numpy.linalg import solve
from scipy.linalg import cholesky
from scipy.optimize import direct, Bounds
from typing import Callable
import time

from .maximize_acq import maximize_acq
from .ksir import sir_dir
from .covSEard import cov_se_ard
from .covSEiso import cov_se_iso


class SIRBO:
    def __init__(
            self, high_dim: int, dim: int, bounds: np.ndarray,
            init_py: np.ndarray, init_f: np.ndarray, hyp: np.ndarray,
            noise: float, cov_model: str, optimizer='CMAES'
    ):
        """
        Args:
            high_dim: ambient dimension.
            dim: embedding dimension.
            bounds: (high_dim, 2)
            init_py: (num, high_dim) init points
            init_f:  (num,) objective values of init points
            hyp: (dim+1,) hyperparameters
            cov_model: covariance model
        """
        self.noise, self.optimizer = noise, optimizer

        if cov_model == 'se':
            self.cov_model = cov_se_iso
        elif cov_model == 'ard':
            self.cov_model = cov_se_ard
        else:
            raise NotImplementedError
        self.kernel_type = cov_model
        self.hyp, self.bounds, self.dim = hyp, bounds, dim
        self.cur_hyp, self.exploit_count, self.hyper_bound = hyp[0], 0, np.log([0.01, 50])

        self.prior_mean, self.max_exploit_count, self.high_dim = 0, 5, False

        self.trainX, self.emb = np.zeros((3000, high_dim)), np.zeros((3000, dim))
        self.trainX[:init_py.shape[0], :] = init_py

        self.fvals, self.mm, self.num = init_f, init_py.shape[0], init_py.shape[0]
        self.trainY = (self.fvals - self.fvals.mean()) / self.fvals.std(ddof=1)

        num_of_slice = min(self.trainY.shape[0], dim+1)
        self.project_mat = sir_dir(init_py, self.trainY.ravel(), num_of_slice, dim)  # shape(high_dim, dim)
        init_pt = init_py @ self.project_mat  # shape(num, dim)

        lower_mat = self.cov_model(hyp, init_pt, init_pt) + self.noise * np.eye(init_pt.shape[0])
        self.lower_mat = cholesky(lower_mat, lower=True)

        self.emb[:init_py.shape[0], :] = init_pt

        idx = np.argmin(init_f)
        self.min_val, self.min_x, self.min_emb = init_f[idx].item(), self.trainX[idx], self.emb[idx]

        self.copts = {'maxfun': 2000, 'TolFun': 1e-8,
                'LBounds': bounds[:, 0], 'UBounds': bounds[:, 1]}


    def update_kernel(self, xx, learn_hyper):
        """Update Kernel after seeing new data.
        """
        mm = self.mm
        k_x = self.cov_model(self.hyp, self.emb[:mm], xx)
        k_tt = self.cov_model(self.hyp, xx, xx)
        mm += xx.shape[0]
        self.mm = mm

        if not learn_hyper:
            # If not learning Hyper-parameter then update the kernel matrix.
            vv = solve(self.lower_mat, k_x)  # (num, 1)
            d_t = np.sqrt(k_tt + self.noise - vv.T @ vv)
            self.lower_mat = np.vstack((
                np.hstack((self.lower_mat, np.zeros((mm-1, 1)))),
                np.vstack((vv, d_t)).T
            ))
        else:
            # If learn hyper-parameter then recompute kernel matrix.
            if self.kernel_type == 'custom':
                raise NotImplementedError
                model.ell = model.cov_model(model.hyp, 0, model.records[:m], model.records[:m]) + np.eye(model.n) * model.noise
            else:
                self.lower_mat = self.cov_model(self.hyp, self.emb[:mm], self.emb[:mm]) + np.eye(self.mm) * self.noise
            # Do Cholesky decomposition.
            self.lower_mat = cholesky(self.lower_mat, lower=True)


    def neg_log_marginal_likelihood(self, hyper_param):
        """negative log marginal likelihood calculations.
        """
        if self.kernel_type == 'custom':
            # Compute kernel matrix
            raise NotImplementedError
            kernel_mat = model.cov_model(hyper_param,)
        else:
            # Compute kernel matrix
            kernel_mat = self.cov_model(
                hyper_param, self.emb[:self.num, :], self.emb[:self.num, :]
            ) + np.eye(self.num) * self.noise
        kernel_chol = cholesky(kernel_mat, lower=False)
        assert self.trainY.ndim == 2
        alpha = solve(kernel_chol, solve(kernel_chol.T, self.trainY))
        return self.trainY.T @ alpha + 2 * np.sum(np.log(np.diag(kernel_chol)))


    def learn_hyper_param(self):
        ss = len(self.hyp) - 1
        def fn_wrapper(x):
            return self.neg_log_marginal_likelihood(
                np.hstack((np.ones(ss) * x, self.hyp[-1]))
            )
        # Learn Hyper-parameters by using direct
        res = direct(fn_wrapper, Bounds(self.hyper_bound[0], self.hyper_bound[1]) , maxfun=500, maxiter=300)
        new_hyp = np.ones(ss) * res.x
        return new_hyp


    def update_model(self, f_t, final_x_at_minY, batch_size):
        # Decide whether the model has been exploting too much.
        # if self.num % 20 == 0:
        xx = np.vstack((self.trainX[:self.num, :], final_x_at_minY))
        yy = np.vstack((self.fvals, f_t))
        yy = (yy - yy.mean()) / yy.std(ddof=1)
        num_of_slice = min(yy.shape[0], self.project_mat.shape[1]+1)
        self.project_mat = sir_dir(xx, yy.ravel(), num_of_slice, self.project_mat.shape[1])

        final_x_at_min = final_x_at_minY @ self.project_mat  # project down
        _, var = self.mean_var(final_x_at_min)
        if (np.sqrt(var) <= 2 * np.sqrt(self.noise)).any():
            self.exploit_count += 1
        else:
            self.exploit_count = 0

        # Book keeping
        self.trainX[self.num: self.num+batch_size] = final_x_at_minY
        self.emb[self.num: self.num+batch_size] = final_x_at_min
        self.fvals = np.vstack((self.fvals, f_t))
        self.trainY = (self.fvals - self.fvals.mean()) / self.fvals.std(ddof=1)
        self.num += batch_size

        idx = f_t.argmin()
        f_t_min, x_min, emb_min = f_t[idx], final_x_at_minY[idx], final_x_at_min[idx]
        if f_t_min < self.min_val:
            self.min_x = x_min
            self.min_emb = emb_min
            self.min_val = f_t_min

        learn_hyper = True
        # if self.num - 1 >= 10 and (self.num - 1) % 20 == 0:
        #     # Routine optimization of hyper-parameters every 10 iterations.
        #     learn_hyper = True
        # else:
        #     learn_hyper = False

        if self.exploit_count >= self.max_exploit_count:
            # If BO has been exploiting too much then lower upper bound of hyper-parameters
            self.hyper_bound = np.hstack((self.hyper_bound[0], self.cur_hyp + np.log(0.9)))
            self.exploit_count = 0
            learn_hyper = True

        if learn_hyper:
            # Learn hyper-parameters.
            new_hyp = self.learn_hyper_param()
            self.cur_hyp = new_hyp[0]
            self.hyp[:-1] = new_hyp

        self.update_kernel(final_x_at_min, learn_hyper)

    def mean_var(self, xx: np.ndarray):
        """
        """
        cov_model = self.cov_model
        k_tt = cov_model(self.hyp, xx, xx)
        k_x = cov_model(self.hyp, self.emb[:self.num, :], xx)

        intermediate = solve(self.lower_mat.T, solve(self.lower_mat, k_x))
        mean = self.prior_mean + intermediate.T @ (self.trainY - self.prior_mean)
        var = np.diag(k_tt - k_x.T @ intermediate)

        return mean, var

def opt(objective: Callable, num_iter: int, max_time, batch_size: int, model, startT, wallclocks):
    dopt = {'maxfun': 500, 'maxiter': 200, 'show_its': False, 'batch_size': batch_size}
    it = 0
    while it < num_iter and time.monotonic() - startT < max_time:
        candidates, _ = maximize_acq(model, dopt, 'ei')

        f_t = np.array([objective((final_x_at_min + 1) / 2) for final_x_at_min in candidates])
        model.update_model(f_t[:, np.newaxis], candidates, batch_size)
        wallclocks.extend([time.monotonic() - startT]*batch_size)
        it += 1
    return model, wallclocks


def sir_bo(obj_fct, dim, high_dim, init_size, n_iter, max_time, batch_size):
    """total_iter: total number of iterations.
        dim: embedding dimension.
        high_dim: ambient dimension.
    """
    bounds = np.array([[-1, 1]] * high_dim)  # Initialize bounds.
    startT = time.monotonic()
    init_py = qmc.Sobol(d=high_dim, scramble=True).random(n=init_size) * 2 - 1  # Initial point.
    # init_py = np.array([[0.1, -0.2], [0.3, -0.4]])  # for test
    init_f = np.array([obj_fct((x+1)/2) for x in init_py])[:, np.newaxis]  # Evaluate initial point.
    wallclocks = [time.monotonic() - startT] * init_size
    hyp = np.hstack((np.ones(dim) * 0.1, 1))  # Setup initial hyper-parameters.
    hyp = np.log(hyp)
    # Initialize model.
    model = SIRBO(high_dim, dim, bounds, init_py, init_f, hyp, 1e-6, 'ard')
    # Do optimization.
    model, wallclocks = opt(obj_fct, n_iter, max_time, batch_size, model, startT, wallclocks)
    return model.trainX, model.fvals, np.array(wallclocks)


