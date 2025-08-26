""" Taken from https://github.com/lamda-bbo/MCTS-VS"""

import numpy as np
import gymnasium as gym
from .filter import RunningStat


class RLEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        # self.env.seed(seed)
        state_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        
        self.dims = state_dims * action_dims
        self.policy_shape = (action_dims, state_dims)
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.rs = RunningStat(state_dims)

        self.num_rollouts = 3

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_shape)
        total_r = 0
        n_samples = 0
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            while True:
                self.rs.push(obs)
                norm_obs = (obs - self.rs.mean) / (self.rs.std + 1e-6)
                action = np.dot(M, norm_obs)
                obs, r, terminated, truncated, _ = self.env.step(action)
                total_r += r
                n_samples += 1
                if terminated or truncated:
                    break
        
        return total_r / self.num_rollouts, n_samples / 3
    

class RLBenchmark:
    def __init__(self, func, dims, valid_idx):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb
        self.ub = func.ub
        # self.save_config = save_config
        
        self.counter = 0
        self.n_samples = 0
        # self.start_time = time.time()
        self.curt_best = float('-inf')
        self.best_value_trace = []

    def __call__(self, x):
        assert len(x) == self.dims
        result, n_samples = self.func(x[self.valid_idx])
        # self.track(result, n_samples)
        return result

    # def track(self, result, n_samples):
    #     self.counter += 1
    #     self.n_samples += n_samples
    #     if result > self.curt_best:
    #         self.curt_best = result
    #     self.best_value_trace.append((
    #         self.counter,
    #         self.curt_best,
    #         time.time() - self.start_time,
    #         self.n_samples
    #     ))

    #     if self.counter % 50 == 0:
    #         df_data = pd.DataFrame(self.best_value_trace, columns=['x', 'y', 't', 'n_samples'])
    #         save_results(
    #             self.save_config['root_dir'],
    #             self.save_config['algo'],
    #             self.save_config['func'],
    #             self.save_config['seed'],
    #             df_data,
    #         )

def testRLenv(func_name):
    assert func_name in ['HalfCheetah', 'Walker2d', 'Hopper']

    if func_name == 'HalfCheetah' or func_name == 'Walker2d':
        rlProblem = RLBenchmark(RLEnv(func_name+'-v4'), 102, list(range(102)))
    elif func_name == 'Hopper':
        rlProblem = RLBenchmark(RLEnv('Hopper-v4'), 33, list(range(33)))
    xs = np.random.uniform(rlProblem.lb, rlProblem.ub, (3, rlProblem.dims))
    ys = [rlProblem(xx) for xx in xs]
    print(ys)
