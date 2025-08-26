#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple
from warnings import warn
from scipy.stats import uniform
import numpy as np

from .utils import add_fixed_features, rejection_sample


class ALEBOInitializer:
    """Sample in a low-dimensional linear embedding, to initialize ALEBO.

    Generates points on a linear subspace of [-1, 1]^D by generating points in
    [-b, b]^D, projecting them down with a matrix B, and then projecting them
    back up with the pseudoinverse of B. Thus points thus all lie in a linear
    subspace defined by B. Points whose up-projection falls outside of [-1, 1]^D
    are thrown out, via rejection sampling.

    To generate n points, we start with nsamp points in [-b, b]^D, which are
    mapped down to the embedding and back up as described above. If >=n points
    fall within [-1, 1]^D after being mapped up, then the first n are returned.
    If there are less than n points in [-1, 1]^D, then b is constricted
    (halved) and the process is repeated until there are at least n points in
    [-1, 1]^D. There exists a b small enough that all points will project to
    [-1, 1]^D, so this is guaranteed to terminate, typically after few rounds.

    Args:
        B: A (dxD) projection down.
        nsamp: Number of samples to use for rejection sampling.
        init_bound: b for the initial sampling space described above.
        seed: seed for UniformGenerator
    """

    def __init__(
        self,
        B: np.ndarray,
        nsamp: int = 10000,
        init_bound: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        warn("ALEBOInitializer is deprecated.", DeprecationWarning)
        # pyre-fixme[4]: Attribute must be annotated.
        self.Q = np.linalg.pinv(B) @ B  # Projects down to B and then back up
        self.nsamp = nsamp
        self.init_bound = init_bound
        self.deduplicate=False
        self._rs = np.random.RandomState(seed=seed)

    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if n > self.nsamp:
            raise ValueError("n > nsamp")
        # The projection is from [-1, 1]^D.
        for b in bounds:
            assert b == (-1.0, 1.0)
        # The following can be easily handled in the future when needed
        assert linear_constraints is None
        assert fixed_features is None
        # Do gen in the high-dimensional space.
        bounds=[(0.0, 1.0)] * self.Q.shape[0]
        max_draws = self.nsamp
        # Always rejection sample, but this only rejects if there are
        # constraints or actual duplicates and deduplicate is specified.
        # If rejection sampling fails, fall back to polytope sampling
        points, attempted_draws = rejection_sample(
            gen_unconstrained=self._gen_unconstrained,
            n=n,
            d=len(bounds),
            tunable_feature_indices=np.arange(len(bounds)),
            deduplicate=self.deduplicate,
            max_draws=max_draws,
            fixed_features=fixed_features,
        )
        X01, w = points, len(points)
        finished = False
        b = float(self.init_bound)
        while not finished:
            # Map to [-b, b]
            X_b = 2 * b * X01 - b
            # Project down to B and back up
            X = X_b @ np.transpose(self.Q)
            # Filter out to points in [-1, 1]^D
            X = X[(X >= -1.0).all(axis=1) & (X <= 1.0).all(axis=1)]
            if X.shape[0] >= n:
                finished = True
            else:
                b = b / 2.0  # Constrict the space
        X = X[:n, :]
        return X, np.ones(n)


    def _gen_unconstrained(
        self,
        n: int,
        d: int,
        tunable_feature_indices: np.ndarray,
        fixed_features: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Generate n points, from an unconstrained parameter space, using _gen_samples.

        Args:
            n: Number of points to generate.
            d: Dimension of parameter space.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            tunable_feature_indices: Parameter indices (in d) which are tunable.

        Returns:
            An (n x d) array of generated points.

        """
        tunable_points = self._gen_samples(n=n, tunable_d=len(tunable_feature_indices))
        points = add_fixed_features(
            tunable_points=tunable_points,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )
        return points

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate samples from the scipy uniform distribution.

        Args:
            n: Number of samples to generate.
            tunable_d: Dimension of samples to generate.

        Returns:
            samples: An (n x d) array of random points.

        """
        return uniform.rvs(size=(n, tunable_d), random_state=self._rs)  # pyre-ignore


