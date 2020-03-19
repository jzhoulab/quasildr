# -*- coding: utf-8 -*-
"""
This module implements nonparametric ridge estimation and relevant
functions.  The main class is `Scms` which provides ridge estimation
and bootstrap functions.
"""
import warnings

import numpy as np
from multiprocess import Pool
from numpy.core.umath_tests import matrix_multiply
from numpy.lib.index_tricks import as_strided
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

from .external import neighbors


def vdist(a, b):
    """
    This function is similar to cdist but returning vectors rather than norm of vectors
    implemented via virtual copies of a and b

    Parameters
    ----------
    a,b : np.ndarray
        2D arrays of size (M,D), (N,D) for which differences for all pairwise combinations
        of rows in the two arrays will be computed. The number of columns have to the same
        between a and b.

    Returns
    ----------
    numpy.array
        3D array of size (N, M, D), containing all pairwise differences of rows in a and b.
    """
    return as_strided(a, strides=(0, a.strides[0], a.strides[1]),
                      shape=(b.shape[0], a.shape[0], a.shape[1])) - \
           as_strided(b, strides=(b.strides[0], 0, b.strides[1]),
                      shape=(b.shape[0], a.shape[0], b.shape[1]))


def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like

    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None

    Returns
    -------
    array_like, numpy.array
        * Bootstrap resampled array
        * 1D array of indices corresponding to input.
    """
    if n is None:
        n = len(X)

    resample_i = np.floor(np.random.rand(n) * len(X)).astype(int)
    samples = X[resample_i]
    return samples, resample_i


def multilevel_compression(data, n=5000, threshold=None, k=20):
    """
    Iteratively combine nearest neighbor pairs below distance threshold.
    This function can be used to reduce the number of data points to represent the whole dataset
    without greatly affect the density estimation quality. Most of the data points in low density
    areas will be unaffected and data points in high density areas will be combined and averaged.

    Parameters
    ----------
    data : array
        2D array of size (N, D)
    n : int, optional
        Choose the distance threshold by the n-th largest k-NN distance; must be smaller than `N x k`.
    threshold : float or None, optional
        The distance threshold for combining nearest neighbor pairs. Override `N` when given.
    k : int, optional
        Number of nearesest neigbors for KNN graph. Choose larger k for better approximation,
        smaller for better speed. Use at least 2.

    Returns
    ----------
    tuple(list, list)
        * list of level 0, level 1, level 2, ... data points. As each level a data point corresponds to
            2^0, 2^1, 2^2, ... original data points.
        * list of input data indices corresponding to level-0,1,2,... data points.

    """

    data = np.asarray(data)
    c, _ = neighbors.neighbors(data, n_neighbors=np.minimum(k, data.shape[0]), metric='euclidean')

    c = c.nonzero()
    d = np.linalg.norm(data[c[0], :] - data[c[1], :], axis=1)
    isorted = np.argsort(d)

    _, iisorted = np.unique(c[0][isorted], return_index=True)

    source_i = c[0][isorted[iisorted]]
    target_i = c[1][isorted[iisorted]]
    source_ik = c[0][isorted]
    target_ik = c[1][isorted]
    dk = d[isorted]
    d = d[isorted[iisorted]]
    assert len(source_i) == data.shape[0]
    if threshold == None:
        threshold = -np.sort(-d)[n]

    solo_index = source_i[d > threshold]
    source_ik = source_ik[dk <= threshold]
    target_ik = target_ik[dk <= threshold]
    dk = dk[dk <= threshold]

    i = (~np.isin(source_ik, solo_index)) * (~np.isin(target_ik, solo_index))
    source_ik = source_ik[i]
    target_ik = target_ik[i]
    dk = dk[i]

    pairs_to_combine = np.vstack([source_ik, target_ik])
    pairs_to_combine_done = []
    while pairs_to_combine.shape[1] > 0:
        pairs_to_combine_flat = pairs_to_combine.T.flatten()
        _, i = np.unique(pairs_to_combine_flat, return_index=True)
        temp = pairs_to_combine_flat[i].copy()
        pairs_to_combine_flat[:] = -1
        pairs_to_combine_flat[i] = temp
        pairs_to_combine_working = np.reshape(pairs_to_combine_flat, (int(len(pairs_to_combine_flat) / 2), 2))
        pairs_to_combine_working = pairs_to_combine_working[~np.any(pairs_to_combine_working == -1, axis=1), :]
        pairs_to_combine_done.append(pairs_to_combine_working)
        pairs_to_combine_working = pairs_to_combine_working.flatten()
        pairs_to_combine = pairs_to_combine[:, (~np.isin(pairs_to_combine[0, :], pairs_to_combine_working)) * (
            ~np.isin(pairs_to_combine[1, :], pairs_to_combine_working))]

    while True:
        retry_index = np.setdiff1d(source_ik, np.vstack(pairs_to_combine_done).flatten())
        if len(retry_index) == 0:
            break
        c_r, _ = neighbors.neighbors(data[retry_index, :], n_neighbors=np.minimum(k, len(retry_index)),
                                     metric='euclidean')
        c_r = c_r.nonzero()
        d_r = np.linalg.norm(data[retry_index, :][c_r[0], :] - data[retry_index, :][c_r[1], :], axis=1)

        isorted = np.argsort(d_r)
        source_ik_r = c_r[0][isorted]
        target_ik_r = c_r[1][isorted]
        dk_r = d_r[isorted]

        source_ik_r = source_ik_r[dk_r < threshold]
        target_ik_r = target_ik_r[dk_r < threshold]
        dk_r = dk_r[dk_r < threshold]
        if len(source_ik_r) == 0:
            break
        pairs_to_combine = np.vstack([retry_index[source_ik_r], retry_index[target_ik_r]])
        while pairs_to_combine.shape[1] > 0:
            pairs_to_combine_flat = pairs_to_combine.T.flatten()
            _, i = np.unique(pairs_to_combine_flat, return_index=True)
            temp = pairs_to_combine_flat[i].copy()
            pairs_to_combine_flat[:] = -1
            pairs_to_combine_flat[i] = temp
            pairs_to_combine_working = np.reshape(pairs_to_combine_flat, (int(len(pairs_to_combine_flat) / 2), 2))
            pairs_to_combine_working = pairs_to_combine_working[~np.any(pairs_to_combine_working == -1, axis=1), :]
            pairs_to_combine_done.append(pairs_to_combine_working)
            pairs_to_combine_working = pairs_to_combine_working.flatten()
            pairs_to_combine = pairs_to_combine[:, (~np.isin(pairs_to_combine[0, :], pairs_to_combine_working)) * (
                ~np.isin(pairs_to_combine[1, :], pairs_to_combine_working))]

    if len(pairs_to_combine_done) == 0:
        solo_data = data
        return [solo_data]
    else:
        pairs_to_combine_done = np.vstack(pairs_to_combine_done)

        solo_index = np.concatenate([solo_index, np.setdiff1d(source_ik, np.vstack(pairs_to_combine_done).flatten())])
        solo_data = data[solo_index, :]
        pair_data = (data[pairs_to_combine_done[:, 0], :] + data[pairs_to_combine_done[:, 1], :]) / 2.0
        if len(pair_data) < 50:
            collapsed_index = [[solo_index], [pairs_to_combine_done[:, 0], pairs_to_combine_done[:, 1]]]
            return [solo_data, pair_data], collapsed_index
        else:
            coarse_data, coarse_index = multilevel_compression(pair_data, 0, threshold)
            collapsed_index = [[solo_index]] + [
                [pairs_to_combine_done[i, 0] for i in ind] + [pairs_to_combine_done[i, 1] for i in ind] for ind in
                coarse_index]
            return [solo_data] + coarse_data, collapsed_index


class Scms(object):
    """
    An implementation of nonparametric ridge estimation.
    The main algorithm implemented is subspace constrained mean-shift (SCMS).

    Parameters
    ----------
    data : array or list
        * if `data` is a 2D array of size (N, D). Regular KDE is used.
        * if `data` is a list. `data` is interpreted as compressed data as given by
        `ridge_estimation.multilevel_compression` function.
        Multi-level KDE is used. Different levels one data point corresponds to 1, 2, 4, 8, 16, ...
        regular datapoints.

    bw : float
        Gaussian kernel bandwidth for KDE.
    min_radius: int
        Set data-point specific minimum bandwidth to be the distance to its `min_radius`-th nearest neighbor.

    Attributes
    ----------
    bw : float
    adaptive_bw : array
        Kernel bandwidth specified for each data point. This is the bandwidth directly used in the algorithm.
    params : dict
        Save parameters used in running Scms.scms(...) for bootstrap.

    """

    def __init__(self, data, bw, min_radius=10):
        self.data = data
        self.bw = bw
        self.min_radius = min_radius
        self.params = {}
        if not isinstance(data, list):
            self._set_adaptive_bandwidth(self.min_radius)
            self.multilevel = False
        else:
            self.multilevel = True

        # self.p = np.exp(self._kernel_density_estimate(data, output_onlylogp=True))
        # self.maxp = np.max(self.p)

    def inverse_density_sampling(self, X, n_samples=2000, n_jobs=1, batch_size=16):
        """
        Extract a subset of data points to represent the whole data. Data points
        from high density areas are most likely to be subsampled and those from
        low density areas are preserved.  This is efficiently implemented with a
        trick of sorting the data by log density + Gumbel noise.

        Parameters
        ----------
        X : 2D array
            Input data points
        n_samples : int, optional
            Number of sub-sampled data points. Default is `2000`.
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is `1`.
        batch_size : int, optional
            Batch size. Set this to a number number to reduce memory usage. Default is `16`

        Returns
        -------
        indices : 1D array
            Subsample indices corresponding to input `X`.

        """
        if n_jobs == 1:
            logp = [self._density_estimate(pos, output_onlylogp=True)
                 for pos in np.array_split(X, np.ceil(X.shape[0] / batch_size))]
        else:
            with Pool(n_jobs) as pool:
                logp = pool.map(
                    lambda pos: self._density_estimate(pos, output_onlylogp=True),
                    np.array_split(X, np.ceil(X.shape[0] / (batch_size * n_jobs))))
        logp = np.concatenate(logp)
        s = -logp + np.random.gumbel(size=logp.shape)
        return np.argsort(-s)[:n_samples]

    def reset_bw(self, bw=None, min_radius=None):
        """
        Reset bandwidth.

        Parameters
        ----------
        bw : float
            Gaussian kernel bandwidth for KDE.
        min_radius: int
            Set data-point specific minimum bandwidth to be the distance to its `min_radius`-th nearest neighbor.


        Returns
        -------
        None
        """
        if bw:
            self.bw = bw
        if min_radius:
            self.min_radius = min_radius

        if self.min_radius > 0:
            self._set_adaptive_bandwidth(self.min_radius)
        else:
            self.adaptive_bw = np.ones(self.data.shape[0]) * self.bw

    def _set_adaptive_bandwidth(self, K):
        if K > 0:
            self.nbrs = NearestNeighbors(n_neighbors=K + 1).fit(self.data)
            self.adaptive_bw = np.maximum(self.nbrs.kneighbors(self.data)[0][:, -1], self.bw)
        else:
            self.adaptive_bw = np.ones(self.data.shape[0]) * self.bw

    def _density_estimate(self, X, output_onlylogp=False):
        """
        Estimate density and relevant quantities for data points specified by X.

        Parameters
        ----------
        X : array
            2D array. Input to density estimation.
        output_onlylogp : bool
            If true, returns logp, else returns p, g, h, msu.


        Returns
        -------
        p : array
            1D array.  Unnormalized probability density. The probability density
            is not normalized due to numerical stability. Exact log probability
            density is returned if `output_onlylogp=True`.
        g : array, optional
            2D array. Gradient of the probability density function.
        h : array, optional
            3D array. Hessian of the probability density function.
        msu :
            2D array. Meanshift update based on the probability density function.

        """
        if self.multilevel:
            if output_onlylogp:
                p = self._multilevel_kernel_density_estimate(X, output_onlylogp)
            else:
                p, g, h, msu = self._multilevel_kernel_density_estimate(X, output_onlylogp)
        else:
            if output_onlylogp:
                p = self._kernel_density_estimate(X, output_onlylogp)
            else:
                p, g, h, msu = self._kernel_density_estimate(X, output_onlylogp)

        if output_onlylogp:
            return p
        else:
            return p, g, h, msu

    def _kernel_density_estimate(self, X, output_onlylogp=False, ):
        """
        Estimate density and relevant quantities for data points specified by X
        with kernel density estimation.

        Parameters
        ----------
        X : array
            2D array including multiple data points. Input to density estimation.
        output_onlylogp : bool
            If true, returns logp, else returns p, g, h, msu.

        Returns
        -------
        p : array
            1D array.  Unnormalized probability density. The probability density
            is not normalized due to numerical stability. Exact log probability
            density is returned if `output_onlylogp=True`.
        g : array, optional
            2D array. Gradient of the probability density function.
        h : array, optional
            3D array. Hessian of the probability density function.
        msu :
            2D array. Meanshift update based on the probability density function.
        """

        # the number of data points and the dimensionality
        n, d = self.data.shape

        # and evaluate the kernel at each distance
        D = cdist(self.data, X)

        # prevent numerical overflow due to large exponentials
        logc = -d * np.log(np.min(self.adaptive_bw)) - d / 2 * np.log(2 * np.pi)
        C = (self.adaptive_bw[:, np.newaxis] / np.min(self.adaptive_bw)) ** (-d) * \
            np.exp(-1 / 2. * (D / self.adaptive_bw[:, np.newaxis]) ** 2)

        if output_onlylogp:
            # return the kernel density estimate
            return np.log(np.mean(C, axis=0).T) + logc
        else:
            # gradient of p
            u = vdist(self.data, X)
            g = np.mean((C / (self.adaptive_bw[:, np.newaxis] ** 2)).T[:, :, np.newaxis] * u, axis=1)

            # hessian of p
            Z = np.eye(d)
            h = matrix_multiply(
                (C / (n * self.adaptive_bw[:, np.newaxis] ** 4)).T[:, np.newaxis, :] * u.transpose((0, 2, 1)), u) - \
                as_strided(Z, strides=(0, Z.strides[0], Z.strides[1]), shape=(C.shape[1], Z.shape[0], Z.shape[1])) * \
                (np.sum(C / (n * self.adaptive_bw[:, np.newaxis] ** 2), axis=0)[:, np.newaxis, np.newaxis])
            # for computation of msu = Cp * g
            Cp = 1. / np.mean(C / (self.adaptive_bw[:, np.newaxis] ** 2), axis=0)
            # return the kernel density estimate
            return np.mean(C, axis=0).T, g, h, Cp[:, np.newaxis] * g

    def _multilevel_kernel_density_estimate(self, X, output_onlylogp=False, normalized=True):
        """
        Estimate density and relevant quantities for data points specified by X
        with kernel density estimation and multi-level data representation.
        Data point-specific bandwidth (self.adaptive_bw) is not supported and self.bw
        is used instead.


        Parameters
        ----------
        X : array
            2D array. Input to density estimation.
        output_onlylogp : bool
            If true, returns logp, else returns p, g, h, msu.

        Returns
        -------
        p : array
            1D array.  Unnormalized probability density. The probability density
            is not normalized due to numerical stability. Exact log probability
            density is returned if `output_onlylogp=True`.
        g : array, optional
            2D array. Gradient of the probability density function.
        h : array, optional
            3D array. Hessian of the probability density function.
        msu :
            2D array. Meanshift update based on the probability density function.
        """

        # the number of data points and the dimensionality
        npgh = []
        for data in self.data:
            n, d = data.shape
            # and evaluate the kernel at each distance
            D = cdist(data, X)
            if normalized:
                # prevent numerical overflow due to large exponentials
                logc = -d * np.log(self.bw) - d / 2 * np.log(2 * np.pi)
            else:
                warnings.warn("Warning: use unnormalized probablity. This does not affect density ridge-related estimates.")
                logc = 0
            C = np.exp(-1 / 2. * (D / self.bw) ** 2)

            if output_onlylogp:
                # return the kernel density estimate
                npgh.append((n, np.log(np.mean(C, axis=0).T) + logc))
            else:
                # gradient of p
                u = vdist(data, X)
                g = np.mean((C / (self.bw ** 2)).T[:, :, np.newaxis] * u, axis=1)

                # hessian of p
                Z = np.eye(d)
                h = matrix_multiply((C / (n * self.bw ** 4)).T[:, np.newaxis, :] * u.transpose((0, 2, 1)), u) - \
                    as_strided(Z, strides=(0, Z.strides[0], Z.strides[1]), shape=(C.shape[1], Z.shape[0], Z.shape[1])) * \
                    (np.sum(C / (n * self.bw ** 2), axis=0)[:, np.newaxis, np.newaxis])
                Cp = 1. / np.mean(C / (self.bw ** 2), axis=0)
                npgh.append((n, np.mean(C, axis=0).T, g, h, Cp))

            if output_onlylogp:
                ns = np.asarray([n * 2 ** i for i, (n, _) in enumerate(npgh)])
                ws = ns / ns.sum()
                logps = [(logp + np.log(ws[i]))[np.newaxis, :] for i, (n, logp) in enumerate(npgh)]
                logps = np.vstack(logps)
                logps = logsumexp(logps, axis=0)

                return logps

            else:
                ns = np.asarray([n * 2 ** i for i, (n, _, _, _, _) in enumerate(npgh)])
                ws = ns / ns.sum()

                ps, gs, hs, cps = npgh[0][1:]
                ps = ps * ws[0]
                gs = gs * ws[0]
                hs = hs * ws[0]
                cps = cps * ws[0]
                for i, (n, p, g, h, cp) in enumerate(npgh[1:]):
                    ps = ps + p * ws[i]
                    gs = gs + g * ws[i]
                    hs = hs + h * ws[i]
                    cps = cps + cps * ws[i]

                return ps, gs, hs, cps[:, np.newaxis] * gs

    def _nlocal_inv_cov(self, X):
        """
        Computes the hessian of log density function.
        This function also returns other density estimation related
        quantities.

        Parameters
        ----------
        X : array
            2D array. Input data points.

        Returns
        -------
        negative_local_inverse_covariance : array
            3D array. Hessian of log probability density function (negative local
            inverse covariance).
        p : array
            1D array.  Unnormalized probability density. The probability density
            is not normalized due to numerical stability. Exact log probability
            density is returned if `output_onlylogp=True`.
        g : array
            2D array. Gradient of the probability density function.
        h : array
            3D array. Hessian of the probability density function.
        msu :
            2D array. Meanshift update based on the probability density function.
        """

        p, g, h, msu = self._density_estimate(X)
        return 1. / p[:, np.newaxis, np.newaxis] * h - 1. / p[:, np.newaxis, np.newaxis] ** 2 * matrix_multiply(
            g[:, :, np.newaxis], g[:, :, np.newaxis].transpose((0, 2, 1))), p, g, h, msu

    def scms_update(self, X, method='LocInv', stepsize=0.01, ridge_dimensionality=1, relaxation=0):
        """
        Compute the constrained mean shift update of the point at x.

        Parameters
        ----------
        X : array
            2D array. Input data points.
        method : str {'Gradient'|'GradientLogp'|'SuRF'|'LocInv'}, optional
            * Gradient
                Projected gradient update based on Hessian of probability density function p.
                Update is specified by projected gradient of p * step size.
            * GradientLogP
                Projected gradient update based on Hessian of log probability density function log(p).
                Update is specified by projected gradient of log(p) * step size.
            * SuRF
                Projected mean-shift update based on Hessian of probability density function p.
                Update is specified by projected mean shift update * step size.
            * LocInv
                Projected mean-shift update based on Hessian of probability density function log p.
                Update is specified by projected mean shift update * step size.
            Default is 'LocInv'.
        stepsize : float, optional
            Step size.
        ridge_dimensionality : int, optional
            Ridge dimensionality.
        relaxation : float [0,1], optional
            Relax the ridge dimensionality based on eigen gap corresponding to the ridge_dimensionality.
            Typically used with `ridge_dimensionality=1`. All point are projected to one-dimensional ridges
            (trajectories) when `relaxation=0`. All points are projected to two-dimensional ridges
            (surfaces) when `relaxation=1`.

        Returns
        ----------
        update : 2D array
            SCMS update.
        relax_bool : 1D array
            Indicate whether the ridge dimensionality of a particular point was increased by 1 relative to
            what `ridge_dimensionality` specified. The amount of relaxation is controlled by `relaxation` argument.

        """

        if method == 'GradientP':
            _, g, h, msu = self._density_estimate(X)
            eigvals, eigvecs = np.linalg.eigh(-h)
            eigvals = -eigvals
            update = g * stepsize

        elif method == 'GradientLogP':
            h, p, g, _, msu = self._nlocal_inv_cov(X)
            g = g / p[:, np.newaxis]

            eigvals, eigvecs = np.linalg.eigh(-h)
            eigvals = -eigvals
            update = g * stepsize
        else:
            if method == 'MSP':
                # h equals hessian of p(x)
                p, g, h, msu = self._density_estimate(X)
                eigvals, eigvecs = np.linalg.eigh(-h)
                eigvals = -eigvals
            elif method == 'LocInv' or method == 'MSLogP':
                # h equals negative hessian of log(p(x)), or "local inverse covariance"
                h, p, g, _, msu = self._nlocal_inv_cov(X)
                eigvals, eigvecs = np.linalg.eigh(-h)
                eigvals = -eigvals
            else:
                raise ValueError('Method has to be one of Gradient, GradientLogp, MSP, MSLogP, or LocInv.')

            update = msu
            update = update * stepsize

        if ridge_dimensionality == 0:
            relax_bool = np.zeros((X.shape[0],), dtype=bool)
            return update, relax_bool
        else:
            eigvecs[:, :, :ridge_dimensionality] = 0
            # sort the eigenvectors by decreasing eigenvalue
            if relaxation > 0 and ridge_dimensionality >= 1:
                relax_bool = (eigvals[:, ridge_dimensionality - 1] - eigvals[:, ridge_dimensionality]) / (
                        eigvals[:, ridge_dimensionality - 1] - eigvals[:, -1]) < relaxation
            else:
                relax_bool = np.zeros((X.shape[0],), dtype=bool)
            eigvecs[relax_bool, :, ridge_dimensionality] = 0


            return matrix_multiply(matrix_multiply(eigvecs, eigvecs.transpose((0, 2, 1))), update[:, :, np.newaxis])[:,
                   :, 0], relax_bool

    def scms(self, X, n_iterations=50, threshold=0, tol=1e-7, method='LocInv', stepsize=0.5, ridge_dimensionality=1,
             relaxation=0, n_jobs=1, batch_size=16):
        """
        Performs subspace constrained mean shift on the entire data set using
        a Gaussian kernel with bandwidth of bw.

        Parameters
        ----------
        X : array
            2D array. Input data points.
        n_iterations : int, optional
            The number of iterations for running the algorithm
        threshold : float, optional
            Filter out data points below a probability density threshold.
            The threshold value is specified relative to the maximum density.
        tol : float, optional
            Tolerance for assessing convergence, stop if `max(abs(new_pos - pos)) < tol`.
        method : str {'GradientP'|'GradientLogP'|'MSP'|'MSLogP'|'InvCov'}, optional
            * GradientP
                Projected gradient update based on Hessian of probability density function p.
                Update is specified by projected gradient of p * step size.
            * GradientLogP
                Projected gradient update based on Hessian of log probability density function log(p).
                Update is specified by projected gradient of log(p) * step size.
            * MSP
                Projected mean-shift update based on Hessian of probability density function p.
                Update is specified by projected mean shift update * step size.
            * MSLogP or InvCov (equivalent)
                Projected mean-shift update based on Hessian of probability density function log p.
                Update is specified by projected mean shift update * step size.
            Default is 'InvCov'.
        stepsize : float, optional
            Step size. Default is 0.5
        ridge_dimensionality : int, optional
            Ridge dimensionality. Default is 1.
        relaxation : float [0,1], optional
            Relax the ridge dimensionality based on eigen gap corresponding to the ridge_dimensionality.
            Typically used with `ridge_dimensionality=1`. All point are projected to one-dimensional ridges
            (trajectories) when `relaxation=0`. All points are projected to two-dimensional ridges
            (surfaces) when `relaxation=1`. Default is 0.
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is 1.
        batch_size : int, optional
            Number of data points to process in a batch. Reduce this number can decrease memory usage.
            Default is 500.

        Returns
        ----------
        pos : array
            Updated data points postions.
        ifilter : array
            Indices of returned positions (useful when `threshold > 0` so part of the input are filtered).
        """

        self.params = {'X': X, 'n_iterations': n_iterations, 'threshold': threshold, 'tol': tol, 'method': method,
                       'stepsize': stepsize, 'ridge_dimensionality': ridge_dimensionality,
                       'relaxation': relaxation}

        self.X = X.copy()
        data_copy = self.data.copy()

        if threshold > 0:
            p = np.exp(self._kernel_density_estimate(X, output_onlylogp=True))
            maxp = np.max(np.exp(self._kernel_density_estimate(self.data, output_onlylogp=True)))
            ifilter = np.where(p >= (maxp * threshold))[0]
        else:
            ifilter = np.arange(X.shape[0])

        pos = X[ifilter, :]
        if n_jobs > 1:
            with Pool(n_jobs) as pool:
                for j in range(n_iterations):
                    pos_old = pos.copy()
                    if X is None:
                        self.data = pos.copy()

                    updates = pool.map(
                        lambda pos: self.scms_update(pos, method=method, stepsize=stepsize,
                                                     ridge_dimensionality=ridge_dimensionality, relaxation=relaxation),
                        np.array_split(pos, np.ceil(pos.shape[0] / (batch_size * n_jobs))))

                    if len(updates) == 1:
                        update = updates[0][0]
                        self.relax_bool = updates[0][1]
                    else:
                        update = np.vstack([i[0] for i in updates])
                        self.relax_bool = np.concatenate([i[1] for i in updates], axis=0)
                    pos = pos + update
                    if np.max(np.sum(np.abs(pos_old - pos), axis=1)) < tol:
                        break
        else:
            for j in range(n_iterations):
                pos_old = pos.copy()
                if X is None:
                    self.data = pos.copy()

                updates = list(map(
                    lambda pos: self.scms_update(pos, method=method, stepsize=stepsize,
                                                 ridge_dimensionality=ridge_dimensionality, relaxation=relaxation),
                    np.array_split(pos, np.ceil(pos.shape[0] / batch_size ))))

                if len(updates)==1:
                    update = updates[0][0]
                    self.relax_bool = updates[0][1]
                else:
                    update = np.vstack([i[0] for i in updates])
                    self.relax_bool = np.concatenate([i[1] for i in updates],axis=0)
                pos = pos + update
                if np.max(np.sum(np.abs(pos_old - pos), axis=1)) < tol:
                    break

        self.data = data_copy
        self.ridge = pos
        self.ifilter = ifilter
        return pos, ifilter


    def boostrap_trajectory(self, n_bootstrap=25, n_jobs=1, copy_bw=True):
        """
        Construct confidence set of density ridge positions through bootstrap. For each bootstrap sample,
        distance to its nearest neighbors in the original ridge estimate is recorded. The bootstrap runs
        will copy the parameters of the previous `scms` call, therefore `scms` has to be executed
        before this function.

        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap runs. Default is 25.
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is 1. Only specify `n_jobs>1` if `n_jobs=1` is used
            when running `scms`.
        copy_bw : bool, optional
            If true, the point-specific kernel bandwidths are unchanged after resampling.
            This option only has an effect is `self.min_radius > 0`.

        Returns
        ----------
        boostrap_n : array, shape (n_samples, n_bootstrap)
            2D array. Bootstrap nearest neighbor distances for each ridge position.
        boostrap_s : array, shape (n_samples, n_bootstrap)
            2D array. Bootstrap point-wise distances for each ridge position. Experimental.
        boostrap_ifilters : array, shape (n_samples, n_bootstrap)
            2D array. Bootstrap resample indices.
        """
        if self.params == {}:
            raise ValueError('Run scms first before bootstrap!')

        if self.min_radius > 0:
            warnings.warn('Boostrap with adaptive bandwidth (min_radius>0) preserves the '
                          'adaptive bandwidths and confidence sets contructed should be '
                          'interpreted as such. Use min_radius=0 to be consistent with '
                          'Chen et al. 2015 algorithm.')

        self.bootstrap_distances_n = []
        self.bootstrap_distances_s = []
        self.bootstrap_ifilters = []
        self.bootstrap_ridges = []

        def boostrap(i):
            np.random.seed(i)
            bdata, bi = bootstrap_resample(self.data)
            b = Scms(bdata, self.bw, min_radius=self.min_radius)
            if copy_bw:
                b.adaptive_bw = self.adaptive_bw[bi]

            _ridge, _ifilter = b.scms(**self.params)
            binaryfilter = np.zeros(self.data.shape[0], dtype=bool)
            binaryfilter[_ifilter] = True
            a = np.zeros(self.data.shape)
            a.fill(np.nan)
            b = np.zeros(self.data.shape)
            b.fill(np.nan)

            a[_ifilter, :] = _ridge
            b[self.ifilter, :] = self.ridge
            dists = linalg.norm(a - b, axis=1)[self.ifilter]
            return NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(_ridge).kneighbors(self.ridge)[0], \
                   dists, binaryfilter, _ridge

        with Pool(n_jobs) as p:
            res = p.map(boostrap, range(n_bootstrap))
            for dist_n, dist_s, ifilter, ridge in res:
                self.bootstrap_distances_n.append(dist_n)
                self.bootstrap_distances_s.append(dist_s)
                self.bootstrap_ifilters.append(ifilter)
                self.bootstrap_ridges.append(ridge)


        self.bootstrap_distances_n = np.hstack(self.bootstrap_distances_n)
        self.bootstrap_distances_s = np.vstack(self.bootstrap_distances_s).T
        self.bootstrap_ifilters = np.hstack(self.bootstrap_ifilters)
        return self.bootstrap_distances_n, self.bootstrap_distances_s, self.bootstrap_ifilters
        # return scoreatpercentile(self.bootstrap_distances_n, percentile_confidence, axis=1),
        #       scoreatpercentile(self.bootstrap_distances_s, percentile_confidence, axis=1)

    def bootstrap_landmark(self, X, n_bootstrap=20, method='LocInv', smooth_bootstrap=False, copy_bw=True, n_jobs=4):
        """
        Experimental function. Work in progress.
        """
        self.landmarks = X
        if method == 'LocInv' or method == 'GradientLogP':
            self.landmarks_h, self.landmarks_p, self.landmarks_g, _ = self._nlocal_inv_cov(self.landmarks)
        else:
            self.landmarks_p, self.landmarks_g, self.landmarks_h = self._kernel_density_estimate(self.landmarks)

        # use megative of h because eigh sort eigen values in ascending order
        eigvals, eigvecs = np.linalg.eigh(-self.landmarks_h)
        eigvals = -eigvals
        self.landmarks_h_eigvecs = eigvecs
        self.landmarks_h_eigvals = eigvals

        self.landmarks_bootstrap_p = []
        self.landmarks_bootstrap_g = []
        self.landmarks_bootstrap_h = []
        self.landmarks_bootstrap_g_proj = []
        self.landmarks_bootstrap_h_proj = []
        self.landmarks_bootstrap_h_dir = []

        def boostrap(i):
            np.random.seed(i)
            if smooth_bootstrap:
                b = Scms(self.data + np.random.randn(*self.data.shape) * self.adaptive_bw, self.bw,
                         min_radius=self.min_radius)
                if copy_bw:
                    b.adaptive_bw = self.adaptive_bw
            else:
                bdata, bi = bootstrap_resample(self.data)
                b = Scms(bdata, self.bw, min_radius=self.min_radius)
                if copy_bw:
                    b.adaptive_bw = self.adaptive_bw[bi]

            if method == 'LocInv' or method == 'GradientLogP':
                bh, bp, bg, _ = b._nlocal_inv_cov(self.landmarks)
            else:
                bp, bg, bh = b._kernel_density_estimate(self.landmarks)

            gproj = np.sum(self.landmarks_g * bg, axis=1) / np.linalg.norm(self.landmarks_g, axis=1)
            hproj = np.sum(
                matrix_multiply(bh.transpose((0, 2, 1)), self.landmarks_h_eigvecs) * self.landmarks_h_eigvecs, axis=1)
            return bp, bg, bh, gproj, hproj

        with Pool(n_jobs) as p:
            res = p.map(boostrap, range(n_bootstrap))
            for bp, bg, bh, gproj, hproj in res:
                self.landmarks_bootstrap_p.append(bp)
                self.landmarks_bootstrap_g.append(bg)
                self.landmarks_bootstrap_h.append(bh)
                self.landmarks_bootstrap_g_proj.append(gproj)
                self.landmarks_bootstrap_h_proj.append(hproj)
                # _, eigvecs = np.linalg.eigh(-bh)
                # self.landmarks_bootstrap_h_dir.append(np.abs(np.sum(eigvecs * self.landmarks_h_eigvecs,axis=1)))

        self.landmarks_bootstrap_g_proj = np.vstack(self.landmarks_bootstrap_g_proj).T
        self.landmarks_bootstrap_h_proj = np.dstack(self.landmarks_bootstrap_h_proj)
        # self.landmarks_bootstrap_h_dir = np.dstack(self.landmarks_bootstrap_h_dir)

        return self.landmarks_bootstrap_g_proj, self.landmarks_bootstrap_h_proj
