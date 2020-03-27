# -*- coding: utf-8 -*-
"""
This module implements the quaslinear representation method GraphDR and relevant
functions. The method is implemented in the graphdr function. A scikit-learn
Estimator-like interface to graphdr is also provided through the GraphDR class.
"""
import os

from sklearn.neighbors import kneighbors_graph
from sklearn.exceptions import NotFittedError
from scipy.sparse import csgraph, csr_matrix
import scipy
import numpy as np

from scipy.sparse import issparse
from sklearn.decomposition import PCA
from multiprocess import Pool
from sklearn.base import BaseEstimator

from .external import neighbors


def knn_graph(X, n_neighbors=15, space='l2', space_params=None, method='hnsw', num_threads=8, params={'post': 2},
              verbose=False):
    """Construct an approximate K-nearest neighbors graph with nmslib.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        A input 2D array to construct KNN graph on. Each row represent a sample.
    n_neighbors : int, optional
        Default is `15`. The number of nearest neighbors per sample.
    space : string, optional
        Default is `l2`. The metric/non-metric distance functions to use to compute distances.
        Note that l2 corresponds to Euclidean distance.
        see https://github.com/nmslib/nmslib/blob/master/manual/manual.pdf
            * bit_hamming
            * l1
            * l1_sparse
            * l2, euclidean
            * l2_sparse
            * linf
            * linf_sparse
            * lp:p=...
            * lp_sparse:p=...
            * angulardist, angulardist sparse, angulardist sparse fast
            * jsmetrslow, jsmetrfast, jsmetrfastapprox
            * leven
            * sqfd minus func, sqfd heuristic func:alpha=..., sqfd gaussian func:alpha=...
            * jsdivslow, jsdivfast, jsdivfastapprox
            * cosinesimil, cosine, cosinesimil sparse
            * normleven
            * kldivfast
            * kldivgenslow, kldivgenfast, kldivgenfastrq
            * itakurasaitoslow, itakurasaitofast, itakurasaitofastrq
            * negdotprod_sparse
            * querynorm_negdotprod_sparse
            * renyi_diverg
            * ab_diverg
    space_params : dict or None, optional
        Parameters for configuring the space.
    method : str
        The nmslib method to use. Default is `hnsw`.
    num_threads : int, optional
        Default is `8`. Number of threads to run nmslib with.
    params : dict, optional
        Default is `{'post': 2}`. Input parameters to nmslib index construction.
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
    verbose: bool, optional
        Default is False. Print extra information.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix representing the constructed KNN graph.
    """

    import nmslib
    if space == 'euclidean':
        space = 'l2'
    elif space == 'manhattan':
        space = 'l1'
    elif space == 'cosine':
        space = 'cosinesimil'

    index = nmslib.init(method=method, space=space, space_params=space_params)
    index.addDataPointBatch(X)
    index.createIndex(params, print_progress=False)
    neighbors = index.knnQueryBatch(X, k=n_neighbors, num_threads=num_threads)

    ind = np.vstack([i for i, d in neighbors])
    sind = np.repeat(np.arange(ind.shape[0]), ind.shape[1])
    tind = ind.flatten()
    g = csr_matrix((np.ones(ind.shape[0] * ind.shape[1]), (sind, tind)), (ind.shape[0], ind.shape[0]))
    return g


def cg(A, B, x0=None, tol=1e-7, atol=1e-15, use_cuda=False):
    """
    Solve X in linear equation AX = B when A is sparse with conjugate gradient method.
    Support GPU with use_cuda=True.

    Parameters
    ----------
    A : sparse matrix or array
        A N-by-N matrix of the linear system.
    B : array
        Right hand side of the linear system.
    x0 : array or None, optional
        If specified, initialize the solution with x0.
    tol, atol : float, optional
        Tolerances for convergence `norm(residual) <= max(tol*norm(b), atol)`.
        Default is `tol=1e-7, atol=1e-15`.
    use_cuda : bool, optional
        Use GPU acceleration with CUDA. A GPU supporting CUDA is needed and CuPy
        package need to be installed.
        Default is `False`.

    Returns
    ----------
    array
        The solution to X.
    """
    if use_cuda:
        import cupy
        alg = cupy
        A = alg.sparse.csr_matrix(A)
        B = alg.array(B)
    else:
        alg = np

    atol = alg.maximum(alg.sqrt(alg.sum(B * B, axis=0)) * tol, atol)
    if not x0:
        X = alg.ones((A.shape[1], B.shape[1]))
    else:
        X = x0
    R = A.dot(X) - B
    P = - R
    r_k_norms = alg.sum(R * R, axis=0)
    while True:
        AP = A.dot(P)
        alphas = r_k_norms / alg.sum(P * AP, axis=0)
        X += alphas[alg.newaxis, :] * P
        R += alphas[alg.newaxis, :] * AP
        r_kplus1_norms = alg.sum(R * R, axis=0)
        betas = r_kplus1_norms / r_k_norms
        r_k_norms = r_kplus1_norms
        if alg.all(alg.sqrt(r_kplus1_norms) < atol):
            break
        P = betas[alg.newaxis, :] * P - R
    return X.get() if use_cuda else X



class GraphDR(BaseEstimator):
    """
    Quasilinear data representation that preserves interpretability of a
    linear space.

    This class provides a scikit-learn type interface that wraps around
    the graphdr function.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neigbors to compute per sample. Default is `10`.
    regularization : float, optional
        Regularization parameter determines the weight on the graph shrinkage term
        relative to the reconstruction error term. A smaller regularization parameter
        leads to output that are more similar to a linear representation. A larger
        regularization parameter provides more contraction based on the graph. Default is `100`.
    n_components : int or None, optional
        Number of dimensions in the output. Specify a n_components smaller than the input
        dimension when `no_rotation=True` can reduce the amount of computation.
        Default is `None`.
    no_rotation : bool, optional
        :math:`W` is fixed to identity matrix so that the output is not rotated
        relative to input. no_rotation can also speed up the computation for large
        input when `n_components` is also specified, because only the required top
        `n_components` are computed. If `no_rotation=True`, you may consider preprocessing
        the input with another linear dimensionality reduction method like PCA.
        Default is `False`.
    metric : str, optional
        distance metric for constructing nearest_neighbors graph. Note that here
        `metric` can be non-metric distance function. The allowed values depend on
        the `method` specified. As three nearest neighbors algorithms are supported,
        and these algorithms support different distance metrics. Default is `euclidean`.
        * 'small' :
            Use sklearn.neighbors.kneighbors_graph
            Supported values are
                * euclidean
                * manhattan
                * chebyshev
                * minkowski
                * wminkowski
                * seuclidean
                * mahalanobis
                * hamming
                * canberra
                * braycurtis
            See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        * 'big_nndescent' or 'big_nnumap' :
            umap.nndescent. Supported values are
                * euclidean
                * manhattan
                * chebyshev
                * minkowski
                * canberra
                * braycurtis
                * mahalanobis
                * wminkowski
                * seuclidean
                * cosine
                * correlation
                * haversine
                * hamming
                * jaccard
                * dice
                * russelrao
                * kulsinski
                * rogerstanimoto
                * sokalmichener
                * sokalsneath
                * yule
        * 'big_nmslib' :
        nmslib. See https://github.com/nmslib/nmslib/blob/master/manual/manual.pdf for details:
                * bit_hamming
                * l1
                * l1_sparse
                * l2, euclidean
                * l2_sparse
                * linf
                * linf_sparse
                * lp:p=...
                * lp_sparse:p=...
                * angulardist, angulardist sparse, angulardist sparse fast
                * jsmetrslow, jsmetrfast, jsmetrfastapprox
                * leven
                * sqfd minus func, sqfd heuristic func:alpha=..., sqfd gaussian func:alpha=...
                * jsdivslow, jsdivfast, jsdivfastapprox
                * cosinesimil, cosine, cosinesimil sparse
                * normleven
                * kldivfast
                * kldivgenslow, kldivgenfast, kldivgenfastrq
                * itakurasaitoslow, itakurasaitofast, itakurasaitofastrq
                * negdotprod_sparse
                * querynorm_negdotprod_sparse
                * renyi_diverg
                * ab_diverg
    metric_params: dict, optional
        Some distance metrics requires additional arguments which can be provided as
        a dictionary. Default is `{}`.
    method: str, optional {'auto', 'small', 'big_nmslib', 'big_umap', 'big_nndescent'}
        Method to use. Different algorithms strongly affect the computation speed.
        Optimal choices are dependent on the input size. Default is `auto`.
        * 'auto'
            Choose automatically based on input size. Recommended for most uses.
        * 'small'
            For input of small size, it is affordable to compute nearest-neigbors exactly
            and use matrix inversion in the algorithm.
        * 'big_nmslib'
            For input of large size. If NMSlib is installed, it is the fastest algorithm that
            is supported.
        * 'big_nndescent'
            For input of large size. The NN-Descent algorithm implemented in the the umap package.
            Do not require external libraries.
        * 'big_nnumap'
            For input of large size. Do not require external libraries.
    _lambda : float or None, optional
        Regularization parameter :math:`\\lambda`. If not specified :math:`\\lambda` is
        computed from `regularization`,
        specifying `regularization` instead of `_lambda` is generally recommended as it
        automatically adjusts based on graph density. Default is None.
    symmetrize : bool, optional
        Symmetrize the nearest-neighbors graph by averaging with its transpose. Default is `True`.
    rescale : bool, optional
        Rescale each dimension in the output to align with the input. It works with and without
        `no_rotation`. Default is `False`
    refine_iter : int, optional
        Number of refinement iterations. Default is `3`.
    refine_threshold : float, optional
        Specify the cutoff for refinement. Unit is IQR (distance between the 25% quantile and the 75% quantile)
        of overall "stress" distribution across all edges. Default is `12`.
    _refine_threshold : float or None, optional
        Used internally. Default is None.
    tol, atol : float, optional
        Tolerances for iterative linear solver `norm(residual) <= max(tol*norm(b), atol)`.
        Default is `tol=1e-7, atol=1e-15`.
    nmslib_params : dict, optional
        Parameters given to nmslib createIndex function.
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
        Default is `{'post': 2}`.
    n_jobs : int, optional
        Number of jobs to run in parallel. Specifying `n_jobs > 1` can speed up computation
        when method is one of `big_nmslib`, `big_nndescent`, or `big_nnumap`. Note that
        when `use_cuda=True` specifying `n_jobs` does not run the jobs in parallel but
        can reduce amount of GPU memory required by splitting the problem and run sequentially.
        Default is 1.
    n_jobs_nmslib : int, optional
        Number of threads used by nmslib. Default is `8`.
    use_cuda : bool, optional
        Use CUDA GPU acceleration. CuPy has to be installed (https://github.com/cupy/cupy)
        and a CUDA-enabled GPU is required. Default is `False`.
    verbose : bool, optional
        Print additional information. Default is `False`.

    Attributes
    ----------
    embedding_ : array, shape (n_samples, n_components)
    W_ : array, shape (n_features, n_components)
        Right linear operator in the transformation Z = KXW. W_ is None when `no_rotation=True`.
"""

    def __init__(
            self,
            n_neighbors=10, regularization=1000, n_components=None, no_rotation=False,
            metric='l2', metric_params={}, method='auto',
            _lambda=None, symmetrize=True, rescale=True,
            refine_iter=0, refine_threshold=12, _refine_threshold=None,
            tol=1e-7, atol=1e-15, nmslib_params={'post': 2},
            n_jobs=1, n_jobs_nmslib=8, use_cuda=False, verbose=False):
        """
        Initialize parameters
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.regularization = regularization
        self._lambda = _lambda
        self.metric = metric
        self.metric_params = metric_params
        self.method = method
        self.symmetrize = symmetrize
        self.rescale = rescale
        self.no_rotation = no_rotation
        self.refine_iter = refine_iter
        self.refine_threshold = refine_threshold
        self._refine_threshold = _refine_threshold
        self.nmslib_params = nmslib_params
        self.tol = tol
        self.atol = atol
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_jobs_nmslib = n_jobs_nmslib
        self.use_cuda = use_cuda

        self.W_ = None
        self.embedding_ = None
        self.fitted = False

    def _validate_parameters(self):
        # TODO
        pass

    def fit(self, X, y=None, custom_graph=None, refine_protected_graph=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features).
            X may be a sparse matrix. If self.skip_pca is true, X must be orthogonal.
        y : Ignored
        custom_graph : sparse matrix, shape (n_samples, n_samples), optional
            If provided, use it instead of the graph constructed from X.
        refine_protected_graph : sparse matrix

        Returns
        ----------
        self : object
            Returns the instance itself.
        """
        output = graphdr(X, n_neighbors=self.n_neighbors, regularization=self.regularization,
                         metric=self.metric, metric_params=self.metric_params,
                         n_components=self.n_components, no_rotation=self.no_rotation, _lambda=self._lambda,
                         custom_graph=custom_graph, rescale=self.rescale, symmetrize=self.symmetrize,
                         refine_iter=self.refine_iter, refine_threshold=self.refine_threshold,
                         _refine_threshold=self._refine_threshold, refine_protected_graph=refine_protected_graph,
                         tol=self.tol, atol=self.atol, nmslib_params=self.nmslib_params, verbose=self.verbose,
                         n_jobs=self.n_jobs, n_jobs_nmslib=self.n_jobs_nmslib, use_cuda=self.use_cuda,
                         return_all=True)

        if self.no_rotation:
            Z, _ = output
            self.W_ = None
        else:
            Z, W, _ = output
            self.W_ = W

        self.embedding_ = Z
        self.fitted = True
        return self

    def fit_transform(self, X, y=None, custom_graph=None, refine_protected_graph=None):
        """Fit the model from data in X and transform X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : Ignored

        Returns
        ---------
        Z : array, shape (n_samples, n_components)
            Embedding of the input data.
        """
        self.fit(X, y, custom_graph=custom_graph, refine_protected_graph=refine_protected_graph)
        return self.embedding_

    def transform(self, X, y=None, custom_graph=None, refine_protected_graph=None):
        """Apply fitted model to transform X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : Ignored

        Returns
        ---------
        Z : array, shape (n_samples, n_components)
            Embedding of the input data.
        """
        if not self.fitted:
            raise NotFittedError('This GraphDR object has not been fitted yet.')

        Z = graphdr(X, n_neighbors=self.n_neighbors, regularization=self.regularization,
                         metric=self.metric, metric_params=self.metric_params,
                         n_components=self.n_components, no_rotation=True, _lambda=self._lambda, 
                         custom_graph=custom_graph, rescale=self.rescale, symmetrize=self.symmetrize,
                         refine_iter=self.refine_iter, refine_threshold=self.refine_threshold,
                         _refine_threshold=self._refine_threshold, refine_protected_graph=refine_protected_graph,
                         tol=self.tol, atol=self.atol, nmslib_params=self.nmslib_params, verbose=self.verbose,
                         n_jobs=self.n_jobs, n_jobs_nmslib=self.n_jobs_nmslib, use_cuda=self.use_cuda,
                         return_all=False)
                         
        if self.W_ is not None:
            Z = Z @ self.W_
        return Z

def graphdr(X, n_neighbors=10, regularization=100, n_components=None, no_rotation=False,
            metric='euclidean', metric_params={}, method='auto',
            _lambda=None, init=None, symmetrize=True,
            custom_graph=None, rescale=False, refine_iter=0, refine_threshold=12,
            _refine_threshold=None, refine_protected_graph=None, tol=1e-7, atol=1e-15,
            nmslib_params={'post': 2}, n_jobs=1, n_jobs_nmslib=8, use_cuda=False, verbose=False,
            return_all=False):
    """
    Quasilinear data representation that preserves interpretability of a linear space.

    Parameters
    ----------
    X : array (n_samples, n_features)
        The input array.
    n_neighbors : int, optional
        Number of nearest neigbors to compute per sample. Default is `10`.
    regularization : float, optional
        Regularization parameter determines the weight on the graph shrinkage term
        relative to the reconstruction error term. A smaller regularization parameter
        leads to output that are more similar to a linear representation. A larger
        regularization parameter provides more contraction based on the graph. Default is `100`.
    n_components : int or None, optional
        Number of dimensions in the output. Specify a n_components smaller than the input
        dimension when `no_rotation=True` can reduce the amount of computation.
        Default is `None`.
    no_rotation : bool, optional
        :param refine_protected_graph:
        :math:`W` is fixed to identity matrix so that the output is not rotated
        relative to input. no_rotation can also speed up the computation for large
        input when `n_components` is also specified, because only the required top
        `n_components` are computed. If `no_rotation=True`, you may consider preprocessing
        the input with another linear dimensionality reduction method like PCA.
        Default is `False`.
    metric : str, optional
        distance metric for constructing nearest_neighbors graph. Note that here
        `metric` can be non-metric distance function. The allowed values depend on
        the `method` specified. As three nearest neighbors algorithms are supported,
        and these algorithms support different distance metrics. Default is `euclidean`.
        * 'small':
            Use sklearn.neighbors.kneighbors_graph
            Supported values are
                * euclidean
                * manhattan
                * chebyshev
                * minkowski
                * wminkowski
                * seuclidean
                * mahalanobis
                * hamming
                * canberra
                * braycurtis
            See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
        * 'big_nndescent' or 'big_nnumap':
            umap.nndescent. Supported values are
                * euclidean
                * manhattan
                * chebyshev
                * minkowski
                * canberra
                * braycurtis
                * mahalanobis
                * wminkowski
                * seuclidean
                * cosine
                * correlation
                * haversine
                * hamming
                * jaccard
                * dice
                * russelrao
                * kulsinski
                * rogerstanimoto
                * sokalmichener
                * sokalsneath
                * yule
        * 'big_nmslib'
        nmslib. See https://github.com/nmslib/nmslib/blob/master/manual/manual.pdf for details:
                * bit_hamming
                * l1
                * l1_sparse
                * l2, euclidean
                * l2_sparse
                * linf
                * linf_sparse
                * lp:p=...
                * lp_sparse:p=...
                * angulardist, angulardist sparse, angulardist sparse fast
                * jsmetrslow, jsmetrfast, jsmetrfastapprox
                * leven
                * sqfd minus func, sqfd heuristic func:alpha=..., sqfd gaussian func:alpha=...
                * jsdivslow, jsdivfast, jsdivfastapprox
                * cosinesimil, cosine, cosinesimil sparse
                * normleven
                * kldivfast
                * kldivgenslow, kldivgenfast, kldivgenfastrq
                * itakurasaitoslow, itakurasaitofast, itakurasaitofastrq
                * negdotprod_sparse
                * querynorm_negdotprod_sparse
                * renyi_diverg
                * ab_diverg
    metric_params: dict, optional
        Some distance metrics requires additional arguments which can be provided as
        a dictionary. Default is `{}`.
    method: str, optional {'auto', 'small', 'big_nmslib', 'big_umap', 'big_nndescent'}
        Method to use. Different algorithms strongly affect the computation speed.
        Optimal choices are dependent on the input size. Default is `auto`.
        * 'auto'
            Choose automatically based on input size. Recommended for most uses.
        * 'small'
            For input of small size, it is affordable to compute nearest-neigbors exactly
            and use matrix inversion in the algorithm.
        * 'big_nmslib'
            For input of large size. If NMSlib is installed, it is the fastest algorithm that
            is supported.
        * 'big_nndescent'
            For input of large size. The NN-Descent algorithm implemented in the the umap package.
            Do not require external libraries.
        * 'big_nnumap'
            For input of large size. Do not require external libraries.
    _lambda : float or None, optional
        Regularization parameter :math:`\\lambda`. If not specified :math:`\\lambda` is
        computed from `regularization`,
        specifying `regularization` instead of `_lambda` is generally recommended as it
        automatically adjusts based on graph density. Default is None.
    init : array or None, optional
        Initialize output representation Z with this array if solved through iterative solver.
        `init` is only used when `no_rotation=True` and method is not `small`.
        Default is None.
    symmetrize : bool, optional
        Symmetrize the nearest-neighbors graph by averaging with its transpose. Default is `True`.
    custom_graph : sparse matrix or None, optional
        If specified, use this user-provided graph instead of constructing nearest-neighbors
        graph from `X`. Default is None.
    rescale : bool, optional
        Rescale each dimension in the output to align with the input. It works with and without
        `no_rotation`. Default is `False`
    refine_iter : int, optional
        Number of refinement iterations. Default is `3`.
    refine_threshold : float, optional
        Specify the cutoff for refinement. Unit is IQR (distance between the 25% quantile and the 75% quantile)
        of overall "stress" distribution across all edges. Default is `12`.
    _refine_threshold : float or None, optional
        Used internally. Default is None.
    refine_protected_graph : sparse matrix or None, optional
        Sparse matrix of size (n_samples, n_samples).
        Specify edges (any non-zero edges in refine_protected_graph) that will
        not be removed during refinement. Useful for graph construction for complex experimental designs
        like batch design so that edges connecting batches will not be removed. Default is `None`.
    tol, atol : float, optional
        Tolerances for iterative linear solver `norm(residual) <= max(tol*norm(b), atol)`.
        Default is `tol=1e-7, atol=1e-15`.
    nmslib_params : dict, optional
        Parameters given to nmslib createIndex function.
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
        Default is `{'post': 2}`.
    n_jobs : int, optional
        Number of jobs to run in parallel. Specifying `n_jobs > 1` can speed up computation
        when method is one of `big_nmslib`, `big_nndescent`, or `big_nnumap`. Note that
        when `use_cuda=True` specifying `n_jobs` does not run the jobs in parallel but
        can reduce amount of GPU memory required by splitting the problem and run sequentially.
        Default is 1.
    n_jobs_nmslib : int, optional
        Number of threads used by nmslib. Default is `8`.
    use_cuda : bool, optional
        Use CUDA GPU acceleration. CuPy has to be installed (https://github.com/cupy/cupy)
        and a CUDA-enabled GPU is required. Default is `False`.
    verbose : bool, optional
        Print additional information. Default is `False`.
    return_all : bool, optional
        Return a tuple of (Z, W, graph). Default is `False`.

    Returns
    -------
    Z: array, (optional) W: array, (optional) Graph: sparse matrix
    * Z : Embedding of the input data.
    * W : Linear operator on the feature space, returned if `return_all==True`.
    * graph : The graph used for GraphDR embedding, returned if `return_all==True`.

    """
    if method == 'auto':
        if X.shape[0] < 10000:
            method = 'small'
        else:
            try:
                import nmslib
                method = 'big_nmslib'
            except ImportError:
                method = 'big_nndescent'

    X_mean = X.mean(axis=0)

    if method == 'small':
        if custom_graph is None:
            graph = kneighbors_graph(np.asarray(X), int(n_neighbors), metric=metric,
                                     metric_params=metric_params, include_self=False)
        else:
            graph = custom_graph

        if symmetrize:
            graph = 0.5 * (graph + graph.T)

        graphL = csgraph.laplacian(graph)
        if _lambda is None:
            _lambda = regularization / (graph.sum(axis=1).mean() / (X.shape[0] / 10000))
        G = scipy.sparse.eye(X.shape[0]) + _lambda * graphL
        if issparse(G):
            Ginv = np.linalg.inv(G.todense())
        else:
            Ginv = np.linalg.inv(G)

        if no_rotation:
            if n_components:
                try:
                    X = X[:, :n_components]
                    X_mean = X_mean[:n_components]
                except:
                    raise ValueError('n_components value is not valid.')
            Z = np.asarray(np.dot(X.T, Ginv).T)
        else:
            C = np.dot(np.dot(X.T, Ginv), X)
            _, W = np.linalg.eigh(C)
            W = np.array(W)
            W = W[:, ::-1]
            if n_components:
                try:
                    W = W[:, :n_components]
                except:
                    raise ValueError('n_components value is not valid.')
            Z = np.asarray(np.dot(np.dot(W.T, X.T), Ginv).T)

        graph = scipy.sparse.csr_matrix(graph)
        graph.eliminate_zeros()

    else:
        if custom_graph is None:
            if method == 'big_nmslib':
                graph = knn_graph(X, n_neighbors=int(n_neighbors), num_threads=n_jobs_nmslib, space=metric,
                                  params=nmslib_params, space_params=metric_params, verbose=verbose)
            elif method == 'big_nnumap':
                graph = \
                neighbors.neighbors(X, n_neighbors=int(n_neighbors), metric=metric, metric_kwds=metric_params)[0]
            elif method == 'big_nndescent':
                graph = \
                neighbors.neighbors(X, n_neighbors=int(n_neighbors), metric=metric, metric_kwds=metric_params,
                                             smoothknn=False)[0]
            else:
                raise ValueError('Unrecognized method {}' % method)
        else:
            graph = custom_graph

        if symmetrize:
            graph = 0.5 * (graph + graph.T)

        graphL = csgraph.laplacian(graph)

        if _lambda is None:
            _lambda = regularization / (graph.sum(axis=1).mean() / (X.shape[0] / 10000))

        G = scipy.sparse.eye(X.shape[0]) + _lambda * graphL

        def msolve(A, Y, init=None):
            if use_cuda:
                Z = np.asarray(np.hstack(list(
                    map(lambda y: cg(A, y, x0=None, tol=tol, atol=atol, use_cuda=True), np.split(Y, n_jobs, axis=1)))))
            else:
                if n_jobs <= 1:
                    Z = np.asarray(np.hstack([scipy.sparse.linalg.cg(A, Y[:, i],
                                                                     x0=init[:, i] if init is not None else None,
                                                                     tol=tol, atol=atol)[0][:, np.newaxis] for i in
                                              range(Y.shape[1])]))
                else:
                    p = Pool(n_jobs)
                    try:
                        if init is None:
                            Z = np.asarray(np.hstack(list(
                                p.map(lambda y: cg(A, y, x0=None, tol=tol, atol=atol, use_cuda=False),
                                      np.split(Y, Y.shape[1], axis=1)))))
                        else:
                            Z = np.asarray(np.hstack(list(
                                p.map(lambda y, init_: cg(A, y, x0=init_, tol=tol, atol=atol, use_cuda=False),
                                      zip(np.split(Y, Y.shape[1], axis=1), np.split(init, init.shape[1], axis=1))))))
                        p.close()
                        p.join()
                    except KeyboardInterrupt:
                        print("Caught KeyboardInterrupt, terminating workers")
                        p.terminate()
                        p.join()
                        raise
            return Z

        if no_rotation:
            if n_components:
                try:
                    X = X[:, :n_components]
                    X_mean = X_mean[:n_components]
                except:
                    raise ValueError('n_components value is not valid.')
            Z = msolve(G, X, init=init)
        else:
            GinvX = msolve(G, X)
            C = np.dot(X.T, GinvX)
            lambdas, W = np.linalg.eigh(C)
            W = np.array(W)
            W = W[:, ::-1]
            if n_components:
                try:
                    W = W[:, :n_components]
                except:
                    raise ValueError('n_components value is not valid.')
            Z = np.dot(GinvX, W)




    # refine the graph by deleting edges with high stress
    if refine_iter > 0:
        if refine_protected_graph is not None:
            frozen_bool = np.isin(graph.nonzero()[0] * graph.shape[0] + graph.nonzero()[1],
                                  refine_protected_graph.nonzero()[0] * refine_protected_graph.shape[0] + refine_protected_graph.nonzero()[1])
        distZ = np.sqrt(np.sum((Z[graph.nonzero()[0], :] - Z[graph.nonzero()[1], :]) ** 2, axis=1))
        stress = distZ
        if verbose:
            print("Refinement round %d start..." % (refine_iter - 1))
            print("Number of starting edges: %d" % np.sum(graph.data > 0))

        if _refine_threshold is None:
            q1, q3 = np.percentile(stress, [25, 75])
            iqr = q3 - q1
            _refine_threshold = q3 + (iqr * refine_threshold)

        if refine_protected_graph is not None:
            graph.data[frozen_bool] = graph.data[frozen_bool] * ((stress < _refine_threshold)[frozen_bool])
        else:
            graph.data = graph.data * (stress < _refine_threshold)
        graph.eliminate_zeros()
        if verbose:
            print("Number of refined edges: %d" % np.sum(graph.data > 0))

        n_components, labels = scipy.sparse.csgraph.connected_components(graph)
        component_sizes = np.bincount(labels)
        if np.any(component_sizes <= 5):
            # largest_component = np.where(component_sizes == component_sizes.max())[0][0]
            if verbose:
                print("Connected components sizes: %s" % str(component_sizes))
            large_components = np.isin(labels, np.where(component_sizes > 5)[0])
            if np.mean(large_components) < 0.95:
                if verbose:
                    print("Connected component(s) contains few than 95 percent of all data points. Stop refinement.")
            else:
                from sklearn.neighbors import NearestNeighbors
                _, fillinds = NearestNeighbors(n_neighbors=1).fit(X[large_components, :]).kneighbors(
                    X[~large_components, :])
                g_reconnect = scipy.sparse.csr_matrix((np.repeat(2, len(fillinds)), (
                    np.where(~large_components)[0], np.where(large_components)[0][fillinds[:, 0]])), graph.shape)
                graph = graph + g_reconnect
                if verbose:
                    print("Number of reconnection edges: %d" % np.sum(g_reconnect.data > 0))
                Z = graphdr(X, custom_graph=graph, init=Z, _lambda=_lambda, method=method, refine_iter=refine_iter - 1,
                            refine_threshold=refine_threshold, _refine_threshold=_refine_threshold,
                            no_rotation=no_rotation, rescale=False, n_jobs_nmslib=n_jobs_nmslib, n_jobs=n_jobs,
                            use_cuda=use_cuda)
        else:
            if verbose:
                print("Number of reconnection edges: %d" % 0)
            Z = graphdr(X, custom_graph=graph, init=Z, _lambda=_lambda, method=method, refine_iter=refine_iter - 1,
                        refine_threshold=refine_threshold, _refine_threshold=_refine_threshold,
                        no_rotation=no_rotation, rescale=False, n_jobs_nmslib=n_jobs_nmslib, n_jobs=n_jobs,
                        use_cuda=use_cuda)


    if rescale:
        if no_rotation:
            scale_factor = (np.mean(X * Z, axis=0) - X.mean(axis=0) * Z.mean(axis=0)) / (
                        np.mean(Z ** 2, axis=0) - np.mean(Z, axis=0) ** 2)
            Z = Z * scale_factor[np.newaxis, :]
            Z = Z - Z.mean(axis=0)[np.newaxis, :] + X_mean[np.newaxis, :]
        else:
            XW = np.dot(X, W)
            scale_factor = (np.mean(XW * Z, axis=0) - XW.mean(axis=0) * Z.mean(axis=0)) / (
                        np.mean(Z ** 2, axis=0) - np.mean(Z, axis=0) ** 2)
            Z = Z * scale_factor[np.newaxis, :]
            Z = Z - Z.mean(axis=0)[np.newaxis, :]

    if return_all:
        if no_rotation:
            return Z, graph
        else:
            return Z, W, graph
    else:
        return Z
