from typing import Union, Optional, Any, Mapping, Callable

import numpy as np
import scipy
from numpy.random import RandomState
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn.metrics import pairwise_distances


N_PCS = 50  # default number of PCs


def neighbors(
    data: np.ndarray,
    n_neighbors: int = 15,
    knn: bool = True,
    random_state: Optional[Union[int, RandomState]] = 0,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    metric_kwds: Mapping[str, Any] = {},
    copy: bool = False,
    smoothknn: bool = True
) -> ():
    """
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='diffmap'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Parameters
    ----------
    data
        data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.

    knn
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
    random_state
        A numpy random seed.
    metric
        A known metric’s name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    copy
        Return a copy instead of writing to data.

    Returns
    -------
    Depending on `copy`, updates or returns `data` with the following:
    connectivities : sparse matrix ( dtype `float32`)
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    distances : sparse matrix (dtype `float32`)
        Instead of decaying weights, this stores distances for each pair of
        neighbors.
    """
    data = data.copy() if copy else data

    neighbors = Neighbors(data)
    neighbors.compute_neighbors(
        n_neighbors=n_neighbors, knn=knn,
        metric=metric, metric_kwds=metric_kwds,
        random_state=random_state, smoothknn = smoothknn)
    return  neighbors.connectivities, neighbors.distances





def compute_neighbors_umap(
        X, n_neighbors, random_state=None,
        metric='euclidean', metric_kwds={}, angular=False,
        verbose=False):
    """This is from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
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
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.
    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    knn_indices, knn_dists : np.arrays of shape (n_observations, n_neighbors)
    """
    from .umap import sparse
    from .umap.umap_ import rptree_leaf_array, make_nn_descent
    from .umap import distances as dist
    from .umap import sparse
    import scipy
    from sklearn.utils import check_random_state

    INT32_MIN = np.iinfo(np.int32).min + 1
    INT32_MAX = np.iinfo(np.int32).max - 1

    random_state = check_random_state(random_state)

    if metric == 'precomputed':
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:, :n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    else:
        if callable(metric):
            distance_func = metric
        elif metric in dist.named_distances:
            distance_func = dist.named_distances[metric]
        else:
            raise ValueError('Metric is neither callable, ' +
                             'nor a recognised string')

        if metric in ('cosine', 'correlation', 'dice', 'jaccard'):
            angular = True

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if scipy.sparse.isspmatrix_csr(X):
            if metric in sparse.sparse_named_distances:
                distance_func = sparse.sparse_named_distances[metric]
                if metric in sparse.sparse_need_n_features:
                    metric_kwds['n_features'] = X.shape[1]
            else:
                raise ValueError('Metric {} not supported for sparse ' +
                                'data'.format(metric))
            metric_nn_descent = sparse.make_sparse_nn_descent(
                distance_func, tuple(metric_kwds.values()))
            leaf_array = rptree_leaf_array(X, n_neighbors,
                                           rng_state, n_trees=10,
                                           angular=angular)
            knn_indices, knn_dists = metric_nn_descent(X.indices,
                                                       X.indptr,
                                                       X.data,
                                                       X.shape[0],
                                                       n_neighbors,
                                                       rng_state,
                                                       max_candidates=60,
                                                       rp_tree_init=True,
                                                       leaf_array=leaf_array,
                                                       verbose=verbose)
        else:
            metric_nn_descent = make_nn_descent(distance_func,
                                                tuple(metric_kwds.values()))
            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            leaf_array = rptree_leaf_array(X, n_neighbors,
                                           rng_state, n_trees=n_trees,
                                           angular=angular)
            knn_indices, knn_dists = metric_nn_descent(X,
                                                       n_neighbors,
                                                       rng_state,
                                                       max_candidates=60,
                                                       rp_tree_init=True,
                                                       leaf_array=leaf_array,
                                                       n_iters=n_iters,
                                                       verbose=verbose)

        if np.any(knn_indices < 0):
            print('Failed to correctly find n_neighbors for some samples. '
                 'Results may be less than ideal. Try re-running with '
                 'different parameters.')

    return knn_indices, knn_dists


def compute_connectivities_umap(knn_indices, knn_dists,
        n_obs, n_neighbors, set_op_mix_ratio=1.0,
        local_connectivity=1.0, bandwidth=1.0):
    """This is from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    from .umap.umap_ import smooth_knn_dist

    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    sims = np.zeros((n_obs * n_neighbors), dtype=np.float64)
    dists = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors,
                                   local_connectivity=local_connectivity)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                sim = 0.0
                dist = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                sim = 1.0
                dist = knn_dists[i, j]
            else:
                sim = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i] *
                                                              bandwidth)))
                dist = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            sims[i * n_neighbors + j] = sim
            dists[i * n_neighbors + j] = dist

    connectivities = coo_matrix((sims, (rows, cols)),
                               shape=(n_obs, n_obs))
    connectivities.eliminate_zeros()

    distances = coo_matrix((dists, (rows, cols)),
                           shape=(n_obs, n_obs))
    distances.eliminate_zeros()

    transpose = connectivities.transpose()

    prod_matrix = connectivities.multiply(transpose)

    connectivities = set_op_mix_ratio * (connectivities + transpose - prod_matrix) + \
             (1.0 - set_op_mix_ratio) * prod_matrix

    connectivities.eliminate_zeros()
    return distances.tocsr(), connectivities.tocsr()


def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)),
                                      shape=(n_obs, n_obs))
    return result.tocsr()


def get_sparse_matrix_from_indices_distances_numpy(indices, distances, n_obs, n_neighbors):
    n_nonzero = n_obs * n_neighbors
    indptr = np.arange(0, n_nonzero + 1, n_neighbors)
    D = scipy.sparse.csr_matrix((distances.copy().ravel(),  # copy the data, otherwise strange behavior here
                                indices.copy().ravel(),
                                indptr),
                                shape=(n_obs, n_obs))
    D.eliminate_zeros()
    return D


def get_indices_distances_from_sparse_matrix(D, n_neighbors):
    indices = np.zeros((D.shape[0], n_neighbors), dtype=int)
    distances = np.zeros((D.shape[0], n_neighbors), dtype=D.dtype)
    n_neighbors_m1 = n_neighbors - 1
    for i in range(indices.shape[0]):
        neighbors = D[i].nonzero()  # 'true' and 'spurious' zeros
        indices[i, 0] = i
        distances[i, 0] = 0
        # account for the fact that there might be more than n_neighbors
        # due to an approximate search
        # [the point itself was not detected as its own neighbor during the search]
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(D[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = D[i][
                neighbors[0][sorted_indices], neighbors[1][sorted_indices]]
        else:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = D[i][neighbors]
    return indices, distances


def get_indices_distances_from_dense_matrix(D, n_neighbors):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors-1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances



class OnFlySymMatrix:
    """Emulate a matrix where elements are calculated on the fly.
    """
    def __init__(self, get_row, shape, DC_start=0, DC_end=-1, rows=None, restrict_array=None):
        self.get_row = get_row
        self.shape = shape
        self.DC_start = DC_start
        self.DC_end = DC_end
        self.rows = {} if rows is None else rows
        self.restrict_array = restrict_array  # restrict the array to a subset

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, np.integer):
            if self.restrict_array is None:
                glob_index = index
            else:
                # map the index back to the global index
                glob_index = self.restrict_array[index]
            if glob_index not in self.rows:
                self.rows[glob_index] = self.get_row(glob_index)
            row = self.rows[glob_index]
            if self.restrict_array is None:
                return row
            else:
                return row[self.restrict_array]
        else:
            if self.restrict_array is None:
                glob_index_0, glob_index_1 = index
            else:
                glob_index_0 = self.restrict_array[index[0]]
                glob_index_1 = self.restrict_array[index[1]]
            if glob_index_0 not in self.rows:
                self.rows[glob_index_0] = self.get_row(glob_index_0)
            return self.rows[glob_index_0][glob_index_1]

    def restrict(self, index_array):
        """Generate a view restricted to a subset of indices.
        """
        new_shape = index_array.shape[0], index_array.shape[0]
        return OnFlySymMatrix(self.get_row, new_shape, DC_start=self.DC_start,
                              DC_end=self.DC_end,
                              rows=self.rows, restrict_array=index_array)


class Neighbors:
    """Data represented as graph of nearest neighbors.

    Represent a data matrix as a graph of nearest neighbor relations (edges)
    among data points (nodes).

    Parameters
    ----------
    data
        Annotated data object.

    """

    def __init__(self, data: np.ndarray):
        self._data = data
        # use the graph in data
        info_str = ''
        self.knn = None
        self._distances = None
        self._connectivities = None
        self._number_connected_components = None

        self._eigen_values = None
        self._eigen_basis = None
        if info_str != '':
            print('    initialized {}'.format(info_str))

    @property
    def distances(self):
        """Distances between data points (sparse matrix).
        """
        return self._distances

    @property
    def connectivities(self):
        """Connectivities between data points (sparse matrix).
        """
        return self._connectivities

    @property
    def transitions(self):
        """Transition matrix (sparse matrix).

        Is conjugate to the symmetrized transition matrix via::

            self.transitions = self.Z *  self.transitions_sym / self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.

        Notes
        -----
        This has not been tested, in contrast to `transitions_sym`.
        """
        if issparse(self.Z):
            Zinv = self.Z.power(-1)
        else:
            Zinv = np.diag(1./np.diag(self.Z))
        return self.Z.dot(self.transitions_sym).dot(Zinv)

    @property
    def transitions_sym(self):
        """Symmetrized transition matrix (sparse matrix).

        Is conjugate to the transition matrix via::

            self.transitions_sym = self.Z /  self.transitions * self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.
        """
        return self._transitions_sym

    @property
    def eigen_values(self):
        """Eigen values of transition matrix (numpy array).
        """
        return self._eigen_values

    @property
    def eigen_basis(self):
        """Eigen basis of transition matrix (numpy array).
        """
        return self._eigen_basis

    @property
    def laplacian(self):
        """Graph laplacian (sparse matrix).
        """
        return self._laplacian


    def compute_neighbors(
        self,
        n_neighbors: int = 30,
        knn: bool = True,
        random_state: Optional[Union[RandomState, int]] = 0,
        write_knn_indices: bool = False,
        metric: str = 'euclidean',
        metric_kwds: Mapping[str, Any] = {},
        smoothknn: bool = True
    ) -> None:
        """\
        Compute distances and connectivities of neighbors.

        Parameters
        ----------
        n_neighbors
             Use this number of nearest neighbors.
        knn
             Restrict result to `n_neighbors` nearest neighbors.


        Returns
        -------
        Writes sparse graph attributes `.distances` and `.connectivities`.
        Also writes `.knn_indices` and `.knn_distances` if
        `write_knn_indices==True`.
        """
        if n_neighbors > self._data.shape[0]:  # very small datasets
            n_neighbors = 1 + int(0.5*self._data.shape[0])
            print('Warning: n_obs too small: adjusting to `n_neighbors = {}`'
                      .format(n_neighbors))

        if self._data.shape[0] >= 10000 and not knn:
            print(
                'Warning: Using high n_obs without `knn=True` takes a lot of memory...')
        self.n_neighbors = n_neighbors
        self.knn = knn
        X = self._data
        # neighbor search

        knn_indices, knn_distances = compute_neighbors_umap(
            X, n_neighbors, random_state, metric=metric, metric_kwds=metric_kwds)
        # write indices as attributes
        if write_knn_indices:
            self.knn_indices = knn_indices
            self.knn_distances = knn_distances

        if smoothknn:
            # we need self._distances also for method == 'gauss' if we didn't
            # use dense distances
            self._distances, self._connectivities = compute_connectivities_umap(
                knn_indices, knn_distances, self._data.shape[0], self.n_neighbors)
        else:
            s = np.repeat(np.arange(knn_indices.shape[0]),knn_indices.shape[1])
            t = knn_indices.flatten()
            w = np.ones(t.shape)
            self._connectivities = scipy.sparse.csr_matrix((w,(s.astype(np.int),t.astype(np.int))),(X.shape[0],X.shape[0]))
            self._distances = scipy.sparse.csr_matrix((knn_distances.flatten(),(s.astype(np.int),t.astype(np.int))),(X.shape[0],X.shape[0]))

        self._number_connected_components = 1
        if issparse(self._connectivities):
            from scipy.sparse.csgraph import connected_components
            self._connected_components = connected_components(self._connectivities)
            self._number_connected_components = self._connected_components[0]
