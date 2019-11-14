import copy

import numpy as np
import pandas as pd
from numpy.core.umath_tests import matrix_multiply
from pynndescent import NNDescent
import scipy
from scipy.linalg import subspace_angles
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors

from .external import neighbors

def locCov(data, query, bw, shrinkage = False):
    """
    Computes Local Covariance matrices. Data points were weighted with a Gaussian kernel centered at
    the query point

    Parameters
    ----------
    data : 2D array
        Array containing data points of shape (N, p).
    query :  2D array
        Array containing  the query point of shape (1, p)
    bw : float
        Gaussian kernel bandwidth
    shrinkage : bool, optional
        Apply OAS shrinkage for computing local covariance. Default is False.
    """
    D = cdist(data, query) # n x 1
    w = np.exp( - 0.5 * (D / bw)**2) # n x 1

    #C = np.cov(data.T,ddof=1,aweights=w.flatten())
    mu = (np.sum(data * w,axis=0)/np.sum(w))[np.newaxis,:] #1 x p

    X = data - mu
    C = np.dot((X*w).T, X)
    effective_N  = np.sum(w)**2/np.sum(w**2)
    C = C / (np.sum(w)*(1-1/effective_N))
    if shrinkage:
        #Apply OAS with sample size replaced by effective_N
        mu = np.trace(C) / data.shape[1]

        # formula from Chen et al.'s **implementation**
        alpha = np.mean(C ** 2)
        num = alpha + mu ** 2
        den = (effective_N + 1.) * (alpha - (mu ** 2) /  data.shape[1])

        shrinkage = 1. if den == 0 else min(num / den, 1.)
        print(shrinkage)
        C = (1. - shrinkage) * C
        C.flat[::data.shape[1] + 1] += shrinkage * mu

    return C, effective_N


def subspace_angles(A, B):
    """
    Fast computation of subspace angles given A and B are orthogonal basis matrices and
    only accuracy of small angles are of concern. Large angle are not numerically accurate.

    Parameters
    ----------
    A, B : 3D array
        Array containing data points of shape (n, p, p).
    """
    C = B - matrix_multiply(A, matrix_multiply(A.transpose((0, 2, 1)), B))
    angles = np.arcsin(np.clip(np.linalg.svd(C)[1], -1, 1)) / np.pi * 180
    angles = np.minimum(angles, 180 - angles)
    return angles


def extract_structural_backbone(t, data, s, max_angle=90, relaxation=0):
    """
    Construct simplified graphs connecting data points there have been projected to density ridges.

    Two graphs are constructed for different purposes. The first graph (g_simple) is constructed with two major steps:
     1. Construct nearest neighbor graph on both ridge positions
    and raw data positions and combine. 2. Simplify graph so that each point is only connected to up
    to 2^ridge_dimensionality points with a set of filtering criteria.

    The second graph (g_mst) has two extra steps: 3. If the graph is not fully connected, connect the components by
    nearest neigbors between all pairs of components. 4. Construct a minimum spanning tree of the graph.

    Parameters
    -----------
    t : 2D array
        Density ridge positions. Typically projected to density ridges with quasildr.dridge.Scms.
    data : 2D array
        Original data points.
    s : object
        `quasildr.dridge.Scms` object which were used to produce t0.
    max_angle : float, optional
        Maximum angle in degree for filtering graph edges. Default is 90.
    relaxation : float, optional
        The relaxation parameter used to produce `t0`. See `quasildr.dridge.Scms.scms` documention.

    Returns
    -----------
    g_simple : sparse matrix
        Simplified graph constructed without explicit shape or connectivity constraints (g_simple)
    and the other one is  from step 2
    g_mst : sparse matrix
        A tree-shaped graph connecting all points (g_mst). from step 4.
    """
    h, _, _, _, _ = s._nlocal_inv_cov(t)
    eigvals, eigvecs = np.linalg.eigh(-h)
    eigvals = -eigvals
    ridge_dims = ((eigvals[:, 0] - eigvals[:, 1]) / (eigvals[:, 0] - eigvals[:, -1]) < relaxation) + 1


    gknn, _ = neighbors.neighbors(t, n_neighbors=50, smoothknn=False)
    gknnZ, _ = neighbors.neighbors(data, n_neighbors=50, smoothknn=False)
    gknn = gknn + gknn.T + gknnZ + gknnZ.T

    gknn.setdiag(0)
    gknn.eliminate_zeros()

    # remove edge connecting structures of different dimensionality
    gknn.data[ridge_dims[gknn.nonzero()[0]] != ridge_dims[gknn.nonzero()[1]]] = 0
    gknn.eliminate_zeros()


    # filter edges
    edges = t[gknn.nonzero()[1], :] - t[gknn.nonzero()[0], :]
    edges_norm = edges / (np.linalg.norm(edges, axis=1)[:, np.newaxis])

    angles = np.zeros(len(gknn.nonzero()[0]))
    angles_edge = np.zeros(len(gknn.nonzero()[0]))
    for d in np.unique(ridge_dims):
        if d == 1:
            ind = ridge_dims[gknn.nonzero()[0]] == 1
            angles[ind] = np.arccos(
                np.clip(np.sum(eigvecs[gknn.nonzero()[0][ind], :, 0] * eigvecs[gknn.nonzero()[1][ind], :, 0], axis=1),
                        -1, 1)) / np.pi * 180
            angles_edge[ind] = np.arccos(
                np.clip(np.sum(eigvecs[gknn.nonzero()[0][ind], :, 0] * edges_norm[ind, :], axis=1), -1,
                        1)) / np.pi * 180
            angles[ind] = np.minimum(angles[ind], 180 - angles[ind])
            angles_edge[ind] = np.minimum(angles_edge[ind], 180 - angles_edge[ind])
            gknn.data[(angles > max_angle) * (angles_edge > max_angle)] = 0
        else:
            # calculate mean principal angles
            ind = ridge_dims[gknn.nonzero()[0]] == d
            angles[ind] = np.mean(
                subspace_angles(eigvecs[gknn.nonzero()[0][ind], :, :d], eigvecs[gknn.nonzero()[1][ind], :, :d]), axis=1)
            gknn.data[(angles > max_angle)] = 0


    gknn.eliminate_zeros()
    # gknn = gknn + gknn.T
    n_components, labels = scipy.sparse.csgraph.connected_components(gknn)

    # simplify graph by connecting only closest nodes in the subspace
    edges_vecs = t[gknn.nonzero()[1], :] - t[gknn.nonzero()[0], :]
    edges_dist = np.linalg.norm(edges_vecs, axis=1)

    rowinds = []
    colinds = []
    # orient eigen vectors in the same directions
    for d in range(eigvecs.shape[2]):
        eigvecs[eigvecs[:, 0, d] < 0, :, d] *= -1

    for d in np.unique(ridge_dims):
        if d == 1:
            ind = ridge_dims[gknn.nonzero()[0]] == 1
            proj_vecs = np.sum(edges_vecs[ind, :] * eigvecs[gknn.nonzero()[0][ind], :, 0], axis=1)[:,
                        np.newaxis] * eigvecs[gknn.nonzero()[0][ind], :, 0]
        else:
            ind = ridge_dims[gknn.nonzero()[0]] == d
            proj_vecs = matrix_multiply(eigvecs[gknn.nonzero()[0][ind], :, :d], \
                                        matrix_multiply(eigvecs[gknn.nonzero()[0][ind], :, :d].transpose((0, 2, 1)),
                                                        edges_vecs[ind, :, np.newaxis]))

        proj_dists = edges_dist[ind]
        gknn.data[ind] = proj_dists
        gknn_directions = []
        for k in range(d):
            gknn_directions.append(gknn.copy())
            gknn_directions[k].data[ind] = proj_vecs[:, k].squeeze()

        from itertools import product
        for directions in list(product([-1, 1], repeat=len(gknn_directions))):
            for i in np.where(ridge_dims == d)[0]:
                dist_data = gknn[i, :].data
                direction_datas = [gd[i, :].data for gd in gknn_directions]
                conditions = [(directions[i] * direction_datas[i]) > 0 for i in range(len(directions))]
                conditions = np.all(np.vstack(conditions).T, axis=1)
                if np.any(conditions):
                    min_dist = np.min(dist_data[conditions])
                    if directions[0] < 0:
                        rowinds.append(gknn[i, :].nonzero()[1][dist_data == min_dist][0])
                        colinds.append(i)
                    else:
                        rowinds.append(i)
                        colinds.append(gknn[i, :].nonzero()[1][dist_data == min_dist][0])

    gedges = np.vstack([rowinds, colinds]).T
    gedges = np.unique(gedges, axis=0)
    g_simple = csr_matrix((np.repeat(1, gedges.shape[0]), (gedges[:, 0], gedges[:, 1])), shape=gknn.shape)

    n_components, labels = scipy.sparse.csgraph.connected_components(g_simple)

    # meta graph connecting each component.
    components_dimensionality = []
    for i in range(n_components):
        components_dimensionality.append(ridge_dims[labels == i][0])

    # To connect or not to connect

    fc_metaedges = []
    fc_edge_indices = []
    for i in range(n_components - 1):
        i_inds = np.where(labels == i)[0]
        if len(i_inds) > 1000:
            index_group_i = NNDescent(t[i_inds, :])
            index_group_data_i = NNDescent(data[i_inds, :])
        else:
            index_group_i = NearestNeighbors(n_neighbors=1).fit(t[i_inds, :])
            index_group_data_i = NearestNeighbors(n_neighbors=1).fit(data[i_inds, :])

        # for g_mst
        for j in range(i + 1, n_components):
            j_inds = np.where(labels == j)[0]
            if len(i_inds) > 1000:
                nn, _ = index_group_i.query(t[j_inds, :], k=1)
                _, dist = index_group_data_i.query(data[j_inds, :], k=1)
            else:
                _, nn = index_group_i.kneighbors(t[j_inds, :])
                dist, _ = index_group_data_i.kneighbors(data[j_inds, :])

            mindist = np.min(dist)
            fc_metaedges.append([i, j])
            fc_edge_indices.append([i_inds[nn[dist == mindist]][0], j_inds[np.where(dist == mindist)[0]][0]])


    if len(fc_edge_indices) > 0:
        fc_edge_indices = np.vstack(fc_edge_indices)
        g_fc_connections = csr_matrix(
            (np.repeat(2, fc_edge_indices.shape[0]), (fc_edge_indices[:, 0], fc_edge_indices[:, 1])), shape=gknn.shape)
        g_fc = g_simple + g_fc_connections
    else:
        g_fc = g_simple
    g_fc.data = np.linalg.norm(t[g_fc.nonzero()[0], :] - t[g_fc.nonzero()[1], :], axis=1)
    g_mst = minimum_spanning_tree(g_fc)
    g_simple.data = np.linalg.norm(t[g_simple.nonzero()[0], :] - t[g_simple.nonzero()[1], :], axis=1)

    return g_simple, g_mst, ridge_dims



def make_mst(t):
    """
    Construct minimum spanning tree.
    Takes input from Euclidean space and construct graph based on pairwise distances

    Parameters
    -----------
    t : 2D array
        Input data points.

    Returns
    -----------
    edge_list : 2D array
        Edge list of the minimum spanning tree
    """
    return mst(squareform(pdist(t[:, :])))


def mst(D):
    """
    Construct minimum spanning tree from dense matrix.

    Parameters
    -----------
    D : array or sparse matrix
        Edge data in 2D square array or matrix format. Non-zero values represent edges.

    Returns
    -----------
    edge_list : 2D array
        Edge list of the minimum spanning tree
    """
    edge_list = minimum_spanning_tree(csgraph_from_dense(D)).nonzero()
    edge_list = np.vstack(edge_list).T
    return edge_list


def make_projected_mst(t, data, radius=0.1):
    """
    Experimental function
    """
    dist_mat = squareform(pdist(data))
    smooth_mat = np.exp(-(squareform(pdist(t)) / radius) ** 2)
    smooth_mat = smooth_mat / smooth_mat.sum(axis=1)[:, np.newaxis]
    dist_mat_transformed = np.dot(np.dot(smooth_mat, dist_mat), smooth_mat.T)
    return mst(dist_mat_transformed)


def extract_segments(edge_list, degree):
    """
    Extract segments from a graph. It can be used for, for example, paritioning a graph into different branches.

    Parameters
    ----------
    edge_list : 2D array or list
        2D array or list where each row or item has two values indicating start and end indices.
    degree: 1D array
        1D array containing the degree of each node. This has to be consistent with edge_list.

    Returns
    -------
    segments : list
        Partitioned segments represented by a list in which each item is a list containing indices in a segments.
    """
    terminals = np.where((degree != 2) * (degree != 0))[0].tolist()
    edge_dict = {}

    for edge in edge_list:
        if edge[0] in edge_dict:
            edge_dict[edge[0]].append(edge[1])
        else:
            edge_dict[edge[0]] = [edge[1]]
        if edge[1] in edge_dict:
            edge_dict[edge[1]].append(edge[0])
        else:
            edge_dict[edge[1]] = [edge[0]]

    # make raw unpruned segments
    segments = []
    edge_dict_copy = copy.deepcopy(edge_dict)
    while len(terminals) > 0:
        seg = []
        seg.append(terminals[0])
        current_cell = terminals[0]
        notfound = False
        while not notfound:
            notfound = True
            for i in copy.copy(edge_dict_copy[current_cell]):
                edge_dict_copy[current_cell].remove(i)
                edge_dict_copy[i].remove(current_cell)
                if i not in terminals:
                    if notfound:
                        notfound = False
                        new_cell = i
                        seg.append(i)
            if notfound:
                assert len(edge_dict_copy[seg[-1]]) == 0
            else:
                current_cell = new_cell
                assert len(edge_dict_copy[seg[-2]]) == 0

        segments.append(seg)
        terminals.remove(terminals[0])

    return segments


def count_degree(edge_list, N):
    """
    Compute degrees of all nodes in a graph.

    Parameters
    ----------
    edge_list : 2D array or list
        2D array or list where each row or item has two values indicating start and end indices.
    N : int
        Number of nodes in the graph.

    Returns
    -------
    degree : 1D array
        1D array containing degrees of each node

    """
    degree = np.zeros(N)
    counted = np.unique(edge_list.flatten(), return_counts=True)
    degree[counted[0]] = counted[1]
    return degree


def make_trajectory(t, output_prefix=None, prune_threshold=3):
    """
    Construct MST, prune minor branches, and segment.

    Parameters
    ----------
    t
    output_prefix
    prune_threshold

    Returns
    -------

    """
    # initialize trajectory with MST
    edge_list = make_mst(t)
    edge_list = np.asarray(edge_list)
    degree = count_degree(edge_list, t.shape[0])

    segments = extract_segments(edge_list, degree)

    # prune minor branches
    seglens = np.asarray([len(seg) for seg in segments if len(seg) != 0])
    seg_min_degrees = np.asarray([np.min(degree[seg]) for seg in segments if len(seg) != 0])
    remove_seginds = (seglens <= prune_threshold) * (seg_min_degrees == 1)
    while np.any(remove_seginds):
        remove_nodeinds = np.concatenate([segments[i] for i in np.where(remove_seginds)[0]])
        # remove_nodeinds = segments[np.where(remove_seginds)[0][np.argmin(seglens[np.where(remove_seginds)[0]])]]
        edge_list = np.asarray(
            [edge for edge in edge_list if edge[0] not in remove_nodeinds and edge[1] not in remove_nodeinds])
        degree = count_degree(edge_list, t.shape[0])
        segments = extract_segments(edge_list, degree)
        seglens = np.asarray([len(seg) for seg in segments if len(seg) != 0])
        seg_min_degrees = np.asarray([np.min(degree[seg]) for seg in segments if len(seg) != 0])
        remove_seginds = (seglens <= prune_threshold) * (seg_min_degrees == 1)

    if output_prefix is not None:
        for i, seg in enumerate(segments):
            # get some basic ordering
            if np.sum(t[seg[-1], :] - t[seg[0], :]) < 0:
                seg = list(reversed(seg))

            np.savetxt(output_prefix + '.segment.' + str(i), seg, fmt='%d')
    return segments, edge_list
