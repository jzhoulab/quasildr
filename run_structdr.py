# -*- coding: utf-8 -*-
"""
Run nonparametric ridge estimation.
"""

import os
from optparse import OptionParser

import networkx as nx
import numpy as np
import pandas as pd
import time

from networkx.algorithms.centrality import betweenness_centrality
from plotnine import *
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.io import mmwrite
from sklearn.decomposition import PCA

import quasildr
from quasildr import structdr
from quasildr.graphdr import *
from quasildr.utils import *

parser = OptionParser()
# main options
parser.add_option("--input", dest="input", type="str",
                  help="Input file")
parser.add_option("--anno_file", dest="anno_file", type="str",
                  help="Annotation file for plotting")
parser.add_option("--anno_column", dest="anno_column", type="str", default="group_id",
                  help="Name of the column to use in annotation file")
parser.add_option("--output", dest="output", type="str",
                  help="Output prefix")
parser.add_option("--suffix", dest="suffix", type="str", default="",
                  help="Output suffix")
parser.add_option("--niter", dest="niter", type="int", default=30,
                  help="Number of iterations. Default is 30.")
parser.add_option("--ndim", dest="ndim", type="int", default=15,
                  help="Number of input dimensions to use. Default is 15.")
parser.add_option("--bw", dest="bw", type="float", default=0.,
                  help="Gaussian KDE kernel bandwidth. "
                       "This is not needed if `automatic_bw` is specified. Default is 0.")
parser.add_option("--adaptive_bw", dest="adaptive_bw", type="int", default=10,
                  help="Set data point-specific minimum bandwidth to its distance to "
                       "`adaptive_bw`-th nearest neighbor. Default is 10.")
parser.add_option("--automatic_bw", dest="automatic_bw", type="float", default=2.0,
                  help="Use pca to select bw. Default is 2.0.")
parser.add_option("--relaxation", dest="relaxation", type="float", default=0,
                  help="Ridge dimensionality relaxation. Default is 0.")
parser.add_option("--stepsize", dest="stepsize", type="float", default=0.5,
                  help="Step size relative to standard SCMS. Default is 0.5.")
parser.add_option("--method", dest="method", type="str", default="LocInv",
                  help="Method. Default is LocInv")
parser.add_option("--batchsize", dest="batchsize", type="int", default=500,
                  help="Decreasing the batch size reduces memory consumption. Default is 500.")
parser.add_option("--ridge_dimensionality", dest="ridge_dimensionality", type="float", default=1, help="ridge dim")
parser.add_option("--bigdata", action="store_true", dest="bigdata",
                  help="Speed up computation for big datasets with multilevel data represention"
                       " (Experimental feature).")
parser.add_option("--n_jobs", dest="n_jobs", type="int", default=1, help="number of jobs")

# extract backbones options
parser.add_option("--extract_backbone", action="store_true", dest="extract_backbone")
parser.add_option("--k", dest="k", type="int", default=50, help="Number of NN to prescreen edges")
parser.add_option("--maxangle", dest="maxangle", type="float", default=90.,
                  help="Prefilter edges by angles")

np.random.seed(0)
(opt, args) = parser.parse_args()

docstr = "niter" + str(opt.niter) + "_ndim" + str(opt.ndim) + "_bw" + str(opt.bw) + \
         "_adaptive_bw" + str(opt.adaptive_bw) + \
         "_automatic_bw" + str(opt.automatic_bw) + "_relaxation" + str(opt.relaxation) + \
         "_ridge" + str(opt.ridge_dimensionality) + \
         "_method" + str(opt.method) + "_stepsize" + str(opt.stepsize) + "_maxangle" + str(opt.maxangle) + "_k" + str(
    opt.k) + opt.suffix + ("_big" if opt.bigdata else "")

outdir = os.path.dirname(opt.output)
if not os.path.exists(outdir):
    os.makedirs(outdir)

data = pd.read_csv(opt.input, delimiter='\t')
data = np.asarray(data)
data = data / data[:, 0].std(axis=0)

if opt.ndim > 0:
    data = data[:, :opt.ndim]

if opt.automatic_bw != 0:
    bw = PCA(np.minimum(20, data.shape[1])).fit_transform(data).std(axis=0)[-1] * np.sqrt(opt.ndim)
    bw *= opt.automatic_bw
else:
    bw = opt.bw

print("bw: "+str(bw))
if opt.anno_file != "":
    anno = pd.read_csv(opt.anno_file, sep='\t')

if opt.bigdata and data.shape[0] > 5000:
    datas = structdr.multilevel_compression(data)
    s = structdr.Scms(datas[0], bw=bw, min_radius=opt.adaptive_bw)
else:
    s = structdr.Scms(data, bw=bw, min_radius=opt.adaptive_bw)

T, ifilter = s.scms(data, method=opt.method, stepsize=opt.stepsize, n_iterations=opt.niter, threshold=0,
                    ridge_dimensionality=opt.ridge_dimensionality,
                    relaxation=opt.relaxation, n_jobs=opt.n_jobs)

np.savetxt(X=T, fname=opt.output + '.' + docstr + '.trajectory')
np.savetxt(X=ifilter, fname=opt.output + '.' + docstr + '.trajectory.ifilter')

if opt.anno_file != "":
    df = pd.DataFrame({'x': T[:, 0], 'y': T[:, 1], 'c': anno[opt.anno_column].map(str)})
    p = ggplot(df, aes('x', 'y', color='c')) + geom_point(size=0.5) + theme_minimal()
else:
    df = pd.DataFrame({'x': T[:, 0], 'y': T[:, 1]})
    p = ggplot(df, aes('x', 'y')) + geom_point(size=0.5) + theme_minimal()

p.save(opt.output + '.' + docstr + '.pdf')

if opt.extract_backbone:
    g_simple, g_mst, ridge_dims = extract_structural_backbone(T, data, s, max_angle=opt.maxangle,
                                                                    relaxation=opt.relaxation)

    mmwrite(opt.output + '.' + docstr + '.g_simple.mm', g_simple)
    mmwrite(opt.output + '.' + docstr + '.g_mst.mm', g_mst)
    np.savetxt(opt.output + '.' + docstr + '.ridge_dims', ridge_dims, fmt='%d')

    df = pd.DataFrame({'x': T[:, 0], 'y': T[:, 1], 'c': anno[opt.anno_column].map(str)})
    df_e = pd.DataFrame({'xs': T[g_simple.nonzero()[0], 0], 'xe': T[g_simple.nonzero()[1], 0],
                         'ys': T[g_simple.nonzero()[0], 1], 'ye': T[g_simple.nonzero()[1], 1]})
    p = ggplot(df) + \
        geom_segment(mapping=aes(x='xs', xend='xe', y='ys', yend='ye'), data=df_e, size=0.5) + \
        geom_point(mapping=aes('x', 'y', color='c'), size=0.5) + theme_minimal()
    p.save(opt.output + '.' + docstr + '.g_simple.pdf')

    df_e = pd.DataFrame({'xs': T[g_mst.nonzero()[0], 0], 'xe': T[g_mst.nonzero()[1], 0],
                         'ys': T[g_mst.nonzero()[0], 1], 'ye': T[g_mst.nonzero()[1], 1]})
    p = ggplot(df) + \
        geom_segment(mapping=aes(x='xs', xend='xe', y='ys', yend='ye'), data=df_e, size=0.5) + \
        geom_point(mapping=aes('x', 'y', color='c'), size=0.5) + theme_minimal()
    p.save(opt.output + '.' + docstr + '.g_mst.pdf')


    G_simple = nx.from_scipy_sparse_matrix(g_simple)
    nodes_bc = betweenness_centrality(G_simple, k=np.minimum(500, g_simple.shape[0]), normalized=False)
    pd.DataFrame(pd.Series(nodes_bc) / g_simple.shape[0]).to_csv(opt.output + '.' + docstr + '.g_simple.bc.csv')

    G_mst = nx.from_scipy_sparse_matrix(g_mst)
    nodes_bc = betweenness_centrality(G_mst, k=np.minimum(500, g_mst.shape[0]), normalized=False)
    pd.DataFrame(pd.Series(nodes_bc) / g_mst.shape[0]).to_csv(opt.output + '.' + docstr + '.g_mst.bc.csv')
