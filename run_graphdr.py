# -*- coding: utf-8 -*-


"""
Run GraphDR.

Usage:
    run_graphdr <input_file> [--max_dim=<max_dim>] [--anno_file=<anno_file>] [--anno_column=<anno_column>] 
  [--n_neighbors=<n_neighbors>] [--reg=<reg>] [--refine_iter=<refine_iter>] 
  [--refine_threshold=<refine_threshold>] [--method=<method>] [--metric=<metric>] 
  [--no_rotation] [--rescale]  [--plot] [--pca|--lda]  [--log] [--transpose] [--scale] [--output=<output>] [--suffix=<suffix>]

Options:
  -h --help                                 Show this screen.
  --version                                 Show version.
  --max_dim=<max_dim>                       Number of input dims to use [default: 20]
  --anno_file=<anno_file>                   Annotation of categories used for plotting or use with `--lda`.
  --anno_column=<anno_column>               Name of the column to use in annotation file [default: group_id].
  --n_neighbors=<n_neighbors>               Number of neighbors to construct KNN graph [default: 10].
  --reg=<reg>                               Regularization parameter  [default: 100]
  --refine_iter=<refine_iter>               Refine iteration [default: 0].
  --refine_threshold=<refine_threshold>     Refine threshold [default: 12].
  --method=<method>                         Method [default: auto].
  --metric=<metric>                         Metric [default: euclidean].
  --no_rotation                             Run GraphDR with --no_rotation option.
  --rescale                                 Postprocess output by rescaling to match input mean and variance.
  --plot                                    Generate a pdf plot of the first two dims of the representation.
  --pca                                     Preprocess input with PCA.
  --lda                                     Preprocess input with LDA (using `anno_column` in `anno_file` as labels)
  --log                                     Preprocess input with log(1+X) transform.
  --transpose                               Preprocess input by transposing the matrix.
  --scale                                   Preprocess input by scaling to unit variance.
  --output=<output>                         Output file prefix, use input_file if not specified [default: ]
  --suffix=<suffix>                         Suffix append to output file name [default: ]

"""

import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings

from quasildr.graphdr import *



if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.01')
    print(arguments)
    if arguments['--output'] != '':
        output = arguments['--output']
    else:
        output = arguments['<input_file>']
        
    DOCSTR = ".dim" + str(arguments['--max_dim']) + "_k" + str(arguments['--n_neighbors']) + \
        "_reg" + str(arguments['--reg']) +\
        "_n"+str(arguments['--refine_iter'])+ "t"+str(arguments['--refine_threshold']) + \
        ("_pca" if arguments['--pca'] else "") + ("_lda" if arguments['--lda'] else "") + \
        ("_no_rotation" if arguments['--no_rotation'] else "") + \
        ("_rescale" if arguments['--rescale'] else "") + \
        ("_log" if arguments['--log'] else "")+ \
        ("_scale" if arguments['--scale'] else "") + \
        ("_transpose" if arguments['--transpose'] else "") +  arguments['--suffix']
    data = pd.read_csv(arguments['<input_file>'],sep='\t')
    if isinstance(data.iloc[0,0],str):
        data = data.iloc[:,1:]
    if arguments['--log']:
        data = np.log(1+data)
    if arguments['--transpose']:
        data = data.T
    if arguments['--scale']:
        data = data / data.values.std(axis=0)


    if arguments['--anno_file']:
        anno = pd.read_csv(arguments['--anno_file'],sep='\t')
    if arguments['--pca']:
        data = PCA(int(arguments['--max_dim']),iterated_power=10).fit_transform(data.values)
    elif arguments['--lda'] and arguments['--anno_file'] is not None:
        data = LDA().fit_transform(data.values, anno['group_id'])[:,:int(arguments['--max_dim'])]
    else:
        data = data.values

    data = data / data[:, 0].std()

    Z = graphdr(data, n_neighbors= int(arguments['--n_neighbors']),
        regularization=float(arguments['--reg']), refine_iter=int(arguments['--refine_iter']),
        refine_threshold=float(arguments['--refine_threshold']),no_rotation=arguments['--no_rotation'],
        rescale=arguments['--rescale'], method=arguments['--method'], metric=arguments['--metric'])

    pd.DataFrame(Z).to_csv(output + DOCSTR + '.graphdr',
                sep='\t', index_label=False)

    if arguments['--plot']:
        try:
            from plotnine import ggplot, aes, geom_point, theme_minimal

            if arguments['--anno_file']:
                df = pd.DataFrame({'x': Z[:, 0], 'y': Z[:, 1], 'c': anno[arguments['--anno_column']].map(str)})
                p = ggplot(df, aes('x', 'y', color='c')) + geom_point(size=0.1) + theme_minimal()
            else:
                df = pd.DataFrame({'x': Z[:, 0], 'y': Z[:, 1]})
                p = ggplot(df, aes('x', 'y')) + geom_point(size=0.1) + theme_minimal()

            p.save(output + DOCSTR + '.pdf')
        except ImportError:
            warnings.warn('plotnine needs to be installed for the plotting function.')
