#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
Run a set of dimensionality reduction / visualization methods.

Usage:
  run_others.py <input_file> 

Options:
  -h --help                                 Show this screen.
  --version                                 Show version.

"""

from docopt import docopt
import pandas as pd
import scanpy.api as sc

#import scms2


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.01')
    f = arguments['<input_file>']
    data = pd.read_csv(f,sep='\t',index_col=0)
    dataanno= pd.read_csv(f.replace('expression','anno'),sep='\t',index_col=0)

    adata = sc.AnnData(data.values, data.index.values,data.columns.values)
    adata.var_names_make_unique()

    sc.tl.pca(adata, n_comps=20)
    sc.tl.phate(adata,n_components=20)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata,n_components=2)
    sc.tl.tsne(adata)

    pd.DataFrame(adata.obsm['X_phate'],index=data.index.values, columns=['Phate'+str(i) for i in range(adata.obsm['X_phate'].shape[1])]).to_csv(f+'.phate20',index_label=False,sep='\t')
    pd.DataFrame(adata.obsm['X_umap'],index=data.index.values, columns=['UMAP'+str(i) for i in range(adata.obsm['X_umap'].shape[1])]).to_csv(f+'.umap2',index_label=False,sep='\t')
    pd.DataFrame(adata.obsm['X_tsne'],index=data.index.values, columns=['tsne'+str(i) for i in range(adata.obsm['X_tsne'].shape[1])]).to_csv(f+'.tsne2',index_label=False,sep='\t')
    pd.DataFrame(adata.obsm['X_pca'],index=data.index.values, columns=['PCA'+str(i) for i in range(adata.obsm['X_pca'].shape[1])]).to_csv(f+'.pca20',index_label=False,sep='\t')


    
    sc.tl.umap(adata,n_components=20)
    pd.DataFrame(adata.obsm['X_umap'],index=data.index.values, columns=['UMAP'+str(i) for i in range(adata.obsm['X_umap'].shape[1])]).to_csv(f+'.umap20',index_label=False,sep='\t')



    sc.pp.neighbors(adata, method='gauss')
    sc.tl.diffmap(adata,n_comps=20)
    pd.DataFrame(adata.obsm['X_diffmap'],index=data.index.values, columns=['Diffmap'+str(i) for i in range(adata.obsm['X_diffmap'].shape[1])]).to_csv(f+'.diffmap20',index_label=False,sep='\t')
