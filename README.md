## Quasilinear data representations for single-cell omics data analysis

Quasildr is a python library for quasilinear data representation methods. 
It mainly implements two methods, a quasilinear data representation or visualization 
method **GraphDR** and a generalized trajectory extraction and inference method **StructDR** (StructDR is based on nonparametric ridge estimation). The Quasildr package is developed for 
single-cell omics data analysis, but you can use it on other 
data types as well. 


## Install

Run `pip install .` in this directory.


## Quick Start

For quick start, see tutorials under `./tutorials/` directory.

An example python snippet that running these methods are below

```python
#GraphDR 
import quasildr.graphdr import graphdr
Z = graphdr(X_pca, regularization=500)

#StructDR
import quasildr.structdr import Scms
Z = Z / Z[:,0].std()
s = Scms(Z, bw=0.1, min_radius = 10)
T = s.scms(Z)
```

If you are analyzing single-cell data, you may consider using our 
graphical interface for single-cell omics data analysis [Trenti](#graphical-interface).


## Documentation
See documentation [here](https://quasildr.readthedocs.io/en/latest/main.html).

# GraphDR - visualization and general-purpose representation: 
GraphDR is a nonlinear representation method 
that preserves the interpretation of a corresponding linear space, while being able to well represent cell
 identities like nonlinear methods. Unlike popular nonlinear methods, GraphDR allows direct 
 comparison across datasets by applying a common linear transform. GraphDR also supports incorporating 
 complex experiment design through graph construction (see example from manuscript and ./Manuscript directory). 
 GraphDR is also very fast. It can process a 1.5 million-cell dataset in 5min (CPU) or 1.5min (CPU) and 
 can easily scale to even larger datasets.
![Schematic overview of GraphDR](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/GraphDR.png "GraphDR")

# StructDR - flexible structure extraction and inference of confidence sets: 
StructDR is based on nonparametric density ridge estimation (NRE). StructDR is a flexible framework 
for structure extraction for single-cell data that unifies cluster, trajectory, and surface estimation 
by casting these problems as identifying 0-, 1-, and 2- dimensional density ridges. StructDR also support
 adaptively decides ridge dimensionality based on data. When used with linear representation such as PCA, 
 StructDR allows inference of confidence sets of density ridge positions. This allows, for example, 
 estimation of uncertainties of the developmental trajectories extracted.
![Schematic overview of StructDR](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/StructDR.png "StructDR")


## Command-line tools 

We provide command-line tools for quick access to most commonly used quasildr functions, with typical data preprocessing and post processing options built-in. You can add the `-h` option to access help information to each tool.

* run_graphdr.py
```
python run_graphdr.py ./example/Dentate_Gyrus.spliced_data.gz --pca --plot --log --transpose --max_dim 50 --refine_iter 4 --reg 500 --no_rotation --anno_file ./example/Dentate_Gyrus.anno.gz --anno_column ClusterName 
```

* run_structdr.py
```
python run_structdr.py --input ./example/Dentate_Gyrus.spliced_data.gz.dim50_k10_reg500_n4t12_pca_no_rotation_log_scale_transpose.drgraph --anno_file ./example/Dentate_Gyrus.anno.gz --anno_column ClusterName  --output ./example/Dentate_Gyrus.spliced_data.gz.dim50_k10_reg500_n4t12_pca_no_rotation_log_scale_transpose.drgraph
```

## Graphical Interface - Trenti

We developed a web-based GUI, Trenti (Trajectory exploration interface), for single cell data visualization and exploratory analysis, supporting GraphDR, StructDR, common dimensionality reduction and clustering methods, and provide a 3D interface for visualization and a gene expression exploration interface. 

To use Trenti, you need to install additional dependencies:
`pip install umap-learn dash==1.8.0 dash-colorscales`

See [./trenti/README.md](https://github.com/jzthree/quasildr/blob/master/trenti/README.md) for details. For a quick-start example, run
` python ./trenti/app.py -i ./example/Dentate_Gyrus.data_pca.gz   -f ./example/Dentate_Gyrus.spliced_data.gz -a ./example/Dentate_Gyrus.anno.gz  --samplelimit=5000 --log --mode graphdr`

![Screenshot of Trenti](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/Trenti.png "StructDR")

