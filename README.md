## Quasilinear data representations for single-cell omics data analysis

Quasildr is a python library for quasilinear data representation methods. 
It mainly implements two methods, a quasilinear data representation or visualization 
method **GraphDR** and a generalized trajectory extraction and inference method **StructDR** (StructDR is based on nonparametric ridge estimation). The Quasildr package is developed for 
single-cell omics data analysis, but supports other 
data types as well. 


## Install

Run `pip install .` in this directory or use `pip install quasildr`.


## Quick Start

For learning about the package, we recommend checking out the [**tutorials**](https://github.com/jzthree/quasildr/blob/master/tutorials). We provide them in both jupyter notebooks format (you may use nteract https://nteract.io/ to open them) or html files rendered from jupyter notebooks. The visualizations are large so Github does not allow preview, and you need to download it first. For various manuscript examples, checkout jupyter notebooks in the [**Manuscript**](https://github.com/jzthree/quasildr/blob/master/Manuscript) directory.

As a quickest possible introduction, a minimum example python snippet that running these methods are below

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
See full documentation [here](https://quasildr.readthedocs.io/en/latest/main.html). For a high-level introduction to two main methods in quasildr, GraphDR and StructDR (DR means Data Representation):

### GraphDR - visualization and general-purpose representation: 
GraphDR is a nonlinear representation method 
that preserves the interpretation of a corresponding linear space, while being able to well represent cell
 identities like nonlinear methods. Unlike popular nonlinear methods, GraphDR allows direct 
 comparison across datasets by applying a common linear transform. GraphDR also supports incorporating 
 complex experiment design through graph construction (see example from manuscript and ./Manuscript directory). 
 GraphDR is also very fast. It can process a 1.5 million-cell dataset in 5min (CPU) or 1.5min (CPU) and 
 can easily scale to even larger datasets.
![Schematic overview of GraphDR](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/GraphDR.png "GraphDR")

### StructDR - flexible structure extraction and inference of confidence sets: 
StructDR is based on nonparametric density ridge estimation (NRE). StructDR is a flexible framework 
for structure extraction for single-cell data that unifies cluster, trajectory, and surface estimation 
by casting these problems as identifying 0-, 1-, and 2- dimensional density ridges. StructDR also support
 adaptively decides ridge dimensionality based on data. When used with linear representation such as PCA, 
 StructDR allows inference of confidence sets of density ridge positions. This allows, for example, 
 estimation of uncertainties of the developmental trajectories extracted.
![Schematic overview of StructDR](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/StructDR.png "StructDR")


### Command-line tools 

We also provide command-line tools to run those methods without writing any code. Basic single-cell data preprocessing options are provided in `run_graphdr.py`, even though we generally recommend preprocessing single cell data with a dedicated package such as scanpy or Seurat to select highly variable genes and normalize before providing it to GraphDR. You can add the `-h` option to access help information to each tool.

* run_graphdr.py
```
python run_graphdr.py ./example/Dentate_Gyrus.spliced_data.gz --pca --plot --log --transpose --scale --max_dim 50 --refine_iter 4 --reg 500 --no_rotation --anno_file ./example/Dentate_Gyrus.anno.gz --anno_column ClusterName 
```

* run_structdr.py
```
python run_structdr.py --bw 0.1 --automatic_bw 0 --input ./example/Dentate_Gyrus.spliced_data.gz.dim50_k10_reg500_n4t12_pca_no_rotation_log_scale_transpose.graphdr.small.gz  --anno_file ./example/Dentate_Gyrus.anno.small.gz --anno_column ClusterName  --output ./example/Dentate_Gyrus.spliced_data.gz.dim50_k10_reg500_n4t12_pca_no_rotation_log_scale_transpose.graphdr.small.gz
```

### Graphical Interface - Trenti

We developed a web-based GUI, Trenti (Trajectory exploration interface), for single cell data visualization and exploratory analysis, supporting GraphDR, StructDR, common dimensionality reduction and clustering methods, and provide a 3D interface for visualization and a gene expression exploration interface. 

To use Trenti, you need to install additional dependencies:
`pip install umap-learn dash==1.9.1 dash-colorscales networkx`

See [./trenti/README.md](https://github.com/jzthree/quasildr/blob/master/trenti/README.md) for details. For a quick-start example, run
` python ./trenti/app.py -i ./example/Dentate_Gyrus.data_pca.gz   -f ./example/Dentate_Gyrus.spliced_data.gz -a ./example/Dentate_Gyrus.anno.gz  --samplelimit=5000 --log --mode graphdr` then visit `localhost:8050` in your browser.

![Screenshot of Trenti](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/Trenti.png "StructDR")

