## An analytical framework for interpretable and generalizable single-cell data analysis

Quasildr is a python library for quasilinear data representation methods. 
It implements two methods, a data representation or visualization 
method **GraphDR** and a generalized trajectory extraction and inference method **StructDR** (StructDR is based on nonparametric ridge estimation). The Quasildr package is developed for 
single-cell omics data analysis, but supports other 
data types as well. The manuscript is available [**here**](https://www.nature.com/articles/s41592-021-01286-1).


## Install

You can install with `pip install quasildr` (or  with `conda install -c main -c conda-forge -c bioconda quasildr`). You can also clone the respository and install with `git clone https://github.com/jzthree/quasildr; cd quasildr; python setup.py install`.


## Quick Start

For learning about the package, we recommend checking out the [**tutorials**](https://github.com/jzthree/quasildr/blob/master/tutorials). We provide them in both jupyter notebooks format (you may use nteract https://nteract.io/ to open them) or html files rendered from jupyter notebooks. The visualizations are large so Github does not allow preview, and you need to download it first. For various manuscript examples, checkout jupyter notebooks in the [**Manuscript**](https://github.com/jzthree/quasildr/blob/master/Manuscript) directory.

As a quickest possible introduction, a minimum example python snippet that running these methods are below

```python
#GraphDR 
from quasildr.graphdr import graphdr
Z = graphdr(X_pca, regularization=500, no_rotation=True)

#StructDR
from quasildr.structdr import Scms
Z = Z / Z[:,0].std()
s = Scms(Z, bw=0.1, min_radius = 10)
T = s.scms(Z)
```

If you are analyzing single-cell data, you may consider using our 
graphical interface for single-cell omics data analysis [Trenti](#graphical-interface).

## Update log
v0.2.2 (10/05/2021): Update the Trenti graphical interface app to use Dash 2.0. Bug fixes for Trenti and speed improvement from Dash 2.0.0.
Please update to Dash 2.0 if you will use Trenti. 


## Documentation
See full API documentation [here](https://quasildr.readthedocs.io/en/latest/main.html). For a high-level introduction to two main methods in quasildr, GraphDR and StructDR (DR means Data Representation):


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

### Further tips for GraphDR:

- Use `no_rotation=True` to not apply any rotation to the feature (e.g. gene or PC) dimension. This makes the output directly comparable with the input. For example, you can use it with PCA input so that the output preserves the interpretion of the PC dimensions, or you can use it directly with gene x cell input to keep the ability to apply any linear transformation later (you may still want to construct graph with PCs and provide graphdr with the custom graph - see below).
- You can provide custom graphs to use with GraphDR with the `custom_graph` argument, and you can ask graphdr to return the graph it constructed too with `return_all=True`. Constructing custom graphs allow you to incorporate experimental design information (batches or time-series) - see the manuscript or the `GraphDR_complexdesign_*.ipynb` notebooks under the Manuscript directory. 
- Selection of `regularization` parameter controls the amount of global shrinkage toward neighbors in graph. With lower regularization parameters, the output will be closer to a linear transformation. Higher regularization parameter applies more shrinkage and while the visualization is usually robust to high values of regularization, very high regularization can shrink all the values toward its center of mass (if you apply it to PCA transformed input, you will observe more shrinkage in higher PCs, which is in fact an expected and desired outcome). You should be mindful of this effect if you intend to compare input with output numerically (it usually does not matter for visualization purpose), and we provide a `rescale` to adjust the scale of the output to be more comparable to the input.
- You can finetune the visualization by controlling the pruning of some of the edges in the graph which is off by default. Checkout documentations about `refine_iter` and `refine_threshold`.
- GraphDR supports GPU. You can use it via `use_cuda=True`.

### Further tips for StructDR:
- Choosing the appropriate bandwidths is important. If you use the CLI (run_structdr.py) it implemented an automatic guess for a bandwidth which works for a wide range of datasets, but we recommend you to try a few bandwidth and compare the results. You can specify bandwidth through two parameters a fixed bandwidth by `bw` and an optional adaptive bandwidth controlled by `min_radius`. The `min_radius` parameter (default to 10) set the adaptive bandwidth to be the distance to the min_radius-th nearest neighbor. The final bandwidth is the maximum between the fixed bandwidth and the adaptive bandwidth, therefore you can specify these values get results with completely fixed or completely adaptive bandwidths or a combination of the two.
- You can project any data to density ridges using the scms method of the object, not just your input data that defines the density ridges.
- If the mapping between data to the positions in density ridges are important for your application, you can reduce the stepsize to integrate through the (projected) gradient curve more accurately (it can lead to slower convergence though). If you only need to extract the density ridges then it does not matter. The default should still work well for most cases even if you use the mapping though.
- If you use the confidence set inference, note that it requires the input to be processed in a way that does not introduce extra dependencies among cells. Generally raw data and linear transformations are fine (StructDR does not model the uncertainty of the linear transform itself though), and most nonlinear methods including GraphDR are not supported. 

### Graphical Interface - Trenti

We developed a web-based GUI, Trenti (Trajectory exploration interface), for single cell data visualization and exploratory analysis, supporting GraphDR, StructDR, common dimensionality reduction and clustering methods, and provide a 3D interface for visualization and a gene expression exploration interface. We developed the interface to support using 3D representations from GraphDR for data exploration tasks (2D is fine, but you get extra information from 3D). There are some extra tools and new features that we put in Trenti too that you may find useful : ).

To use Trenti, you need to install additional dependencies:
`pip install umap-learn dash==2.0.0 dash-colorscales networkx`

See [./trenti/README.md](https://github.com/jzthree/quasildr/blob/master/trenti/README.md) for details. For a quick-start example, run
` python ./trenti/app.py -i ./example/Dentate_Gyrus.data_pca.gz   -f ./example/Dentate_Gyrus.spliced_data.gz -a ./example/Dentate_Gyrus.anno.gz  --samplelimit=5000 --log --mode graphdr` then visit `localhost:8050` in your browser.

![Screenshot of Trenti](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/Trenti.png "StructDR")

Note: even though Trenti is a web interface, it is meant to be used as a single user application because multiple users's actions can interfere with each other.

