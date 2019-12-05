# Quasilinear data representation

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

#NRE
import quasildr.structdr import Scms
Z = Z / Z[:,0].std()
s = Scms(Z, bw=0.1, min_radius = 10)
T = s.scms(Z)
```

If you are analyzing single-cell data, you may consider using our 
graphical interface for single-cell omics data analysis [Trenti](#graphical-interface).


## Documentation
See documentation here.

“Quasilinear” approaches: GraphDR - visualization and general-purpose representation
![Schematic overview of GraphDR](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/GraphDR.png "GraphDR")

“Quasilinear” approaches: StructDR - flexible structure extraction and inference of confidence sets
![Schematic overview of StructDR](https://github.com/jzthree/quasildr/blob/master/docs/source/_static/StructDR.png "StructDR")


## Commandline tools 

We provide commandline tools for quick access to most commonly used quasildr functions, with typical data preprocessing and post processing options built-in. You can add the `-h` option to access help information to each tool.

* run_graphdr.py
```
python run_graphdr.py ./example/Dentate_Gyrus.spliced_data.gz --pca --plot --log --transpose --max_dim 50 --refine_iter 4 --reg 500 --no_rotation --anno_file ./example/Dentate_Gyrus.anno.gz --anno_column ClusterName 
```

* run_structdr.py
```
 python run_structdr.py --input ./example/Dentate_Gyrus.spliced_data.gz.dim50_k10_reg500_n4t12_pca_no_rotation_log_scale_transpose.drgraph --anno_file ./example/Dentate_Gyrus.anno.gz --anno_column ClusterName  --output ./example/Dentate_Gyrus.spliced_data.gz.dim50_k10_reg500_n4t12_pca_no_rotation_log_scale_transpose.drgraph
```

## Graphical Interface

For an example, run
` python ./trenti/app.py -i ./example/Dentate_Gyrus.data_pca.gz   -f ./example/Dentate_Gyrus.spliced_data.gz -a ./example/Dentate_Gyrus.anno.gz  --samplelimit=5000 --log --mode graphdr`

See `./trenti/README.md` for a further example.
