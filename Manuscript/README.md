This directory contains code for reproducing all analyses in the manuscript.

To run the analyses, first download the required data by  
`sh download_data.sh`.

There are additional dependencies you will need to install for running some of the experiments.

python:
- scanpy
- bokeh

R:
- tidyverse
- data.table
- ggplot2
- Matrix
- igraph
- useful
- FNN
- dyneval (https://github.com/dynverse/dyneval)
- dynwrap (https://github.com/dynverse/dynwrap)


Most analyses are implemented in the format of jupyter notebooks with python and R backends. For rerunning benchmarks you will need to first run a series of bash scripts following the numerical order indicated in the file names. Note that some of the scripts contains a large number of commands that can be parallelized, so you can consider running them in parallel or submit to computer cluster. You may also load our precomputed benchmark results in the jupyter notebook.

Here is a quick index of jupyter notebooks:

Figure 1: 
- GraphDR_examples.ipynb
- GraphDR benchmark.ipynb	
- GraphDR_planarian_comparison.ipynb
- GraphDR_planarian_comparison_plot.ipynb

Figure 2:
- StructDR_example.ipynb
- StructDR_hippocampus.ipynb
- StructDR_hippocampus_plot.ipynb
- StructDR_performance.ipynb

Figure 3:
- Schematic figure made from screenshot of `trenti/app.py`

Supplementary Figure 1:
- GraphDR_planarian_comparison.ipynb
- GraphDR_planarian_comparison_plot.ipynb

Supplementary Figure 2:
- Schematic figure

Supplementary Figure 3:
- GraphDR_complexdesign_zebrafish.ipynb
- GraphDR_complexdesign_zebrafish_plot.ipynb

Supplementary Figure 4:
- GraphDR_complexdesign_xenopus.ipynb
- GraphDR_complexdesign_xenopus_plot.ipynb

Supplementary Figure 5:
- StructDR_performance.ipynb

Supplementary Figure 6:
- StructDR_hippocampus.ipynb
- StructDR_hippocampus_plot.ipynb

Supplementary Figure 7:
- StructDR_CI_simulation.ipynb


