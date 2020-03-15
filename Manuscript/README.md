This directory contains code for reproducing all analyses in the manuscript.

To run the analyses, first download the required data by  
`sh download_data.sh`.

There are additional dependencies you will need for running some of the experiments.

python:
scanpy
bokeh

R:
tidyverse
data.table
ggplot2
Matrix
igraph
useful
FNN
dyneval (https://github.com/dynverse/dyneval)
dynwrap (https://github.com/dynverse/dynwrap)

Most of the experiments are implemented in jupyter notebooks. Each notebook reproduces a different experiment. For rerunning benchmarks you will need to first run each bash script following the numerical order indicated in the file names. Note that some of the scripts contains a large number of commands that can be parallelized, so you can consider running them in parallel or submit to computer cluster. You may also load our precomputed benchmark results in the jupyter notebook.