
Graphical interface

## TRENTI

TRajectory Estimation NavigaTion Interface (TRENTI) is a
graphical interface for single cell data visualization and exploratory analysis. It starts an local webserver that can be accessed through your browser.

### Basic Usage

To run, cd to the directory that contains the source, run
`python app.py -i Dentate_Gyrus.data_pca   -f Dentate_Gyrus.spliced_data -a Dentate_Gyrus.anno -v Dentate_Gyrus.data_pca_v --samplelimit=5000 --log--mode graphdr`

In your browser go to localhost:8050. 



### Options
* `-f`: Feature file.   Gene x Sample / Cell expression matrix. Expect log-normalized values by default. Use `--log` argument if the input is not log-transformed.
* `-i`: Input file. Sample/Cell x Dims matrix with headers. Dims can be genes, PCs or embedding dimensions. This file will be used for trajectory / clustering / projection analyses, and is typically a PCA transformed matrix, or scaled log expression matrix. If this file not provided we will compute from feature file - however it is highly recommended to provided input file that has been preprocessed (e.g. select highly variable genes, removing batch effect, cell cycle effect, standardize etc.).
* `-a`: Annotation file. Annotations for visualization.
* `-v`: Velocity file (experimental - work in progress).  Velocity matrix with the same dimensions as input file.
* `--SAMPLELIMIT=`: This can be used to load fewer cells from your sample to save some computation time.
* `--log`: If the feature file is npt .
Note for all input files, you can append '.T' to transpose the matrix after loading.


### Brief instructions
Most functionalities are probably self-explanatory and so I just try to mention some more high level or hidden poinys. Note that most of the functions are in the 'Advanced options' part that are hidden by default.

**Basic visualization**

Display gene expression:

Click on a gene on the right, or use the dropdown below or type gene name. Now the gene plot it self reflect expressions in the cells selected in the cell selectors above (default are all cells).

To show annotations overlay:

Visualization -> Check "Annotation" -> Select column display

Show K-nearest neighbor graph:

Visualization -> Check one of the KNN options


**Projection**

Dimensionality reductions. PCA or DRGraph (manuscript in preparation) are recommended for trajectory analysis.

**Trajectory**

Select number of interations and click `run` button to run. Important parameters are bandwidth / adaptive bandwidth.

**Clustering**
Note clustering can be ran on both the original data (default), or the trajectory generated.

**Bootstrap**
Bootstrapped trajectories allow estimating uncertainties of trajectory estimations.

**Network**
This shows local covariation of expression (only supports PCA projection mode now). Still in development too.

## Tips:

To save as image for presentation or publication:
Click on the 'Save' button on toolbar and it will bring up a windows that allows detailed adjustment to the figure. 

Figure not large enough: You can check the "Fullscreen" checkbox to display it on the full page.
