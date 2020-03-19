


## 3D Interactive Visualization and Exploration Interface: Trenti

TRajectory Exploration and NavigaTion Interface (Trenti) is a
graphical interface for single cell data visualization and exploratory analysis. It starts an local webserver that can be accessed through your browser.

### Basic Usage

To run, cd to the directory that contains the source, run
`python ./app.py -i ../example/Dentate_Gyrus.data_pca.gz -f ../example/Dentate_Gyrus.spliced_data.gz -a ../example/Dentate_Gyrus.anno.gz --samplelimit=5000 --log --mode graphdr`

In your browser go to localhost:8050. 



### Options
* `-f`: **Feature file**.   Gene x Sample / Cell expression matrix. Expect log-normalized values by default. Use `--log` argument if the input is not log-transformed.
* `-i`: **Input file**. Sample/Cell x Dims matrix with headers. Dims can be genes, PCs or embedding dimensions. This file will be used for trajectory / clustering / projection analyses, and is typically a PCA transformed matrix, or scaled log expression matrix. If this file not provided we will compute from feature file - however it is highly recommended to provided input file that has been preprocessed (e.g. select highly variable genes, removing batch effect, cell cycle effect, standardize etc.).
* `-a`: **Annotation file**. Annotations for visualization.
* `-v`: **Velocity file (experimental - work in progress)**.  Velocity matrix with the same dimensions as input file.
* `--SAMPLELIMIT=`: Integer. This can be used to load fewer cells from your sample to save some computation time.
* `--log`: If specified, the feature file will transformed by log(1+count).
* `--mode`: Should be one of 'graphdr', 'pca', or 'none', specify the transformation to apply to the input file matrix. If 'none', no transformation is applied.

Tip: For all input files, you can append `.T` to the filename, so Trenti will transpose the matrix after loading.


### Brief instructions
The interface is divided into two parts. The left panel provides the main visualization and method interfaces, it displays the data in 3D, provides various analysis options and you can click on buttons to upload your files (loading the files through command-line is recommended and better supported). The right panels are interactive selective interfaces for selecting genes and cells while displaying useful information at gene or cell levels. There are a lot more functionalities that can be accessed through 'Advanced options' that are hidden by default.

**Basic visualization**







**Bootstrap**
Bootstrapped trajectories allow estimating uncertainties of trajectory estimations.

**Network**
This shows local covariation of expression (only supports PCA projection mode now). Still in development too.

## Tips:

Basic visualization:
* **Gene expression explorer**: Click on a gene in the *gene selector interface* (right bottom), or use the dropdown below or type gene name. The *gene selector interface* display useful information about each gene to help the selection. The *gene selector interface* is connected to the *cell selector interface*. When cells are selected in the *cell selector interace*, the *gene selector interface* will be updated to reflect expression values of the cells selected (default are all cells). There are two modes of *gene selector interface*, you can choose between displaying "mean vs. SD" or "mean vs. diff between selected cells and other cells".
* **Overlay custom annotations with cells**: Visualization -> Check "Annotation" -> Select column to display. You can choose between treating a column as categorical variable or numerical variable. Default is auto.
* **K-nearest neighbor graph**: Visualization -> Check one of the KNN options
* To prepare figure for presentation or publication:
Click on the 'Save' button on toolbar and it will bring up a windows that allows detailed adjustment to the figure. 
* You can check the "Fullscreen" checkbox to display it on the full page.

Methods interface: 
* **Projection (Dimensionality Reduction)**:
Access to data representation methods including GraphDR. PCA or GraphDR are recommended for StructDR density ridge / trajectory analysis.

* **Algorithm (StructDR - Nonparametric Density Ridge estimation)**:
Select number of iteration and click `run` button to run. Important parameters are bandwidth / adaptive bandwidth.

* **Clustering**:
Access to multiple common clustering methods. Clustering can be ran on both the original data (default), or the projected data displayed. 



