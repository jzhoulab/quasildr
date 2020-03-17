# -*- coding: utf-8 -*-
"""
Usage:
    app.py [-i INPUT_FILE] [-f FEATURE_FILE] [-a ANNOTATION_FILE] [-v VELOCITY_FILE]  [-m PROJECTION_MODE]  [-n NETWORK_DATA] [--samplelimit=<n>] [--log]  [--port=<n>]
    app.py -h | --help

Options:
    -h --help     Show this screen.
    -i INPUT_FILE, --input=INPUT_FILE                   input file
    -f FEATURE_FILE, --feature=FEATURE_FILE             feature file
    -a ANNOTATION_FILE, --annotation=ANNOTATION_FILE    annotation file
    -v VELOCITY_FILE, --velocity=VELOCITY_FILE          velocity file (same dimensions as input)
    -m PROJECTION_MODE, --mode=PROJECTION_MODE          default projection mode (pca, graphdr, or none) [default: graphdr]
    -n NETWORK_DATA, --networkdata=NETWORK_DATA         network data (feature or input) [default: feature]
    --samplelimit=<n>                                   sample size limit [default: 100000]
    --port=<n>                                          port [default: 8050]
    --log                                               apply log transform to feature file

"""
##TODO:  Loom, DataPool

import base64
import io
import re
import sys
import time
import zipfile
from functools import reduce

import dash
import dash_colorscales
import dash_core_components as dcc
import dash_html_components as html
import multiprocess
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import umap
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from docopt import docopt
from plotly.colors import DEFAULT_PLOTLY_COLORS
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import quasildr.structdr as scms2
from quasildr import utils
from quasildr.graphdr import *


def match(x, y):
    ydict = {}
    for i, yy in enumerate(y):
        ydict[yy] = i
    inds = []
    for xx in x:
        if xx in ydict:
            inds.append(ydict[xx])
        else:
            inds.append(-1)
    return np.array(inds)


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")

    SAMPLELIMIT = int(arguments['--samplelimit'])
    MAX_PCS = 100
    DEFAULT_PCS = 30
    DEFAULT_DR_K = 10
    DEFAULT_DR_LAMBDA = 10
    COLORPATTERN = re.compile("^#[0-9,A-F,a-f][0-9,A-F,a-f][0-9,A-F,a-f][0-9,A-F,a-f][0-9,A-F,a-f][0-9,A-F,a-f]$")
    ITERATIONS = [1, 5, 10, 20, 40, 80, 160]
    BATCHSIZE = 500
    BINS = [0.0, 0.06666666666666667, 0.13333333333333333, 0.2, 0.26666666666666666,
            0.3333333333333333, 0.4, 0.4666666666666667, 0.5333333333333333, 0.6, 0.6666666666666666,
            0.7333333333333333, 0.8, 0.8666666666666667, 0.9333333333333333, 1.0]

    SYMBOLS = ["circle", "cross", "square", "diamond", "circle-open", "square-open", "diamond-open"]
    DEFAULT_COLORSCALE = ['#440154', '#471867', '#472a79', '#413d84', '#3a4e8c', '#2f5e8f',
                          '#296d90', '#1f7c91', '#1b8a90', '#16988d', '#21af83', '#5bc865',
                          '#89d54a', '#b1dd2f', '#d8e324', '#fee825']

    DEFAULT_OPACITY = 0.8
    CELLCOLOR = '#ff6138'
    FEATURECOLOR = '#00a388'
    BGCOLOR = '#FFFFFF'  ##FAFBFC'

    message = []

    # configure data
    if arguments['--input'] is not None:
        try:
            if arguments['--input'].endswith('.T'):
                input_data = pd.read_csv(arguments['--input'][:-2], delimiter='\t', index_col=0).T
            else:
                input_data = pd.read_csv(arguments['--input'], delimiter='\t', nrows=SAMPLELIMIT + 1, index_col=0)
            input_data = input_data.iloc[:SAMPLELIMIT, :]
            if input_data.shape[1] <= 3:
                input_data['z'] = 0
            #input_data_sd = np.std(input_data.values, axis=0)
            #input_data = input_data.iloc[:, np.argsort(-input_data_sd)]
            with_user_input_data = True
        except Exception as e:
            print(e)
            with_user_input_data = False
            message.append("Warning: cannot read input data.")
    else:
        with_user_input_data = False

    if arguments['--feature'] is not None:
        try:
            if arguments['--feature'].endswith('.T'):
                feature_data = pd.read_csv(arguments['--feature'][:-2], delimiter='\t', nrows=SAMPLELIMIT + 1,
                                           index_col=0).T
            else:
                feature_data = pd.read_csv(arguments['--feature'], delimiter='\t', index_col=0)
            feature_data = feature_data.iloc[:, :SAMPLELIMIT]
            if arguments['--log']:
                feature_data = np.log(feature_data + 1)
            feature_data_sd = np.std(feature_data.values, axis=1)
            feature_data = feature_data.iloc[np.argsort(-feature_data_sd), :]
            with_feature_data = True
        except Exception as e:
            print(e)
            with_feature_data = False
            message.append("Warning: feature data not loaded. Feature related functions disabled.")
    else:
        with_feature_data = False

    if not with_feature_data and not with_user_input_data:
        sys.exit("Each feature file or input file need to be readable.")

    if arguments['--velocity'] is not None:
        try:
            if arguments['--velocity'].endswith('.T'):
                velocity_input_data = pd.read_csv(arguments['--velocity'][:-2], delimiter='\t', index_col=0).T
            else:
                velocity_input_data = pd.read_csv(arguments['--velocity'], delimiter='\t', nrows=SAMPLELIMIT + 1,
                                                  index_col=0)
            velocity_input_data = velocity_input_data.iloc[:SAMPLELIMIT, :]
            with_velocity_input_data = True
        except Exception as e:
            print(e)
            with_velocity_input_data = False
            message.append("Warning: cannot read velocity data.")
    else:
        with_velocity_input_data = False

    # Prepare input data
    if with_feature_data and not with_user_input_data:
        input_data = feature_data.T
        with_user_input_data = False
    else:
        with_user_input_data = True

    if with_velocity_input_data:
        if np.any(input_data.shape != velocity_input_data.shape) or np.any(
                input_data.index != velocity_input_data.index):
            with_velocity_input_data = False
            message.append('Warning: Velocity data does not match input data.')

    N_PCs = np.minimum(MAX_PCS, np.minimum(input_data.shape[0], input_data.shape[1]))
    if arguments['--mode'] == 'none':
        data = input_data.copy()
        projection_mode = 'none'
        if with_velocity_input_data:
            velocity_data = velocity_input_data.copy()
            with_velocity_data = True
        else:
            with_velocity_data = False
    else:
        input_data_pca = PCA(N_PCs)
        data = pd.DataFrame(input_data_pca.fit_transform(input_data.values), index=input_data.index,
                            columns=['PC' + str(i) for i in range(1, N_PCs + 1)])
        if with_velocity_input_data:
            velocity_data = pd.DataFrame(input_data_pca.transform(velocity_input_data.values),
                                         index=velocity_input_data.index,
                                         columns=['PC' + str(i) for i in range(1, N_PCs + 1)])
            with_velocity_data = True
        else:
            with_velocity_data = False

        if arguments['--mode'] == 'pca':
            projection_mode = 'pca'
        elif arguments['--mode'] == 'graphdr':
            mapped = graphdr(data.values[:, :DEFAULT_PCS], n_neighbors=DEFAULT_DR_K, regularization=DEFAULT_DR_LAMBDA)
            data = pd.DataFrame(mapped, index=data.index,
                                columns=['GraphDR' + str(i) for i in range(1, mapped.shape[1] + 1)])
            projection_mode = 'graphdr'
        else:
            raise ValueError('Default mode has to be either pca or graphdr')

    if with_velocity_data:
        velocity_data = velocity_data / np.std(data.iloc[:, 0])

    data = (data - np.mean(data, axis=0)) / np.std(data.iloc[:, 0])

    if with_user_input_data:
        if len(np.intersect1d(feature_data.columns, data.index)) != len(data.index):
            with_feature_data = False
            print(feature_data.columns)
            print(data.index)
            message.append("Warning: feature data column names does not match with input data row names.")
    else:
        assert len(np.intersect1d(feature_data.columns, data.index)) == len(data.index)

    if arguments['--networkdata'] == 'feature' and with_feature_data:
        network_data = feature_data
        with_network_data = True
    elif arguments['--networkdata'] == 'input':
        network_data = input_data.T
        with_network_data = True
    else:
        with_network_data = False
        message.append("Warning: --networkdata has to be either \"feature\" with -f option specified or \"input\".")

    if with_network_data:
        network_data_pca = PCA(N_PCs)
        network_data_pca_z = network_data_pca.fit_transform(network_data.values.T)

    if arguments['--annotation'] is not None:
        try:
            # set low memory to false to avoid mixed types
            if arguments['--annotation'].endswith('.T'):
                annotation_data = pd.read_csv(arguments['--annotation'][:-2], delimiter='\t', low_memory=False,
                                              index_col=0).T
            else:
                annotation_data = pd.read_csv(arguments['--annotation'], delimiter='\t', low_memory=False,
                                              nrows=SAMPLELIMIT + 1, index_col=0)
            annotation_data = annotation_data.iloc[:SAMPLELIMIT, :]
            with_annotation_data = True
        except Exception as e:
            print(e)
            with_annotation_data = False
            message.append("Warning: cannot read annotation data.")

        if with_annotation_data:
            try:
                assert np.all(annotation_data.index == data.index)
            except:
                with_annotation_data = False
                message.append("Warning: annotation data row names does not match with input data row names.")
    else:
        with_annotation_data = False

    if not with_annotation_data:
        annotation_data = data.iloc[:, :0].copy()

    with_trajectory_data = False

    # initialize
    ndim = 6
    history = []
    s = scms2.Scms(np.asarray(data.iloc[:, :ndim]).copy(), 0)
    traj = data.iloc[:, :ndim].copy()
    history.append(traj.copy())
    output_dict = {'index': traj.index.values}

    app = dash.Dash(__name__)
    server = app.server

    '''
    ~~~~~~~~~~~~~~~~
    ~~ APP LAYOUT ~~
    ~~~~~~~~~~~~~~~~
    '''

    app.layout = html.Div(children=[
        html.Div(id='notification'),
        html.Div(id='all-plots', children=[
            html.H3(children='TRENTI',
                    style={'color': '#1f1f27', 'font-size': '1.5vw', 'margin-left': '1.1%', 'margin-top': '0.5rem',
                           'margin-bottom': '0.5rem'}),
            html.Hr(style={'margin': '1rem 53% 1.5rem 1.1%'}),
            html.Div(id='pane_left', children=[
                html.Div(children=[
                    html.P('Configure files:'),
                    html.Div(className='row', children=[
                        html.Div(children=[
                            dcc.Upload(
                                id='upload_feature',
                                children=html.Div(id='upload_feature_label',
                                                  children=['Feature ' + (u" \u2713" if with_feature_data else "")]),
                                style={
                                    'lineHeight': '3rem',
                                    'borderWidth': '0.1rem',
                                    'borderStyle': 'solid' if with_feature_data else 'dashed',
                                    'borderRadius': '0.5rem',
                                    'textAlign': 'center',
                                }
                            )],
                            style={
                                'margin-left': '0.8%',
                                'margin-right': '0.8%',
                                'font-size': '0.75vw',
                                'width': '17%',
                                'height': '3rem',
                                'display': 'inline-block',
                                'text-overflow': 'clip',
                            }),
                        html.Div(children=[
                            dcc.Upload(
                                id='upload',
                                children=html.Div(id='upload_label', children=[
                                    'Input (optional)' + (u" \u2713" if with_user_input_data else "")]),
                                style={
                                    'lineHeight': '3rem',
                                    'borderWidth': '0.1rem',
                                    'borderStyle': 'solid' if with_user_input_data else 'dashed',
                                    'borderRadius': '0.5rem',
                                    'textAlign': 'center',
                                }
                            )],
                            style={
                                'margin-left': '0.8%',
                                'margin-right': '0.8%',
                                'font-size': '0.75vw',
                                'width': '17%',
                                'height': '3rem',
                                'display': 'inline-block',
                                'text-overflow': 'clip',
                            }),
                        html.Div(children=[
                            dcc.Upload(
                                id='upload_annotation',
                                children=html.Div(id='upload_annotation_label', children=[
                                    'Annotation (optional)' + (u" \u2713" if with_annotation_data else "")]),
                                style={
                                    'lineHeight': '3rem',
                                    'borderWidth': '0.1rem',
                                    'borderStyle': 'solid' if with_annotation_data else 'dashed',
                                    'borderRadius': '0.5rem',
                                    'textAlign': 'center',
                                }
                            )],
                            style={
                                'margin-left': '0.8%',
                                'margin-right': '0.8%',
                                'font-size': '0.75vw',
                                'width': '20%',
                                'height': '3rem',
                                'display': 'inline-block',
                                'text-overflow': 'clip',
                            }),
                        html.Div(children=[
                            dcc.Upload(
                                id='upload_velocity',
                                children=html.Div(id='upload_velocity_label', children=[
                                    'Velocity (optional)' + (u" \u2713" if with_velocity_input_data else "")]),
                                style={
                                    'lineHeight': '3rem',
                                    'borderWidth': '0.1rem',
                                    'borderStyle': 'solid' if with_velocity_input_data else 'dashed',
                                    'borderRadius': '0.5rem',
                                    'textAlign': 'center',
                                }
                            )],
                            style={
                                'margin-left': '0.8%',
                                'width': '18%',
                                'height': '3rem',
                                'font-size': '0.75vw',
                                'display': 'inline-block',
                                'text-overflow': 'clip',
                            }),
                        html.Div(children=[
                            dcc.Upload(
                                id='upload_trajectory',
                                children=html.Div(id='upload_trajectory_label', children=[
                                    'Trajectory (optional)' + (u" \u2713" if with_trajectory_data else "")]),
                                style={
                                    'lineHeight': '3rem',
                                    'borderWidth': '0.1rem',
                                    'borderStyle': 'solid' if with_trajectory_data else 'dashed',
                                    'borderRadius': '0.5rem',
                                    'textAlign': 'center',
                                }
                            )],
                            style={
                                'margin-left': '0.8%',
                                'width': '18%',
                                'height': '3rem',
                                'font-size': '0.75vw',
                                'display': 'inline-block',
                                'text-overflow': 'clip',
                            }), ], style={'margin': '2% 2% 3% 0%'}),
                    html.P('Drag the slider to select the number of SCMS steps:'),
                    html.Div(className='row', children=[
                        html.Div([
                            dcc.Slider(
                                id='ITERATIONS-slider',
                                min=min(ITERATIONS),
                                max=max(ITERATIONS),
                                value=min(ITERATIONS),
                                step=None,
                                marks={str(n): str(n) for n in ITERATIONS},
                            ),
                        ], style={'width': '42%', 'display': 'inline-block', 'margin-right': '2%',
                                  'margin-top': '0.5rem', 'margin-bottom': '0.5rem'}),
                        html.Div([
                            html.Button('Run', id='run-button', style={'width': '100%'})
                        ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '12%'}),
                        html.Div([
                            html.Button('Reset', id='reset-button', style={'width': '100%'})
                        ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '12%'}),
                        html.Div([
                            html.Button('Bootstrap', id='bootstrap-button', style={'width': '100%'})
                        ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '12%'}),
                        html.Div([
                            html.Button('Save', id='save-button', style={'width': '100%'})
                        ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '12%'}),
                    ], style={'margin': '2%'}),
                    html.Br(),
                    html.Div(className='row', children=[
                        html.P('Dot size:',
                               style={
                                   'display': 'inline-block',
                                   'position': 'absolute',
                               }
                               ),
                        html.Div([
                            dcc.Slider(
                                id='dotsize-slider',
                                min=0,
                                max=10,
                                value=np.maximum(6 - np.log10(data.shape[0]), 0),
                                step=0.01,
                                marks={i: str(i) for i in range(1, 11)},
                            )
                        ], style={'width': '40.5%', 'display': 'inline-block', 'margin-left': '2%',
                                  'marginBottom': '1rem', 'margin-top': '2.5rem'}),

                        html.Div([
                            dash_colorscales.DashColorscales(
                                id='colorscale-picker',
                                colorscale=DEFAULT_COLORSCALE,
                                nSwatches=16,
                                fixSwatches=True
                            )
                        ], style={'display': 'inline-block'}),
                        html.Div([
                            html.P('Advanced options:',
                                   style={
                                       'verticalAlign': 'top',
                                   }
                                   ),
                            html.Div([
                                dcc.RadioItems(
                                    options=[
                                        {'label': 'Algorithm',
                                         'value': 'show_alg_options'},
                                        {'label': 'Visualization',
                                         'value': 'show_disp_options'},
                                        {'label': 'Projection',
                                         'value': 'show_embedding_options',
                                         'disabled': False if with_feature_data else True},
                                        {'label': 'Clustering',
                                         'value': 'show_cluster_options',
                                         'disabled': False if with_feature_data else True},
                                        {'label': 'Network',
                                         'value': 'show_network_options',
                                         'disabled': not with_network_data},
                                        {'label': 'None',
                                         'value': 'show_no_options'}
                                    ],
                                    labelStyle={'display': 'inline-block', 'margin-right': '0.3vw'},
                                    id='show-options',
                                    value='show_no_options',
                                )], style={'display': 'inline-block'}),
                        ], style={'display': 'inline-block', 'width': '27%'}),
                    ]),
                ], style={'margin': '0 2.2% 2.2% 2.2%'}),
                html.Div(
                    className="row",
                    children=[
                        html.Div(id="alg-options",
                                 className="three columns",
                                 children=[
                                     html.Label('Density Ridge Type'),
                                     dcc.Dropdown(
                                         id='dimensionality_dropdown',
                                         options=[
                                             {'label': '0 (Cluster)', 'value': 0},
                                             {'label': '1 (Trajectory)', 'value': 1},
                                             {'label': '2 (Surface)', 'value': 2}
                                         ],
                                         value=1,
                                         clearable=False,
                                     ),
                                     html.Label('Input Dim.'),
                                     dcc.Dropdown(
                                         id='ndim_dropdown',
                                         options=[{'label': str(i), 'value': i}
                                                  for i in range(2, data.shape[1] + 1)
                                                  ],
                                         value=6,
                                         clearable=False,
                                     ),
                                     html.Label('Bandwidth'),
                                     dcc.Dropdown(
                                         id='bandwidth_dropdown',
                                         options=[
                                             {'label': '0 (Adaptive bandwidth)' if i == 0 else '{: .2f}'.format(i),
                                              'value': i}
                                             for i in np.linspace(0, 5, 101)
                                         ],
                                         value=0.3,
                                         clearable=False,
                                     ),
                                     html.Label('Adpative Bandwidth'),
                                     html.Label('(kth-neighbors)'),
                                     dcc.Dropdown(
                                         id='min_radius_dropdown',
                                         options=[
                                             {'label': '0 (Uniform bandwidth)' if i == 0 else str(i), 'value': i}
                                             for i in range(0, 201)
                                         ],
                                         value=10,
                                         clearable=False,
                                     ),
                                     html.Label('Stepsize'),
                                     dcc.Dropdown(
                                         id='stepsize_dropdown',
                                         options=[
                                             {'label': '{: .2f}'.format(i), 'value': i}
                                             for i in np.linspace(0.05, 1, 20)
                                         ],
                                         value=1.0,
                                         clearable=False,
                                     ),
                                     html.Label('Relaxation'),
                                     dcc.Dropdown(
                                         id='relaxation_dropdown',
                                         options=[
                                             {'label': '{: .1f}'.format(i), 'value': i}
                                             for i in np.linspace(0, 4, 41)
                                         ],
                                         value=0,
                                         clearable=False,
                                     ),
                                     html.Label('Threads'),
                                     dcc.Dropdown(
                                         id='njobs_dropdown',
                                         options=[
                                             {'label': str(i), 'value': i}
                                             for i in range(1, multiprocess.cpu_count() + 1)
                                         ],
                                         value=1 if SAMPLELIMIT < 1000 else multiprocess.cpu_count() / 2,
                                         clearable=False,
                                     ),
                                     html.Label('Method'),
                                     dcc.RadioItems(
                                         id='method_checkbox',
                                         options=[
                                             {'label': 'MSLogP', 'value': 'MSLogP'},
                                             {'label': 'MSP', 'value': 'MSP'},
                                         ],
                                         value='MSLogP',
                                     ),
                                     html.Div([
                                         html.Button('Subsampling to:', id='subsample_button', style={'width': '100%'})
                                     ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '100%',
                                               'margin-top': '1rem'}),
                                     dcc.Dropdown(
                                         id='subsample_dropdown',
                                         options=[
                                             {'label': str(i * 100), 'value': i * 100}
                                             for i in range(1, 101) if i * 100 < data.shape[0]
                                         ],
                                         value=2000 if data.shape[0] >= 2000 else data.shape[0],
                                         clearable=False,
                                     ),
                                 ], style={'padding': '1rem 2.2% 0rem 2.2%', 'margin-left': 0, 'display': 'none'}),
                        html.Div(id="disp-options",
                                 className="three columns",
                                 children=[
                                     html.Div([
                                         html.Label('Opacity'),
                                         dcc.Slider(
                                             id='opacity-slider',
                                             min=0, max=1, value=DEFAULT_OPACITY, step=0.1,
                                             marks={0: '0', 0.5: '0.5', 1: '1'},
                                         ), ], style={'margin-bottom': '2.5rem'}),
                                     html.Div([
                                         html.Label('Smoothing radius'),
                                         dcc.Slider(
                                             id='smoothing-slider',
                                             min=0.,
                                             max=1.,
                                             value=0.,
                                             step=0.01,
                                             marks={0: '0', 0.5: '0.5', 1: '1'},
                                         )], style={'margin-bottom': '2.5rem'}),
                                     html.Div([
                                         html.Label('Velocity arrow size'),
                                         dcc.Slider(
                                             id='conesize-slider',
                                             min=-1.,
                                             max=3.,
                                             value=0.5,
                                             step=0.1,
                                             marks={-1: '0.1', 0: '1', 1: '10', 2: '100', 3: '1000'},
                                         )], style={'margin-bottom': '2.5rem'}),
                                     html.Div(className='row', children=[
                                         html.Label('3D plot dimensions'),
                                         html.Div([
                                             dcc.Dropdown(
                                                 id='x_dropdown',
                                                 options=[
                                                     {'label': str(i + 1) if i != -1 else '', 'value': i}
                                                     for i in range(-1, 6)
                                                 ],
                                                 value=0,
                                                 clearable=False,
                                             )], style={'display': 'inline-block', 'width': '33%'}),
                                         html.Div([
                                             dcc.Dropdown(
                                                 id='y_dropdown',
                                                 options=[
                                                     {'label': str(i + 1) if i != -1 else '', 'value': i}
                                                     for i in range(-1, 6)
                                                 ],
                                                 value=1,
                                                 clearable=False,
                                             )], style={'display': 'inline-block', 'width': '33%'}),
                                         html.Div([
                                             dcc.Dropdown(
                                                 id='z_dropdown',
                                                 options=[
                                                     {'label': str(i + 1) if i != -1 else '', 'value': i}
                                                     for i in range(-1, 6)
                                                 ],
                                                 value=2 if traj.shape[1] > 2 else -1,
                                                 clearable=False,
                                             )], style={'display': 'inline-block', 'width': '33%'}),
                                     ]),
                                     html.Div([
                                         html.Label('Aspect ratio:', style={'margin-top': '1rem'}),
                                         dcc.RadioItems(
                                             id='scatter3d_aspect_options',
                                             options=[
                                                 {'label': 'Fixed', 'value': 'data'},
                                                 {'label': 'Auto', 'value': 'auto'},
                                             ],
                                             value='auto',
                                             labelStyle={'display': 'inline-block', 'margin-right': '0.3vw'},
                                         ),
                                         html.Label('Display / Compute:', style={'margin-top': '1rem'}),
                                         dcc.Checklist(
                                             options=[
                                                 {'label': 'Colorbar ',
                                                  'value': 'show_legend'},
                                                 {'label': 'Selected Cells',
                                                  'value': 'show_selected'},
                                                 {'label': 'Original Data',
                                                  'value': 'show_original'},
                                                 {'label': 'Projection Paths',
                                                  'value': 'show_traces'},
                                                 {'label': 'Log Density',
                                                  'value': 'show_logp'},
                                                 {'label': 'KNN Graph (Input)',
                                                  'value': 'show_knn'},
                                                 {'label': 'KNN Graph (Traj.)',
                                                  'value': 'show_knn_traj'},
                                                 {'label': 'MST',
                                                  'value': 'show_mst'},
                                                 {'label': '↳ Segment',
                                                  'value': 'show_segments'},
                                                 {'label': '↳ ↳ Cell order',
                                                  'value': 'show_order'},
                                                 {'label': 'Velocity (if avai.)',
                                                  'value': 'show_velocity',
                                                  'disabled': not with_velocity_data},
                                                 {'label': 'Bootstrap (if avai.)',
                                                  'value': 'show_bootstrap'},
                                                 {'label': 'Annotation',
                                                  'value': 'show_annotation',
                                                  'disabled': annotation_data.shape[1] == 0}, ],
                                             value=['show_legend', 'show_selected', 'show_velocity', 'show_bootstrap'],
                                             labelStyle={},
                                             id='display-checklist',
                                         ),
                                     ], style={}),
                                     html.Div(id='annotation_dropdown_div', children=[
                                         dcc.Dropdown(
                                             id='annotation_dropdown',
                                             options=[

                                             ],
                                             value=0,
                                             clearable=False, ),
                                         html.Label('Annotation type', style={'margin-top': '1rem'}),
                                         dcc.RadioItems(
                                             id='annotation_type',
                                             options=[
                                                 {'label': 'Auto', 'value': 'auto'},
                                                 {'label': 'Numerical', 'value': 'numerical'},
                                                 {'label': 'Categorical', 'value': 'categorical'},
                                                 {'label': 'None', 'value': 'none'},
                                             ],
                                             value='auto',
                                             labelStyle={'display': 'inline-block', 'margin-right': '0.3vw'},
                                         ),
                                         dcc.Checklist(
                                             options=[
                                                 {'label': 'Label ',
                                                  'value': 'show_label'}],
                                             value=['show_label'],
                                             labelStyle={},
                                             id='label_checklist',
                                         )
                                     ], style={'display': 'block' if with_annotation_data else 'none'}),
                                 ], style={'padding': '1rem 2.2% 0rem 2.2%', 'margin-left': 0, 'display': 'none'}),
                        html.Div(id="network-options",
                                 className="three columns",
                                 children=[
                                     html.Div([
                                         html.Label('Hover over a cell to display the local network, click to cluster.',
                                                    style={'margin-top': '1rem'}),
                                         html.Label('Bandwidth'),
                                         dcc.Dropdown(
                                             id='network_bandwidth_dropdown',
                                             options=[
                                                 {'label': '0 (Adaptive bandwidth)' if i == 0 else '{: .2f}'.format(i),
                                                  'value': i}
                                                 for i in np.linspace(0, 5, 101)
                                             ],
                                             value=0.2,
                                             clearable=False,
                                         ),
                                         html.Label('Adpative Bandwidth'),
                                         html.Label('(kth-neighbors)'),
                                         dcc.Dropdown(
                                             id='network_min_radius_dropdown',
                                             options=[
                                                 {'label': '0 (Uniform bandwidth)' if i == 0 else str(i), 'value': i}
                                                 for i in range(0, 201)
                                             ],
                                             value=0,
                                             clearable=False,
                                         ),
                                         html.Label('N PCs'),
                                         dcc.Dropdown(
                                             id='network_n_pcs',
                                             options=[
                                                 {'label': '0 (All dimensions)' if i == 0 else str(i), 'value': i}
                                                 for i in range(0, MAX_PCS + 1)
                                             ],
                                             value=MAX_PCS,
                                             clearable=False,
                                         ),
                                         html.Label('Display:', style={'margin-top': '1rem'}),
                                         dcc.Checklist(
                                             options=[
                                                 {'label': 'Colorbar ',
                                                  'value': 'show_legend'},
                                                 {'label': 'Values ',
                                                  'value': 'show_values'},
                                                 {'label': 'Diagnonal',
                                                  'value': 'show_diagonal'}],
                                             value=['show_diagonal', 'show_values'],
                                             labelStyle={},
                                             id='heatmap_checklist',
                                         ),
                                         html.Label('Network type:', style={'margin-top': '1rem'}),
                                         dcc.RadioItems(
                                             id='heatmap_precision_options',
                                             options=[
                                                 {'label': 'Local precision', 'value': 'show_precision'},
                                                 {'label': 'Local covariance', 'value': 'show_covariance'},
                                             ],
                                             value='show_covariance'),
                                         html.Label('Local neighborhood space:', style={'margin-top': '1rem'}),
                                         dcc.RadioItems(
                                             id='heatmap_reference_options',
                                             options=[
                                                 {'label': 'Original', 'value': 'cell'},
                                                 {'label': 'Trajectory', 'value': 'trajectory'},
                                             ],
                                             value='trajectory'
                                         ),
                                         # html.Label('Max PCs to display:',style={'margin-top':'1rem'}),
                                         # dcc.Dropdown(
                                         #     id='heatmap_dim_dropdown',
                                         #     options=[
                                         #         {'label': str(i), 'value': i}
                                         #     for i in range(1,500+1)
                                         #     ],
                                         #     value=20,
                                         #     clearable=False,
                                         # ),
                                         html.Div([
                                             html.Button('Reset node order', id='reset-heatmap-order-button',
                                                         style={'width': '100%'})
                                         ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '100%',
                                                   'margin-top': '1rem'}),

                                     ], style={}),

                                 ], style={'padding': '1rem 2.2% 0rem 2.2%', 'margin-left': 0, 'display': 'none'}),

                        html.Div(id="embedding-options",
                                 className="three columns",
                                 children=[
                                     html.Label('Pre-processing:', style={'margin-top': '1rem'}),
                                     dcc.RadioItems(
                                         id='dr_method',
                                         options=[
                                             {'label': 'PCA',
                                              'value': 'pca'},
                                             {'label': 'GraphDR',
                                              'value': 'graphdr'},
                                             {'label': 'Diffusion Map',
                                              'value': 'diffusion_map'},
                                             {'label': 'UMAP',
                                              'value': 'umap'},
                                             {'label': 'None',
                                               'value': 'none'}],
                                         value=arguments['--mode'], ),
                                     html.Div([
                                         html.Button('Run projection', id='run-projection-button',
                                                     style={'width': '100%'})
                                     ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '100%',
                                               'margin-top': '1rem'}),
                                     html.Label('Projection options:', style={'margin-top': '1rem'}),
                                     dcc.Checklist(
                                         options=[
                                             {'label': 'Standardize',
                                              'value': 'scale'},
                                             {'label': 'Use selected cells ',
                                              'value': 'subset'}],
                                         value=[],
                                         labelStyle={},
                                         id='dr_checklist',
                                     ),
                                     html.Div(id="embedding-method-options",
                                              children=[
                                                  html.Label('Number of Input PCs'),
                                                  dcc.Dropdown(
                                                      id='dr_N_PCs',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in range(2, MAX_PCS + 1)
                                                      ],
                                                      value=DEFAULT_PCS,
                                                      clearable=False,
                                                  ),
                                                  html.Label('Metric'),
                                                  dcc.Dropdown(
                                                      id='dr_metric_dropdown',
                                                      options=[
                                                          {'label': i, 'value': i}
                                                          for i in ['euclidean',
                                                                    'chebyshev',
                                                                    'canberra',
                                                                    'braycurtis',
                                                                    'mahalanobis',
                                                                    'seuclidean',
                                                                    'cosine',
                                                                    'correlation',
                                                                    'hamming',
                                                                    'jaccard']
                                                      ],
                                                      value='euclidean',
                                                      clearable=False,
                                                  ),
                                                  html.Label('Number of Neighbors'),
                                                  dcc.Dropdown(
                                                      id='dr_n_neighbors_dropdown',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in range(2, 201)
                                                      ],
                                                      value=DEFAULT_DR_K,
                                                      clearable=False,
                                                  ),
                                                  html.Label('Output Dim'),
                                                  dcc.Dropdown(
                                                      id='dr_dim_dropdown',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in range(1, MAX_PCS + 1)
                                                      ],
                                                      value=3,
                                                      clearable=False,
                                                  ),
                                                  html.Label('Min distance'),
                                                  dcc.Dropdown(
                                                      id='dr_min_dist_dropdown',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                                                      ],
                                                      value=0.1,
                                                      clearable=False,
                                                  ),
                                                  html.Label('Regularization (nonlinearity)'),
                                                  dcc.Dropdown(
                                                      id='dr_lambda_dropdown',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in
                                                          [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
                                                      ],
                                                      value=DEFAULT_DR_LAMBDA,
                                                      clearable=False,
                                                  ), ]),
                                     html.Label('Post processing (Visualize trajectory):',
                                                style={'margin-top': '1rem'}),
                                     html.Div([
                                         dcc.RadioItems(
                                             id='embedding-checklist',
                                             options=[
                                                 {'label': 'None',
                                                  'value': 'show_absolutely_nothing'},
                                                 {'label': 'ISOMAP',
                                                  'value': 'show_isomap'}],
                                             value='show_absolutely_nothing',
                                         ),
                                         html.Label('Isomap dimensions'),
                                         dcc.Dropdown(
                                             id='isomap_dim',
                                             options=[
                                                 {'label': str(i), 'value': i}
                                                 for i in range(2, 4)
                                             ],
                                             value=3,
                                             clearable=False,
                                         ),
                                         html.Label('N neighbors'),
                                         dcc.Dropdown(
                                             id='isomap_n_neighbors_dropdown',
                                             options=[
                                                 {'label': str(i), 'value': i}
                                                 for i in range(5, 101)
                                             ],
                                             value=15,
                                             clearable=False,
                                         ),
                                     ]),
                                 ],
                                 style={'padding': '1rem 2.2% 0rem 2.2%', 'margin-left': 0, 'display': 'none'}),
                        html.Div(id="cluster-options",
                                 className="three columns",
                                 children=[
                                     html.Label('Clustering methods:', style={'margin-top': '1rem'}),
                                     dcc.RadioItems(
                                         id='cl_method',
                                         options=[
                                             {'label': 'Spectral clustering',
                                              'value': 'spectral'},
                                             {'label': 'K-means',
                                              'value': 'kmeans'},
                                             {'label': 'Gaussian mixture',
                                              'value': 'gmm'},
                                             {'label': 'Meanshift',
                                              'value': 'meanshift'},
                                         ],
                                         value='spectral'),
                                     html.Div([
                                         html.Button('Run clustering', id='run-cluster-button', style={'width': '100%'})
                                     ], style={'display': 'inline-block', 'margin': '0.5%', 'width': '100%',
                                               'margin-top': '1rem'}),
                                     html.Label('Clustering input:', style={'margin-top': '1rem'}),
                                     dcc.RadioItems(
                                         id='cl-input-checklist',
                                         options=[
                                             {'label': 'Use input data',
                                              'value': 'cl_use_input'},
                                             {'label': 'Use embedding',
                                              'value': 'cl_use_embedding'}],
                                         value='cl_use_input',
                                     ),
                                     html.Div(id="cluster-method-options",
                                              children=[
                                                  html.Label('Number of Neighbors'),
                                                  dcc.Dropdown(
                                                      id='cl_n_neighbors_dropdown',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in range(2, 201)
                                                      ],
                                                      value=30,
                                                      clearable=False,
                                                  ),
                                                  html.Label('Number of Clusters'),
                                                  dcc.Dropdown(
                                                      id='cl_n_clusters_dropdown',
                                                      options=[
                                                          {'label': str(i), 'value': i}
                                                          for i in range(2, 201)
                                                      ],
                                                      value=20,
                                                      clearable=False,
                                                  ),
                                                  html.Div(children=[
                                                      html.Label('Bandwidth'),
                                                      dcc.Dropdown(
                                                          id='cl-meanshift-bandwidth',
                                                          options=[
                                                              {'label': '{: .2f}'.format(i), 'value': i}
                                                              for i in np.linspace(0, 5, 101)
                                                          ],
                                                          value=0.5,
                                                          clearable=False,
                                                      ), ], style={'display': 'none'})
                                              ]
                                              )
                                 ],
                                 style={'padding': '1rem 2.2% 0rem 2.2%', 'margin-left': 0, 'display': 'none'},
                                 ),
                        html.Div(id='scatter_3d_div',
                                 className="nine columns", children=[
                                dcc.Graph(
                                    id='scatter_3d',
                                    figure=dict(
                                        data=[
                                            go.Scatter3d(
                                                x=traj.iloc[:, 0],
                                                y=traj.iloc[:, 1],
                                                z=traj.iloc[:, 2],
                                                mode='markers',
                                                marker=dict(
                                                    size=np.maximum(6 - np.log10(data.shape[0]), 1),
                                                    color=traj.iloc[:, 0],
                                                    line=dict(
                                                        color='rgba(217, 217, 217, 0.14)',
                                                        width=0
                                                    ),
                                                    opacity=0.8,
                                                    showscale=True,
                                                    colorscale=list(zip(BINS, DEFAULT_COLORSCALE)),
                                                    colorbar=dict(len=0.5, yanchor='top', y=0.85),
                                                )
                                            ),
                                        ],
                                        layout=go.Layout(
                                            margin=dict(
                                                l=0,
                                                r=0,
                                                b=0,
                                                t=0
                                            ),
                                            legend=dict(orientation='h'),
                                            paper_bgcolor=BGCOLOR,
                                            plot_bgcolor=BGCOLOR,
                                        )
                                    ),
                                    style={'height': '55vh'},
                                ),
                                # dcc.Graph(
                                #     id='gic_plot',
                                #     figure={
                                #         'data': ff.create_bullet(pd.DataFrame({'history':[[0,1]],'name':'GIC','marker':[[0]]}),subtitles='name',markers='marker',ranges='history')['data'],
                                #         'layout': {'annotations': [{'font': {'color': '#0f0f0f', 'size': 13},
                                #                     'showarrow': False,
                                #                     'text': 'GIC',
                                #                     'textangle': 0,
                                #                     'x': -0.01,
                                #                     'xanchor': 'right',
                                #                     'xref': 'paper',
                                #                     'y': 0.5,
                                #                     'yanchor': 'middle',
                                #                     'yref': 'paper'}],
                                #                 'margin':dict(
                                #                     l=150,
                                #                     r=250,
                                #                     b=0,
                                #                     t=0
                                #                 ),
                                #                 'barmode': 'stack',
                                #                 'shapes': [],
                                #                 'showlegend': False,
                                #                 'height': 15,
                                #                 'xaxis1': {'anchor': 'y1',
                                #                     'domain': [0.0, 1.0],
                                #                     'showgrid': False,
                                #                 'zeroline': False},
                                #                 'yaxis1': {'anchor': 'x1',
                                #                     'domain': [0, 1],
                                #                     'range': [0, 1],
                                #                     'showgrid': False,
                                #                     'showticklabels': False,
                                #                     'zeroline': False},
                                #                 'paper_bgcolor':BGCOLOR,
                                #                 'plot_bgcolor':BGCOLOR}
                                #     },
                                #     config={
                                #         'displayModeBar': False
                                #     },
                                #     style={'margin-top':'1rem'}
                                # )
                            ], style={'margin-left': '12.5%'}),

                    ]),
            ], className='six columns', style={'margin': 0}),

            html.Div(id='pane_right', children=[
                html.Div(id='selector_panel', children=[
                    html.P('Cell selector (Lasso select):',
                           style={'display': 'inline-block', 'margin': '0rem 1rem 1rem 1rem'}),
                    html.Div([
                        html.Div(
                            dcc.Graph(
                                id='select-sample1',
                                selectedData={'points': [], 'range': None},
                                figure=dict(
                                    data=[],
                                    layout=dict(
                                        paper_bgcolor=BGCOLOR,
                                        plot_bgcolor=BGCOLOR,
                                    )),
                                style={'height': '28vh'}
                            ), className="four columns"
                        ),
                        html.Div(
                            dcc.Graph(
                                id='select-sample2',
                                selectedData={'points': [], 'range': None},
                                figure=dict(
                                    data=[],
                                    layout=dict(
                                        paper_bgcolor=BGCOLOR,
                                        plot_bgcolor=BGCOLOR,
                                    )),
                                style={'height': '28vh'}
                            ), className="four columns"),
                        html.Div(
                            dcc.Graph(
                                id='select-sample3',
                                selectedData={'points': [], 'range': None},
                                figure=dict(
                                    data=[],
                                    layout=dict(
                                        paper_bgcolor=BGCOLOR,
                                        plot_bgcolor=BGCOLOR,
                                    )),
                                style={'height': '28vh'}
                            ), className="four columns")
                    ], className="row"),
                    html.Div([
                        html.P('Feature selector (Click or drag and use dropdown below):',
                               style={'display': 'inline-block', 'margin': '3rem 1rem 1rem 1rem'}),
                        html.Div([
                            dcc.RadioItems(
                                options=[
                                    {'label': 'Mean-SD plot',
                                     'value': 'mean_sd'},
                                    {'label': 'Mean-Diff plot',
                                     'value': 'mean_diff'},
                                ],
                                labelStyle={'display': 'inline-block', 'margin': '0.25vw'},
                                id='feature_plot_options',
                                value='mean_sd',
                            )], style={'margin-left': '1rem'}),
                    ], style={'display': 'inline-block'}),
                    dcc.Graph(
                        id='select-feature',
                        selectedData={'points': [], 'range': None},
                        figure=dict(
                            data=[],
                            layout=dict(
                                paper_bgcolor=BGCOLOR,
                                plot_bgcolor=BGCOLOR,
                            )
                        ),
                        style={'height': '38vh'}
                        # animate = True
                    ),
                    html.P('Type or select feature / gene name:',
                           style={'display': 'inline-block', 'margin': '2rem 1rem 1rem 1rem'}),
                    dcc.Dropdown(
                        options=[],
                        id='gene-dropdown'
                    ), ], style={'margin': '0 0 2.2%'}),
                html.Div(id='coexpression_panel',
                         children=[
                             # html.Label('Local gene expression'),
                             # dcc.RadioItems(
                             #             options=[
                             #                 {'label': 'Local',
                             #                     'value': 'show_local'},
                             #                 {'label': 'Global',
                             #                     'value': 'show_global'},
                             #             ],
                             #             labelStyle={'display': 'inline-block', 'margin-right':'0.3vw'},
                             #             id='local-exp-options',
                             #             value = 'show_global',
                             #         ),
                             # dcc.Graph(id = 'localexp_scatter',
                             #     figure = { 'layout': go.Layout(
                             #                 margin = dict(t=0,b=0,l=0,r=0),
                             #                 legend = dict(orientation = 'h'),
                             #                 paper_bgcolor=BGCOLOR,
                             #                 plot_bgcolor=BGCOLOR
                             #             )},
                             #     style={'height':'30vh','width':'30vw','margin-left':'10vw',}),
                             html.Div([
                                 html.Label(
                                     'Select displayed features / genes  (Click on above or use dropdown below):'),
                                 dcc.Dropdown(
                                     options=[{'label': gene, 'value': gene} for gene in
                                              network_data.index] if with_network_data else [],
                                     id='networkgene-dropdown',
                                     multi=True,
                                     value=network_data.index[:20].tolist() if with_network_data else [],
                                 ), ], style={'margin': '0 0 2.2%'}),
                             html.Label('Local covariation network'),
                             html.Label('Effective sample size: ', id='effective_n',
                                        style={'text-align': 'center', 'margin-top': '2%'}),
                             dcc.Graph(id='coexp_heatmap',
                                       figure={'data': [go.Heatmap(x=network_data.index[:20].tolist(),
                                                                   y=network_data.index[:20].tolist(),
                                                                   z=np.zeros((20, 20)), colorscale='Viridis', xgap=1,
                                                                   ygap=1,
                                                                   showscale=False)] if with_network_data else [],
                                               'layout': go.Layout(
                                                   margin=dict(t=10),
                                                   legend=dict(orientation='h'),
                                                   paper_bgcolor=BGCOLOR,
                                                   plot_bgcolor=BGCOLOR
                                               )},
                                       style={'height': '60vh', 'width': '40vw', 'margin-left': '5vw',
                                              'margin-top': '2%'})
                         ],
                         style={'margin': '0 0 2.2%', 'display': 'none'})
            ], className='six columns', style={'margin': '0'})]),
        html.Div(id='fullscreen_div',
                 className="twelve columns", children=[
                dcc.Graph(
                    id='scatter_3d_fc',
                    figure=dict(
                        data=[],
                        layout=go.Layout(
                            margin=dict(
                                r=0,
                                t=0
                            ),
                            legend=dict(orientation='h'),
                            paper_bgcolor=BGCOLOR,
                            plot_bgcolor=BGCOLOR
                        )
                    ),
                    style={'height': '90vh', 'width': '100vw'}
                )], style={'display': 'none'}),
        html.Div([
            dcc.Checklist(
                options=[
                    {'label': 'Full screen',
                     'value': 'full_screen'}],
                value=[],
                labelStyle={'display': 'inline-block'},
                id='full-screen-options',
            )], className='twelve columns', style={'margin-left': '1.1%'}),
        html.Div(id='dummy', style={'display': 'none'}),
        html.Div(id='dummy2', style={'display': 'none'}),
        html.Div(id='dummy3', style={'display': 'none'}),
        html.Div(id='dummy4', style={'display': 'none'}),
        html.Div(id='dummy_dr', style={'display': 'none'}),
        html.Div(id='dummy_cl', style={'display': 'none'})

    ])

    # app.css.append_css(
    #        {'external_url': 'https://codepen.io/jzthree/pen/ERrLwd.css'})

    save_button_counter = 0


    @app.callback(
        Output('dummy', 'children'),
        [Input('save-button', 'n_clicks'),
         Input('notification', 'n_clicks')])
    def save_traj_notification(n_clicks_save, n_clicks_alert):
        global save_button_counter
        global message
        if n_clicks_save != None and n_clicks_save != save_button_counter:
            save_button_counter = n_clicks_save
            traj.to_csv('./output.txt', sep='\t', index_label=False)
            message.append('Cell coordinates saved to ./output.txt.')
            if len(output_dict) > 1:
                output_df = pd.DataFrame.from_dict(output_dict)
                output_df = output_df.set_index('index')
                output_df.to_csv('./output_info.txt', sep='\t', index_label=False)
                message.append('Computed cell state information saved to ./output_info.txt.')

        return []


    @app.callback(Output('scatter_3d_fc', 'figure'),
                  [Input('scatter_3d', 'figure'),
                   Input('fullscreen_div', 'style')],
                  [State('full-screen-options', 'value'),
                   State('scatter_3d_fc', 'figure')])
    def update_scatter_3d_fc(figure, style, value, bfigure):
        if 'full_screen' in value:
            bfigure['data'] = figure['data']
            bfigure['layout'] = figure['layout']
            return bfigure
        else:
            return bfigure


    @app.callback(Output('fullscreen_div', 'style'),
                  [Input('full-screen-options', 'value')])
    def update_fullscreen_div(value):
        if 'full_screen' in value:
            return {'display': 'block'}
        else:
            return {'display': 'none'}


    @app.callback(Output('all-plots', 'style'),
                  [Input('full-screen-options', 'value')])
    def update_all_plots(value):
        if 'full_screen' in value:
            return {'display': 'none'}
        else:
            return {'display': 'block'}


    @app.callback(
        Output('notification', 'children'),
        [Input('dummy', 'children'),
         Input('upload_label', 'children'),
         Input('upload_feature_label', 'children'),
         Input('upload_annotation_label', 'children'),
         Input('upload_trajectory_label', 'children'),
         Input('scatter_3d', 'figure'),
         Input('show-options', 'options'),
         Input('dummy_dr', 'children')])
    def notify(*args):
        global message
        if len(message) > 0:
            message_delivered = message
            message = []
            return html.Div(id='alert', children="; ".join(message_delivered), className='alert')
        else:
            return []


    @app.callback(
        Output('alg-options', 'style'),
        [Input('show-options', 'value')],
        [State('alg-options', 'style')]
    )
    def show_options_a(value, style):
        if value == 'show_alg_options':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output('disp-options', 'style'),
        [Input('show-options', 'value')],
        [State('disp-options', 'style')]
    )
    def show_options_b(value, style):
        if value == 'show_disp_options':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output('embedding-options', 'style'),
        [Input('show-options', 'value')],
        [State('embedding-options', 'style')]
    )
    def show_options_c(value, style):
        if value == 'show_embedding_options':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output('cluster-options', 'style'),
        [Input('show-options', 'value')],
        [State('cluster-options', 'style')]
    )
    def show_options_d(value, style):
        if value == 'show_cluster_options':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output('network-options', 'style'),
        [Input('show-options', 'value')],
        [State('network-options', 'style')]
    )
    def show_options_e(value, style):
        if value == 'show_network_options':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output('selector_panel', 'style'),
        [Input('show-options', 'value')],
        [State('selector_panel', 'style')]
    )
    def update_selector_panel(value, style):
        if value == 'show_network_options':
            style['display'] = 'none'
        else:
            style['display'] = 'block'
        return style


    @app.callback(
        Output('coexpression_panel', 'style'),
        [Input('show-options', 'value')],
        [State('coexpression_panel', 'style')]
    )
    def update_coexpression_panel(value, style):
        if value == 'show_network_options':
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


    @app.callback(
        Output('scatter_3d_div', 'style'),
        [Input('show-options', 'value')],
        [State('scatter_3d_div', 'style')]
    )
    def update_scatter_3d_div_style(value, style):
        if value != 'show_no_options':
            style['margin-left'] = 0
        else:
            style['margin-left'] = '12.5%'
        return style


    @app.callback(
        Output('x_dropdown', 'options'),
        [Input('ndim_dropdown', 'value')])
    def update_x_dropdown(ndim):
        return [{'label': str(i + 1) if i != -1 else '', 'value': i} for i in range(-1, ndim)]


    @app.callback(
        Output('y_dropdown', 'options'),
        [Input('ndim_dropdown', 'value')])
    def update_y_dropdown(ndim):
        return [{'label': str(i + 1) if i != -1 else '', 'value': i} for i in range(-1, ndim)]


    @app.callback(
        Output('z_dropdown', 'options'),
        [Input('ndim_dropdown', 'value')])
    def update_z_dropdown(ndim):
        return [{'label': str(i + 1) if i != -1 else '', 'value': i} for i in range(-1, ndim)]


    @app.callback(
        Output('x_dropdown', 'value'),
        [Input('ndim_dropdown', 'value')],
        [State('x_dropdown', 'value')])
    def update_x_dropdown_value(ndim, value):
        if value >= ndim:
            return -1
        else:
            return value


    @app.callback(
        Output('y_dropdown', 'value'),
        [Input('ndim_dropdown', 'value')],
        [State('y_dropdown', 'value')])
    def update_y_dropdown_value(ndim, value):
        if value >= ndim:
            return -1
        else:
            return value


    @app.callback(
        Output('z_dropdown', 'value'),
        [Input('ndim_dropdown', 'value')],
        [State('z_dropdown', 'value'),
         State('x_dropdown', 'value'),
         State('y_dropdown', 'value')])
    def update_z_dropdown_value(ndim, value, valuex, valuey):
        if value >= ndim:
            return -1
        else:
            if value == -1 and valuex == 0 and valuey == 1 and ndim > 2:
                return 2
            else:
                return value


    @app.callback(
        Output('annotation_dropdown', 'options'),
        [Input('upload_annotation_label', 'children'),
         Input('dummy_cl', 'children')])
    def update_annotation_dropdown_options(children, dummy):
        return [{'label': annotation_data.columns.values[i], 'value': annotation_data.columns.values[i]} for i in
                range(annotation_data.shape[1])]


    @app.callback(
        Output('annotation_dropdown', 'value'),
        [Input('dummy_cl', 'children')])
    def update_annotation_dropdown_value(cl_name):
        if len(cl_name) > 0:
            return cl_name[0]


#    @app.callback(
#        Output('annotation_dropdown_div', 'style'),
#        [Input('upload_annotation_label', 'children')])
#    def update_annotation_dropdown_div_style(children):
#        if annotation_data.shape[1] > 1:
#            return {'display': 'block'}
#        else:
#            return {'display': 'none'}


    @app.callback(
        Output('show-options', 'options'),
        [Input('dummy_dr', 'children'),
         Input('upload_label', 'children'),
         Input('upload_feature_label', 'children')],
        [State('show-options', 'options')]
    )
    def disable_network_options(a, b, c, options):
        global message
        assert options[-2]['label'] == 'Network'
        options[-2]['disabled'] = not with_network_data
        if options[-2]['disabled']:
            message.append("Network disabled.")

        assert options[-4]['label'] == 'Projection'
        options[-4]['disabled'] = not with_feature_data
        if options[-4]['disabled']:
            message.append("Projection disabled.")
        return options


    @app.callback(
        Output('show-options', 'value'),
        [Input('show-options', 'options')],
        [State('show-options', 'value')]
    )
    def disable_network_value(options, value):
        global message
        assert options[-2]['label'] == 'Network'
        assert options[-2]['value'] == 'show_network_options'
        if options[-2]['disabled'] and value == 'show_network_options':
            value = 'show_no_options'
        return value


    @app.callback(
        Output('dummy_cl', 'children'),
        [
            Input('run-cluster-button', 'n_clicks'),
        ],
        [
            State('cl_method', 'value'),
            State('cl_n_neighbors_dropdown', 'value'),
            State('cl_n_clusters_dropdown', 'value'),
            State('cl-input-checklist', 'value'),
            State('cl-meanshift-bandwidth', 'value'),
            State('njobs_dropdown', 'value'),

        ]
    )
    def run_clustering(n_clicks_run_clustering, cl_method, n_neighbors, n_clusters, cl_input, bandwidth, n_jobs):
        global annotation_data
        global output_dict
        if n_clicks_run_clustering == None or n_clicks_run_clustering == 0:
            return []
        if cl_method == 'spectral':
            model = SpectralClustering(affinity='nearest_neighbors', assign_labels='discretize',
                                       n_neighbors=n_neighbors, n_clusters=n_clusters, n_jobs=n_jobs)
            c_name = 'c_' + cl_method + '_n' + str(n_neighbors) + '_k' + str(n_clusters)
        elif cl_method == 'kmeans':
            model = KMeans(n_clusters, n_jobs=n_jobs)
            c_name = 'c_' + cl_method + '_k' + str(n_clusters)
        elif cl_method == 'gmm':
            model = GaussianMixture(n_clusters)
            c_name = 'c_' + cl_method + '_k' + str(n_clusters)
        elif cl_method == 'meanshift':
            model = MeanShift(bandwidth, n_jobs=n_jobs)
            c_name = 'c_' + cl_method + '_h' + '{: .2f}'.format(bandwidth)

        cl_data = input_data.values if cl_input == 'cl_use_input' else data.values
        model.fit(cl_data)
        output_dict[c_name] = model.labels_ if cl_method != 'gmm' else model.predict(cl_data)
        annotation_data[c_name] = output_dict[c_name]
        return [c_name]




    @app.callback(
        Output('dummy_dr', 'children'),
        [
            Input('run-projection-button', 'n_clicks'),
            Input('upload_feature_label', 'children'),
            Input('upload_label', 'children'),
            Input('upload_velocity_label', 'children'),
        ],
        [
            State('dr_method', 'value'),
            State('dr_checklist', 'value'),
            State('dr_n_neighbors_dropdown', 'value'),
            State('dr_N_PCs', 'value'),
            State('dr_min_dist_dropdown', 'value'),
            State('dr_metric_dropdown', 'value'),
            State('dr_dim_dropdown', 'value'),
            State('dr_lambda_dropdown', 'value'),
            State('bandwidth_dropdown', 'value'),
            State('min_radius_dropdown', 'value'),
            State('njobs_dropdown', 'value'),
            State('select-sample1', 'selectedData'),
            State('select-sample2', 'selectedData'),
            State('select-sample3', 'selectedData'),
        ],

    )
    def run_projection(n_clicks_run_projection, dummy, dummy2, dummy3, dr_method, dr_checklist, dr_n_neighbors,
                       dr_N_PCs, \
                       dr_min_dist, dr_metric, dr_dim, dr_lambda, bw, min_radius, n_jobs,
                       selectedData1, selectedData2, selectedData3):
        global data
        global traj
        global history
        global output_dict
        global s
        global with_pca
        global projection_mode
        global n_clicks_run_projection_counter
        global input_data_pca
        global N_PCs
        global with_velocity_data
        global velocity_data
        global run_projection_initial_call
        
        # prevent it from running during initialization
        if n_clicks_run_projection:
            pass
        else:
            return []
        
        print("Run Projection...")

        if 'subset' in dr_checklist:
            index = input_data.index.values
            for _, d in enumerate([selectedData1, selectedData2, selectedData3]):
                if d:
                    selected_index = [p['customdata'] for p in d['points']]
                else:
                    selected_index = []
                if len(selected_index) > 0:
                    index = np.intersect1d(index, selected_index)

            # if no cell is selected, compute for all cells
            if len(index) == 0:
                selectind = np.arange(input_data.shape[0])
            else:
                selectind = match(index, input_data.index.values)
        else:
            selectind = np.arange(input_data.shape[0])

        N_PCs = reduce(np.minimum, [len(selectind), MAX_PCS, input_data.shape[0], input_data.shape[1]])
        input_data_pca = PCA(N_PCs)

        if dr_method == "none":
            data = input_data.copy()
            projection_mode = 'none'
        else:
            if 'scale' in dr_checklist:
                input_data_scaler = StandardScaler()
                data = pd.DataFrame(
                    input_data_pca.fit_transform(input_data_scaler.fit_transform(input_data.values[selectind, :])),
                    index=input_data.index[selectind], columns=['PC' + str(i) for i in range(1, N_PCs + 1)])
                if with_velocity_input_data:
                    velocity_data = pd.DataFrame(
                        input_data_pca.transform(velocity_input_data.values[selectind, :] / input_data_scaler.scale_),
                        index=velocity_input_data.index[selectind], columns=['PC' + str(i) for i in range(1, N_PCs + 1)])
                    with_velocity_data = True
            else:
                data = pd.DataFrame(input_data_pca.fit_transform(input_data.values[selectind, :]),
                                    index=input_data.index[selectind], columns=['PC' + str(i) for i in range(1, N_PCs + 1)])
                if with_velocity_input_data:
                    velocity_data = pd.DataFrame(input_data_pca.transform(velocity_input_data.values[selectind, :]),
                                                 index=velocity_input_data.index[selectind],
                                                 columns=['PC' + str(i) for i in range(1, N_PCs + 1)])
                    with_velocity_data = True

            if 'diffusion_map' == dr_method:
                D = squareform(pdist(data.values[:, :dr_N_PCs], metric=dr_metric))
                bws = np.median(D, axis=1)
                # D = kneighbors_graph(data.values[:,:dr_N_PCs], dr_n_neighbors, mode='distance', n_jobs=n_jobs)
    
                bw_square_sums = np.add.outer(bws ** 2, bws ** 2)
                D = np.exp(- D ** 2 / bw_square_sums) * np.sqrt(2 * np.multiply.outer(bws, bws) / bw_square_sums)
                # make symmetric
                W = D
                q = 1.0 / np.asarray(W.sum(axis=0))
                W = W * q[:, np.newaxis] * q[np.newaxis, :]
                z = 1.0 / np.sqrt(np.asarray(W.sum(axis=0)))
                W = W * z[:, np.newaxis] * z[np.newaxis, :]
                eigvals, eigvecs = np.linalg.eigh(W)
                # eigvals, eigvecs = eigsh(W, k=N_PCs, which='LM')
                eigvecs = eigvecs[:, ::-1][:, :N_PCs]
                data = pd.DataFrame(eigvecs, index=feature_data.columns,
                                    columns=['DC' + str(i) for i in range(1, eigvecs.shape[1] + 1)])
                projection_mode = 'diffusion_map'
            elif 'umap' == dr_method:
                mapped = umap.UMAP(n_components=dr_dim, n_neighbors=dr_n_neighbors, min_dist=dr_min_dist,
                                   metric=dr_metric).fit_transform(data.values[:, :dr_N_PCs])
                data = pd.DataFrame(mapped, index=feature_data.columns,
                                    columns=['UMAP' + str(i) for i in range(1, mapped.shape[1] + 1)])
                projection_mode = 'umap'
            elif 'graphdr' == dr_method:
                mapped = graphdr(data.values[:, :dr_N_PCs], n_neighbors=dr_n_neighbors, regularization=dr_lambda,
                                 metric=dr_metric)
                data = pd.DataFrame(mapped, index=feature_data.columns,
                                    columns=['GraphDR' + str(i) for i in range(1, mapped.shape[1] + 1)])
                projection_mode = 'graphdr'
            else:
                projection_mode = 'pca'

        if projection_mode not in ['pca', 'graphdr', 'none']:
            if with_velocity_input_data:
                with_velocity_data = False
                message.append('Velocity is only supported for PCA, GraphDR, or no projection.')

        # scale
        if with_velocity_data:
            velocity_data = velocity_data / np.std(data.iloc[:, 0])

        data = (data - np.mean(data, axis=0)) / np.std(data.iloc[:, 0])

        # reinitialize
        traj = data.iloc[:, :ndim].copy()
        s = scms2.Scms(np.asarray(data.iloc[:, :ndim]).copy(), bw, min_radius=min_radius)
        history = [traj.copy()]
        output_dict = {'index': traj.index.values}
        return []


    current_gene = None
    run_button_counter = 0
    reset_button_counter = 0
    bootstrap_button_counter = 0
    bootstrap_trajs = []


    # note upload_label, upload_annotation_label, ndim_dropdown(value) and isplay-checklist(values)  should not be in the input and it has been covered by dependencies
    @app.callback(
        Output('scatter_3d', 'figure'),
        [
            Input('run-button', 'n_clicks'),
            Input('reset-button', 'n_clicks'),
            Input('bootstrap-button', 'n_clicks'),
            Input('upload_trajectory_label', 'children'),
            Input('opacity-slider', 'value'),
            Input('dotsize-slider', 'value'),
            Input('colorscale-picker', 'colorscale'),
            Input('gene-dropdown', 'value'),
            Input('select-sample1', 'selectedData'),
            Input('select-sample2', 'selectedData'),
            Input('select-sample3', 'selectedData'),
            Input('smoothing-slider', 'value'),
            Input('conesize-slider', 'value'),
            Input('scatter3d_aspect_options', 'value'),
            Input('x_dropdown', 'value'),
            Input('y_dropdown', 'value'),
            Input('z_dropdown', 'value'),
            Input('annotation_dropdown', 'value'),
            Input('embedding-checklist', 'value'),
            Input('isomap_n_neighbors_dropdown', 'value'),
            Input('annotation_type', 'value'),
            Input('label_checklist', 'value'),
            Input('dummy_dr', 'children'),
            Input('dummy4', 'children')],
        [State('scatter_3d', 'figure'),
         State('scatter_3d', 'relayoutData'),
         State('ITERATIONS-slider', 'value'),
         State('ndim_dropdown', 'value'),
         State('dimensionality_dropdown', 'value'),
         State('bandwidth_dropdown', 'value'),
         State('min_radius_dropdown', 'value'),
         State('relaxation_dropdown', 'value'),
         State('stepsize_dropdown', 'value'),
         State('njobs_dropdown', 'value'),
         State('method_checkbox', 'value'),
         State('display-checklist', 'value'),
         ])
    def update_traj_3d(n_clicks_run, n_clicks_reset, n_clicks_bootstrap, upload_trajectory_label, opacity, dotsize,
                       colorscale, selected_gene, selectedData1, selectedData2, selectedData3, smooth_radius, conesize,
                       scatter3d_aspect_option, dimx, dimy, dimz, annotation_index, embedding_value, isomap_n_neighbors,
                       annotation_type, label_checklist_value, dummy_dr, dummy4, \
                       figure, relayoutData, n_iter, ndim_, dim, bw, min_radius, relaxation, step_size, n_jobs, method,
                       display_value):
        global s
        global traj
        global data
        global history
        global ndim
        global run_button_counter
        global reset_button_counter
        global bootstrap_button_counter
        global output_dict
        global seg_identity
        global mst_betweenness_centrality
        global message
        global maxlogp
        global bootstrap_trajs
        # global traj_copy
        cm = list(zip(BINS, colorscale))

        def select_traj(traj, dimx, dimy, dimz):
            if dimx != -1:
                x = traj.iloc[:, dimx]
            else:
                x = np.zeros(traj.shape[0])
            if dimy != -1:
                y = traj.iloc[:, dimy]
            else:
                y = np.zeros(traj.shape[0])
            if dimz != -1:
                z = traj.iloc[:, dimz]
            else:
                z = np.zeros(traj.shape[0])
            return x, y, z

        if (n_clicks_reset != None and n_clicks_reset != reset_button_counter) or ndim_ != ndim:
            traj = data.iloc[:, :ndim_].copy()
            s = scms2.Scms(np.asarray(data.iloc[:, :ndim_]).copy(), bw, min_radius=min_radius)
            reset_button_counter = n_clicks_reset
            ndim = ndim_
            history = [traj.copy()]
            bootstrap_trajs = []
            bootstrap_traces = []
            output_dict = {'index': traj.index.values}

        if s.min_radius != min_radius or s.bw != bw:
            s.reset_bw(bw, min_radius=min_radius)

        # run SCMS

        if n_clicks_run != None and n_clicks_run != run_button_counter:
            start_time = time.time()
            if n_jobs > 1:
                pool = multiprocess.Pool(n_jobs)
            for _ in range(n_iter):
                # s.reset_bw(bw, min_radius=min_radius)
                if n_jobs == 1:
                    update = np.vstack([s.scms_update(batch_data, method=method, stepsize=step_size,
                                                      ridge_dimensionality=dim,
                                                      relaxation=relaxation)[0] for batch_data in
                                        np.array_split(traj.iloc[:, :ndim].values, np.ceil(traj.shape[0] / BATCHSIZE))])
                else:
                    update = pool.map(
                        lambda pos: s.scms_update(pos, method=method, stepsize=step_size, ridge_dimensionality=dim,
                                                  relaxation=relaxation)[0],
                        np.array_split(np.asarray(traj.iloc[:, :ndim]), np.ceil(traj.shape[0] / (BATCHSIZE * n_jobs))))
                    update = np.vstack(update)

                traj.iloc[:, :ndim] = traj.iloc[:, :ndim] + update
                history.append(traj.copy())

            if n_jobs > 1:
                pool.close()
                pool.terminate()
                pool.join()
            run_button_counter = n_clicks_run
            print("Elapsed time: {: .2f}".format(time.time() - start_time))

        # if gene is selected, color by gene value
        if selected_gene:
            c = feature_data.loc[:, traj.index].values[feature_data.index.values == selected_gene, :].flatten()
            if smooth_radius > 0:
                smooth_mat = np.exp(-(squareform(pdist(traj)) / smooth_radius) ** 2)
                c = smooth_mat.dot(c) / np.sum(smooth_mat, axis=1)
        else:
            c = np.asarray(traj.iloc[:, 0])

        # run bootstrap
        bootstrap_traces = []
        if n_clicks_bootstrap != None and n_clicks_bootstrap != bootstrap_button_counter:
            bootstrap_button_counter = n_clicks_bootstrap
            if projection_mode == 'pca' or projection_mode == 'none':
                bootstrap_trajs = []
                for i in range(5):
                    b = scms2.Scms(scms2.bootstrap_resample(np.asarray(data.iloc[:, :ndim].copy()))[0], bw,
                                   min_radius=min_radius)
                    bootstrap_traj = data.copy()
                    bootstrap_traj.iloc[:, :ndim] = np.vstack([b.scms(batch_data, n_iterations=n_iter, threshold=0,
                                                                      method=method, stepsize=step_size,
                                                                      ridge_dimensionality=dim,
                                                                      relaxation=relaxation,
                                                                      n_jobs=n_jobs)[0] for batch_data in
                                                               np.array_split(bootstrap_traj.iloc[:, :ndim].values,
                                                                              np.ceil(bootstrap_traj.shape[0] / (
                                                                                          BATCHSIZE * n_jobs)))])
                    bootstrap_trajs.append(bootstrap_traj)
                traj = data.copy()
                s = scms2.Scms(np.asarray(data.iloc[:, :ndim]).copy(), bw, min_radius=min_radius)
                traj.iloc[:, :ndim] = np.vstack([s.scms(batch_data, n_iterations=n_iter, threshold=0, method=method,
                                                        stepsize=step_size, ridge_dimensionality=dim,
                                                        relaxation=relaxation)[0] for
                                                 batch_data in np.array_split(traj.iloc[:, :ndim].values, np.ceil(
                        traj.shape[0] / (BATCHSIZE * n_jobs)))])
            else:
                message.append("Boostrap is only supported for PCA projection or no projection.")

        if 'show_bootstrap' in display_value and len(bootstrap_trajs) > 0:
            for i, traj in enumerate(bootstrap_trajs):
                x, y, z = select_traj(traj, dimx, dimy, dimz)
                bootstrap_traces.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='markers',
                        marker=dict(
                            size=dotsize * 0.5,
                            color=c,
                            # line=dict(
                            #     color='rgba(217, 217, 217, 0.14)',
                            #     width=0.5
                            # ),
                            opacity=0.8,
                            showscale=False
                        ),
                        name='Bootstrap ' + str(i + 1)
                    )
                )

        input_trace = []
        if 'show_original' in display_value:
            datax, datay, dataz = select_traj(data, dimx, dimy, dimz)

        def prune_segments(edge_list, prune_threshold=3):
            edge_list = np.asarray(edge_list)
            degree = utils.count_degree(edge_list, traj.shape[0])
            segments = utils.extract_segments(edge_list, degree)
            prune_threshold = 3
            seglens = np.asarray([len(seg) for seg in segments if len(seg) != 0])
            seg_min_degrees = np.asarray([np.min(degree[seg]) for seg in segments if len(seg) != 0])
            remove_seginds = (seglens <= prune_threshold) * (seg_min_degrees == 1)
            while np.any(remove_seginds):
                remove_nodeinds_segments = [segments[i] for i in np.where(remove_seginds)[0]]
                # remove_nodeinds = segments[np.where(remove_seginds)[0][np.argmin(seglens[np.where(remove_seginds)[0]])]]
                remove_nodeinds_segments_includebranchpoint = [np.any(degree[nodeinds] > 2) for nodeinds in
                                                               remove_nodeinds_segments]

                edge_list_new = []
                for edge in edge_list:
                    remove = False
                    for includebranchpoint, nodeinds in zip(remove_nodeinds_segments_includebranchpoint,
                                                            remove_nodeinds_segments):
                        if includebranchpoint:
                            if edge[0] in nodeinds and edge[1] in nodeinds:
                                remove = True
                        else:
                            if edge[0] in nodeinds or edge[1] in nodeinds:
                                remove = True
                    if not remove:
                        edge_list_new.append(edge)

                edge_list = edge_list_new
                edge_list = np.asarray(edge_list)
                degree = utils.count_degree(edge_list, traj.shape[0])
                segments = utils.extract_segments(edge_list, degree)
                seglens = np.asarray([len(seg) for seg in segments if len(seg) != 0])
                seg_min_degrees = np.asarray([np.min(degree[seg]) for seg in segments if len(seg) != 0])
                remove_seginds = (seglens <= prune_threshold) * (seg_min_degrees == 1)
            return segments, edge_list

        isomap_trace = []
        if 'show_isomap' == embedding_value:
            e = Isomap(n_components=3, n_neighbors=isomap_n_neighbors).fit_transform(traj.values)
            x = e[:, 0]
            y = e[:, 1]
            z = e[:, 2]
            isomap_trace.append(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                customdata=traj.index,
                marker=dict(
                    size=dotsize,
                    color=c,
                    opacity=opacity * 0.3,
                    colorscale=cm,
                    showscale=False
                ),
                name='ISOMAP'
            ))
        else:
            x, y, z = select_traj(traj, dimx, dimy, dimz)
            if with_velocity_data:
                u, v, w = select_traj(velocity_data, dimx, dimy, dimz)

        mst_traces = []
        segment_traces = []
        order_trace = []

        if 'show_mst' in display_value:
            edge_list_raw = utils.make_mst(np.asarray(traj.iloc[:, :ndim]))

            if 'show_segments' in display_value:
                segments, edge_list = prune_segments(edge_list_raw)

                seg_identity = np.zeros(traj.shape[0])
                for i, seg in enumerate(segments):
                    seg_identity[seg] = i + 1
                output_dict['Segment'] = seg_identity
                print(str(np.sum(seg_identity == 0)) + ' cells are not assigned to segments.')
                if 'show_order' in display_value:
                    g = nx.from_edgelist(edge_list)
                    mst_betweenness_centrality_dict = nx.betweenness_centrality(g)
                    mst_betweenness_centrality = np.empty(traj.shape[0])
                    mst_betweenness_centrality.fill(np.nan)
                    for k in mst_betweenness_centrality_dict:
                        mst_betweenness_centrality[k] = mst_betweenness_centrality_dict[k]
                    output_dict['MST Betweenness Centrality'] = mst_betweenness_centrality
                    output_dict[
                        'Cell Order (MST betweenness centrality rank)'] = mst_betweenness_centrality.argsort().argsort()
                    valid_inds = ~np.isnan(mst_betweenness_centrality)

                    order_trace.append(
                        go.Scatter3d(
                            x=x[valid_inds],
                            y=y[valid_inds],
                            z=z[valid_inds],
                            text=['%.3e' % x for x in mst_betweenness_centrality[valid_inds]],
                            mode='markers',
                            customdata=traj.index[valid_inds],
                            marker=dict(
                                size=dotsize,
                                color=mst_betweenness_centrality[valid_inds],
                                opacity=1,
                                colorscale=cm,
                                showscale='show_legend' in display_value,
                                colorbar=dict(len=0.5, yanchor='top', y=0.85),
                            ),
                            hoverinfo='text',
                            name='Betweenness centrality',
                            visible='legendonly'
                        )
                    )

            if 'show_segments' in display_value:
                if len(segments) < 100:
                    for i in range(len(segments)):
                        if 'show_original' in display_value:
                            segment_traces.append(
                                go.Scatter3d(
                                    x=datax[seg_identity == (i + 1)],
                                    y=datay[seg_identity == (i + 1)],
                                    z=dataz[seg_identity == (i + 1)],
                                    mode='markers',
                                    customdata=traj.index[seg_identity == (i + 1)],
                                    marker=dict(
                                        symbol=SYMBOLS[int(i / 10)],
                                        size=dotsize,
                                        color=DEFAULT_PLOTLY_COLORS[i % 10],
                                        opacity=opacity * 0.3,
                                        showscale=False
                                    ),
                                    name='Original S' + str(i + 1),
                                )
                            )
                        segment_traces.append(
                            go.Scatter3d(
                                x=x[seg_identity == (i + 1)],
                                y=y[seg_identity == (i + 1)],
                                z=z[seg_identity == (i + 1)],
                                mode='markers',
                                customdata=traj.index[seg_identity == (i + 1)],
                                marker=dict(
                                    symbol=SYMBOLS[int(i / 10)],
                                    color=DEFAULT_PLOTLY_COLORS[i % 10],
                                    size=dotsize,
                                    opacity=opacity,
                                    showscale=False
                                ),
                                name='S' + str(i + 1),
                            )
                        )
                    if 'show_original' in display_value:
                        segment_traces.append(
                            go.Scatter3d(
                                x=datax[seg_identity == 0],
                                y=datay[seg_identity == 0],
                                z=dataz[seg_identity == 0],
                                mode='markers',
                                customdata=traj.index[seg_identity == 0],
                                marker=dict(
                                    size=dotsize,
                                    symbol=SYMBOLS[int((i + 1) / 10)],
                                    color=DEFAULT_PLOTLY_COLORS[i % 10],
                                    opacity=opacity * 0.3,
                                    showscale=False
                                ),
                                # visible = 'legendonly',
                                name='Original Segments Unassigned',
                            )
                        )
                    segment_traces.append(
                        go.Scatter3d(
                            x=x[seg_identity == 0],
                            y=y[seg_identity == 0],
                            z=z[seg_identity == 0],
                            customdata=traj.index[seg_identity == 0],
                            mode='markers',
                            marker=dict(
                                size=dotsize,
                                symbol=SYMBOLS[int((i + 1) / 10)],
                                color=DEFAULT_PLOTLY_COLORS[(i + 1) % 10],
                                opacity=opacity,
                                showscale=False
                            ),
                            # visible = 'legendonly',
                            name='Segments Unassigned',
                        )
                    )
                else:
                    message.append(
                        ">100 many segments. Maybe the trajectory hasn't converged or used inappropriate parameter?")

            mst_traces = []
            list_x = []
            list_y = []
            list_z = []
            list_color = []
            for edge in edge_list_raw:
                i, j = edge
                if 'show_segments' in display_value:
                    if seg_identity[i] == 0 or seg_identity[j] == 0:
                        continue

                if dimx != -1:
                    xs = [traj.iloc[i, dimx], traj.iloc[j, dimx]]
                else:
                    xs = [0, 0]
                if dimy != -1:
                    ys = [traj.iloc[i, dimy], traj.iloc[j, dimy]]
                else:
                    ys = [0, 0]
                if dimz != -1:
                    zs = [traj.iloc[i, dimz], traj.iloc[j, dimz]]
                else:
                    zs = [0, 0]

                list_x.extend(xs)
                list_y.extend(ys)
                list_z.extend(zs)
                list_color.extend(xs)
                list_x.append(None)
                list_y.append(None)
                list_z.append(None)
                list_color.append('#FFFFFF')

            mst_traces.append(
                go.Scatter3d(
                    x=list_x,
                    y=list_y,
                    z=list_z,
                    mode='lines',
                    line=dict(
                        color=list_color,
                        width=dotsize * 0.5,
                        showscale=False,
                    ),
                    name='MST',
                )
            )

        knn_traces = []
        if 'show_knn' in display_value or 'show_knn_traj' in display_value:
            if 'show_knn_traj' in display_value:
                nbrs = NearestNeighbors(n_neighbors=5).fit(np.asarray(traj.iloc[:, :ndim]))
                edge_list_raw = np.vstack(nbrs.kneighbors_graph(np.asarray(traj.iloc[:, :ndim])).nonzero()).T
            else:
                nbrs = NearestNeighbors(n_neighbors=5).fit(np.asarray(data.iloc[:, :ndim]))
                edge_list_raw = np.vstack(nbrs.kneighbors_graph(np.asarray(data.iloc[:, :ndim])).nonzero()).T

            list_x = []
            list_y = []
            list_z = []
            list_color = []
            for edge in edge_list_raw:
                i, j = edge
                if 'show_segments' in display_value and 'show_mst' in display_value:
                    if seg_identity[i] == 0 or seg_identity[j] == 0:
                        continue
                if dimx != -1:
                    xs = [traj.iloc[i, dimx], traj.iloc[j, dimx]]
                else:
                    xs = [0, 0]
                if dimy != -1:
                    ys = [traj.iloc[i, dimy], traj.iloc[j, dimy]]
                else:
                    ys = [0, 0]
                if dimz != -1:
                    zs = [traj.iloc[i, dimz], traj.iloc[j, dimz]]
                else:
                    zs = [0, 0]
                list_x.extend(xs)
                list_y.extend(ys)
                list_z.extend(zs)
                list_color.extend(xs)
                list_x.append(None)
                list_y.append(None)
                list_z.append(None)
                list_color.append('#FFFFFF')

            knn_traces.append(
                go.Scatter3d(
                    x=list_x,
                    y=list_y,
                    z=list_z,
                    mode='lines',
                    line=dict(
                        color=list_color,
                        width=dotsize * 0.5,
                        showscale=False,
                    ),
                    name='KNN Graph'
                )
            )

        history_traces = []
        if 'show_traces' in display_value and len(history) > 1:
            list_x = []
            list_y = []
            list_z = []
            list_color = []
            for i in range(traj.shape[0]):
                if 'show_segments' in display_value:
                    if seg_identity[i] == 0:
                        continue
                if dimx != -1:
                    xs = [traj.iloc[i, dimx] for traj in history]
                else:
                    xs = [0 for traj in history]
                if dimy != -1:
                    ys = [traj.iloc[i, dimy] for traj in history]
                else:
                    ys = [0 for traj in history]
                if dimz != -1:
                    zs = [traj.iloc[i, dimz] for traj in history]
                else:
                    zs = [0 for traj in history]

                list_x.extend(xs)
                list_y.extend(ys)
                list_z.extend(zs)
                list_color.extend(xs)
                list_x.append(None)
                list_y.append(None)
                list_z.append(None)
                list_color.append('#FFFFFF')

            history_traces.append(
                go.Scatter3d(
                    x=list_x,
                    y=list_y,
                    z=list_z,
                    mode='lines',
                    opacity=opacity,
                    line=dict(
                        color=list_color,
                        colorscale=cm,
                        width=1,
                        showscale=False,
                    ),
                    name='Projection traces',
                )
            )

        # highlight selected points
        selected_trace = []
        # Commented now because of colorscale issue. May still be useful if that is fixed (good to show in trace names).
        index = traj.index
        for _, d in enumerate([selectedData1, selectedData2, selectedData3]):
            if d:
                selected_index = [p['customdata'] for p in d['points']]
            else:
                selected_index = []
            if len(selected_index) > 0:
                index = np.intersect1d(index, selected_index)

        if len(index) > 1 and len(index) != traj.shape[0]:
            inds = np.isin(traj.index, index)
            selected_trace.append(
                go.Scatter3d(
                    x=x[inds],
                    y=y[inds],
                    z=z[inds],
                    mode='markers',
                    customdata=traj.index[inds],
                    marker=dict(
                        size=dotsize,
                        symbol='circle-open',
                        color='rgba(0, 0, 0, 0.8)',
                        opacity=opacity,
                        showscale=False,
                        # colorscale=cm,
                        # showscale='show_legend' in display_value,
                        # line=dict(
                        #     color='rgba(0, 0, 0, 0.8)',
                        #     width=2
                        # ),
                    ),
                    name='Selected',
                    visible=True if 'show_selected' in display_value else 'legendonly',
                ))

        annotation_trace = []
        annotation_label_trace = []
        if 'show_annotation' in display_value:
            try:
                annotation_data_selected = annotation_data.loc[traj.index, :]
                # rule of thumb to determine whether to plot as numeric or as discrete values
                valid_inds = annotation_data_selected.loc[:, annotation_index].notnull()
                n_unique_values = len(np.unique(annotation_data_selected[valid_inds].loc[:, annotation_index]))

                if np.issubdtype(annotation_data_selected.loc[:, annotation_index].dtype, np.number) and (
                        n_unique_values > np.maximum(5, annotation_data_selected.shape[
                                                            0] / 5) or n_unique_values > 50) and annotation_type != 'categorical' and annotation_type != 'none' or annotation_type == 'numerical':
                    # display as continuous
                    if 'show_original' in display_value:
                        annotation_trace.append(
                            go.Scatter3d(
                                x=datax[~valid_inds],
                                y=datay[~valid_inds],
                                z=dataz[~valid_inds],
                                mode='markers',
                                marker=dict(
                                    size=dotsize,
                                    color='#444444',
                                    opacity=opacity * 0.3,
                                    showscale=False
                                ),
                                showlegend=False,
                                name='Empty or NA',
                            )
                        )
                        annotation_trace.append(
                            go.Scatter3d(
                                x=datax[valid_inds],
                                y=datay[valid_inds],
                                z=dataz[valid_inds],
                                mode='markers',
                                customdata=traj.index[valid_inds],
                                text=annotation_data_selected[valid_inds].loc[:, annotation_index].map(str),
                                marker=dict(
                                    color=annotation_data_selected[valid_inds].loc[:, annotation_index],
                                    colorscale=cm,
                                    size=dotsize,
                                    opacity=opacity * 0.3,
                                    showscale=True,
                                    colorbar=dict(len=0.5, yanchor='top', y=0.85),
                                ),
                                hoverinfo='text',
                                name='Original ' + annotation_index,
                            )
                        )

                    annotation_trace.append(
                        go.Scatter3d(
                            x=x[~valid_inds],
                            y=y[~valid_inds],
                            z=z[~valid_inds],
                            mode='markers',
                            customdata=traj.index[~valid_inds],
                            marker=dict(
                                size=dotsize,
                                color='#444444',
                                opacity=opacity,
                                showscale=False
                            ),
                            name='Empty or NA',
                        )
                    )
                    annotation_trace.append(
                        go.Scatter3d(
                            x=x[valid_inds],
                            y=y[valid_inds],
                            z=z[valid_inds],
                            mode='markers',
                            text=annotation_data_selected[valid_inds].loc[:, annotation_index].map(str),
                            customdata=traj.index[valid_inds],
                            marker=dict(
                                color=annotation_data_selected[valid_inds].loc[:, annotation_index],
                                colorscale=cm,
                                size=dotsize,
                                opacity=opacity,
                                showscale=True,
                                colorbar=dict(len=0.5, yanchor='top', y=0.85),
                            ),
                            hoverinfo='text',
                            name=annotation_index,
                        )
                    )
                else:
                    # display as categorical
                    if n_unique_values < 80:
                        unique_values = np.unique(annotation_data_selected[valid_inds].loc[:, annotation_index])
                        if 'show_label' in label_checklist_value:
                            annox = []
                            annoy = []
                            annoz = []
                            annotext = []
                            for i, v in enumerate(unique_values):
                                inds = np.asarray(annotation_data_selected.loc[:, annotation_index] == v)
                                annox.append(x[inds].mean())
                                annoy.append(y[inds].mean())
                                annoz.append(z[inds].mean())
                                annotext.append(str(v))

                            annotation_label_trace.append(
                                go.Scatter3d(
                                    x=annox,
                                    y=annoy,
                                    z=annoz,
                                    mode='text',
                                    text=annotext,
                                    name='Label'
                                )
                            )
                        if annotation_type != 'none':
                            if 'show_original' in display_value:
                                annotation_trace.append(
                                    go.Scatter3d(
                                        x=datax[~valid_inds],
                                        y=datay[~valid_inds],
                                        z=dataz[~valid_inds],
                                        mode='markers',
                                        marker=dict(
                                            size=dotsize,
                                            color='#444444',
                                            opacity=opacity * 0.3,
                                            showscale=False
                                        ),
                                        showlegend=False,
                                        name='Empty or NA',
                                    )
                                )
                            annotation_trace.append(
                                go.Scatter3d(
                                    x=x[~valid_inds],
                                    y=y[~valid_inds],
                                    z=z[~valid_inds],
                                    mode='markers',
                                    customdata=traj.index[~valid_inds],
                                    marker=dict(
                                        size=dotsize,
                                        color='#444444',
                                        opacity=opacity,
                                        showscale=False
                                    ),
                                    name='Empty or NA',
                                )
                            )

                            for i, v in enumerate(unique_values):
                                inds = np.asarray(annotation_data_selected.loc[:, annotation_index] == v)
                                if 'show_original' in display_value:
                                    annotation_trace.append(
                                        go.Scatter3d(
                                            x=datax[inds],
                                            y=datay[inds],
                                            z=dataz[inds],
                                            mode='markers',
                                            marker=dict(
                                                size=dotsize,
                                                color=str(v).upper() if COLORPATTERN.match(str(v)) else
                                                DEFAULT_PLOTLY_COLORS[i % 10],
                                                symbol=SYMBOLS[int(i / 10)],
                                                opacity=opacity * 0.3,
                                                showscale=False
                                            ),
                                            name='Original ' + str(v),
                                        )
                                    )

                                annotation_trace.append(
                                    go.Scatter3d(
                                        x=x[inds],
                                        y=y[inds],
                                        z=z[inds],
                                        mode='markers',
                                        customdata=traj.index[inds],
                                        marker=dict(
                                            size=dotsize,
                                            color=str(v).upper() if COLORPATTERN.match(str(v)) else
                                            DEFAULT_PLOTLY_COLORS[i % 10],
                                            symbol=SYMBOLS[int(i / 10)],
                                            opacity=opacity,
                                            showscale=False
                                        ),
                                        name=str(v),
                                    )
                                )
                    else:
                        message.append("The selected annotation column has too many categories to display.")
            except:
                pass

        # if show log density is selected, show trace color by
        logp_trace = []
        eigengap_trace = []
        if 'show_logp' in display_value or 'show_eigengap' in display_value:
            p, g, h, _ = s._density_estimate(np.asarray(traj.iloc[:, :ndim]))
            output_dict['Probablity Density'] = p
            if 'show_logp' in display_value:
                logp_trace.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        text=['%.3e' % x for x in np.log(p)],
                        mode='markers',
                        customdata=traj.index,
                        marker=dict(
                            size=dotsize,
                            color=np.log(p),
                            showscale='show_legend' in display_value,
                            colorscale=cm,
                            colorbar=dict(len=0.5, yanchor='top', y=0.85),
                        ),
                        hoverinfo='text',
                        name='Log Density',
                        visible='legendonly'
                    )
                )

            if 'show_eigengap' in display_value:
                eigengap = -(np.linalg.eigh(-h)[0][:, np.maximum(dim, 1) - 1] - np.linalg.eig(-h)[0][:,
                                                                                np.maximum(dim, 1)])
                output_dict['Eigenvalue Gap (#' + str(np.maximum(dim, 1)) + ' - #' + str(
                    np.maximum(dim, 1) + 1) + ') of the Hessian of PDF'] = eigengap
                eigengap_trace.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        text=['%.3f' % x for x in eigengap],
                        mode='markers',
                        customdata=traj.index,
                        marker=dict(
                            size=dotsize,
                            color=eigengap,
                            colorscale=cm,
                            opacity=opacity,
                            showscale='show_legend' in display_value,
                            colorbar=dict(len=0.5, yanchor='top', y=0.85),
                        ),
                        hoverinfo='text',
                        name='Eigenvalue Gap (#' + str(np.maximum(dim, 1)) + ' - #' + str(np.maximum(dim, 1) + 1) + ')',
                        visible='legendonly'

                    )
                )


        if 'show_original' in display_value:
            input_trace.append(go.Scatter3d(
                x=datax,
                y=datay,
                z=dataz,
                mode='markers',
                marker=dict(
                    size=dotsize,
                    color=c,
                    opacity=opacity * 0.3,
                    colorscale=cm,
                    showscale=False
                ),
                name='Origin',
                visible='legendonly' if len(segment_traces) > 0 or len(annotation_trace) > 0 else True
            ))

        # finally, the trajectory
        traj_traces = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                text=['%.3f' % x for x in c],
                customdata=traj.index,
                mode='markers',
                marker=dict(
                    size=dotsize,
                    color=c,
                    opacity=opacity,
                    colorscale=cm,
                    showscale='show_legend' in display_value,
                    colorbar=dict(len=0.5, yanchor='top', y=0.85),
                ),
                hoverinfo='text' if selected_gene else "x+y+z",
                name=selected_gene if selected_gene else 'Trajectory',
                visible='legendonly',
            ),
        ]
        if 'show_velocity' in display_value and with_velocity_data:
            traj_traces.append(dict(type="cone",
                                    x=x,
                                    y=y,
                                    z=z,
                                    u=u,
                                    v=v,
                                    w=w,
                                    customdata=traj.index,
                                    sizeref=10 ** conesize,
                                    sizemode='scaled',
                                    showscale=False,
                                    colorscale=[(0, '#000000'), (1, '#000000')],
                                    name='Velocity',
                                    visible='legendonly',
                                    showlegend=True,
                                    hoverinfo="x+y+z",
                                    anchor='tail',
                                    opacity=opacity * 0.1,
                                    marker=dict(
                                        size=dotsize,
                                        color=c,
                                        opacity=opacity,
                                        colorscale=cm,
                                        showscale='show_legend' in display_value,
                                        colorbar=dict(len=0.5, yanchor='top', y=0.85),
                                    ),
                                    )
                               )

        # x, y, z = select_traj(traj_copy,dimx,dimy,dimz)
        # traj_copy_traces =[
        #         go.Scatter3d(
        #             x=x,
        #             y=y,
        #             z=z,
        #             text = ['%.3f' % x for x in c],
        #             mode='markers',
        #             marker=dict(
        #                 size=dotsize,
        #                 opacity=opacity,
        #                 showscale=False,
        #                 colorbar=dict(len=0.5,yanchor='top',y=0.85),
        #             ),
        #             hoverinfo = 'text' if selected_gene else "x+y+z",
        #             name = 'Trajectory with dim 1 locked',
        #         )
        #     ]
        for cell_traces in [annotation_trace, order_trace, segment_traces, eigengap_trace,
                            logp_trace, isomap_trace, traj_traces]:
            if len(cell_traces) > 0:
                for trace in cell_traces:
                    trace['visible'] = True
                break

        for cell_traces in [annotation_trace, order_trace, segment_traces, eigengap_trace,
                            logp_trace, isomap_trace, traj_traces]:
            if len(cell_traces) == 1:
                if len(input_trace) > 0:
                    input_trace[0]['marker']['color'] = cell_traces[0]['marker']['color']
                    input_trace[0]['marker']['colorscale'] = cell_traces[0]['marker']['colorscale']
                break

        if len(isomap_trace) > 0:
            if len(selected_trace) > 0:
                selected_trace[0]['visible'] = False

        figure['data'] = traj_traces + bootstrap_traces + history_traces + mst_traces + input_trace + \
                         logp_trace + eigengap_trace + segment_traces + order_trace + \
                         knn_traces + annotation_trace + selected_trace + isomap_trace + annotation_label_trace

        if 'scene' not in figure['layout']:
            figure['layout']['scene']=dict(xaxis= go.layout.XAxis(title='x | Dim ' + str(dimx+1)),
                yaxis= go.layout.XAxis(title='y | Dim ' + str(dimy+1)),
                zaxis= go.layout.XAxis(title='z | Dim ' + str(dimz+1)),
                aspectmode = scatter3d_aspect_option,
                showlegend = True if  selected_gene or len(figure['data'])>1 else False)
        else:
            figure['layout']['scene']['xaxis']['title'] = 'x | Dim ' + str(dimx + 1)
            figure['layout']['scene']['yaxis']['title'] = 'y | Dim ' + str(dimy + 1),
            figure['layout']['scene']['zaxis']['title'] = 'z | Dim ' + str(dimz + 1),
            figure['layout']['scene']['aspectmode'] = scatter3d_aspect_option,
            figure['layout']['scene']['showlegend'] = True if selected_gene or len(figure['data']) > 1 else False
            
        if "scene.camera" in relayoutData:
            figure['layout']['scene']['camera'] = relayoutData['scene.camera']

        return figure


    def update_cell_plots(i, j):
        def callback(*selectedDatas):
            index = traj.index
            dims = selectedDatas[5:]
            relayoutData = selectedDatas[4]
            dotsize = selectedDatas[3]
            for k in range(0, 3):
                if selectedDatas[k]:
                    selected_index = [p['customdata'] for p in selectedDatas[k]['points']]
                else:
                    selected_index = []
                if len(selected_index) > 0:
                    index = np.intersect1d(index, selected_index)

            if ndim > dims[i] and ndim > dims[j] and dims[i] >= 0 and dims[j] >= 0:
                x = traj.iloc[:, dims[i]]
                y = traj.iloc[:, dims[j]]
            else:
                x = []
                y = []
            figure = {
                'data': [
                    dict({
                        'type': 'scattergl',
                        'x': x, 'y': y,  # 'text': traj.index,
                        'customdata': traj.index,
                        'text': traj.index,
                        'hoverinfo': 'text',
                        'mode': 'markers',
                        'marker': {'size': dotsize},
                        'selectedpoints': match(index, traj.index),
                        'selected': {
                            'marker': {
                                'color': CELLCOLOR
                            },
                        },
                        'unselected': {
                            'marker': {
                                'color': '#bbbbbbPP'
                            }
                        }
                    }),
                ],
                'layout': {
                    'margin': {'l': 25, 'r': 0, 'b': 20, 't': 5},
                    'dragmode': 'lasso',
                    'hovermode': 'closest',
                    'showlegend': False,
                    'paper_bgcolor': BGCOLOR,
                    'plot_bgcolor': BGCOLOR,
                    'xaxis': {'title': 'Dim ' + str(dims[i] + 1), 'automargin': True},
                    'yaxis': {'title': 'Dim ' + str(dims[j] + 1), 'automargin': True},
                }
            }
            if relayoutData:
                if 'xaxis.range[0]' in relayoutData:
                    figure['layout']['xaxis']['range'] = [
                        relayoutData['xaxis.range[0]'],
                        relayoutData['xaxis.range[1]']
                    ]
                if 'yaxis.range[0]' in relayoutData:
                    figure['layout']['yaxis']['range'] = [
                        relayoutData['yaxis.range[0]'],
                        relayoutData['yaxis.range[1]']
                    ]

            return figure

        return callback


    app.callback(
        Output('select-sample1', 'figure'),
        [Input('select-sample1', 'selectedData'),
         Input('select-sample2', 'selectedData'),
         Input('select-sample3', 'selectedData'),
         Input('dotsize-slider', 'value'),
         ],
        [State('select-sample1', 'relayoutData'),
         State('x_dropdown', 'value'),
         State('y_dropdown', 'value'),
         State('z_dropdown', 'value')]
    )(update_cell_plots(0, 1))

    app.callback(
        Output('select-sample2', 'figure'),
        [Input('select-sample2', 'selectedData'),
         Input('select-sample1', 'selectedData'),
         Input('select-sample3', 'selectedData'),
         Input('dotsize-slider', 'value'), 
         ],
        [State('select-sample2', 'relayoutData'),
         State('x_dropdown', 'value'),
         State('y_dropdown', 'value'),
         State('z_dropdown', 'value')]
    )(update_cell_plots(0, 2))

    app.callback(
        Output('select-sample3', 'figure'),
        [Input('select-sample3', 'selectedData'),
         Input('select-sample1', 'selectedData'),
         Input('select-sample2', 'selectedData'),
         Input('dotsize-slider', 'value'),
         ],
        [State('select-sample3', 'relayoutData'),
         State('x_dropdown', 'value'),
         State('y_dropdown', 'value'),
         State('z_dropdown', 'value')]
    )(update_cell_plots(1, 2))


    @app.callback(
        Output('gene-dropdown', 'options'),
        [Input('select-feature', 'selectedData'),
         Input('upload_annotation_label', 'children')])
    def update_dropdown(selectedGene, children):
        rindex = feature_data.index.values
        selected_rindex = [p['customdata'] for p in selectedGene['points'] if 'customdata' in p]
        if len(selected_rindex) > 0:
            rindex = np.intersect1d(rindex, selected_rindex)
        options = [{'label': gene, 'value': gene} for gene in np.sort(rindex)]
        return options


    @app.callback(
        Output('gene-dropdown', 'value'),
        [Input('select-feature', 'clickData'),
         Input('coexp_heatmap', 'clickData')])
    def select_dropdown(clickData, clickDataheatmap):
        if clickData != None:
            return clickData['points'][0]['customdata']
        if clickDataheatmap != None:
            return clickDataheatmap['points'][0]['y']


    current_sampleindex = 0


    @app.callback(
        Output('select-feature', 'figure'),
        [Input('select-sample1', 'selectedData'),
         Input('select-sample2', 'selectedData'),
         Input('select-sample3', 'selectedData'),
         Input('upload_feature_label', 'children'),
         Input('select-feature', 'selectedData'),
         Input('feature_plot_options', 'value')],
        [State('njobs_dropdown', 'value')])
    def update_feature_plot(selectedData1, selectedData2, selectedData3, upload_feature_label, selectedGene,
                            feature_plot_option, n_jobs):
        global current_sampleindex
        global featureplot_x
        global featureplot_y
        if not with_feature_data:
            return {
                'data': [
                ],
                'layout': {
                    'margin': {'l': 25, 'r': 0, 'b': 20, 't': 5},
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'showlegend': False,
                    'paper_bgcolor': BGCOLOR,
                    'plot_bgcolor': BGCOLOR,
                    'xaxis': {'title': 'Mean', 'automargin': True},
                    'yaxis': {
                        'title': 'SD' if 'mean_sd' == feature_plot_option else 'Average Difference (Selected - Unselected)',
                        'automargin': True},
                }
            }
        index = feature_data.columns.values
        for _, data in enumerate([selectedData1, selectedData2, selectedData3]):
            if data:
                selected_index = [p['customdata'] for p in data['points']]
            else:
                selected_index = []
            if len(selected_index) > 0:
                index = np.intersect1d(index, selected_index)

        # if no cell is selected, compute for all cells
        if len(index) == 0:
            selectind = np.arange(feature_data.shape[1])
        else:
            selectind = match(index, feature_data.columns.values)

        # compute mean and variance for selected columns
        feature_data_select = feature_data.values[:, selectind]

        # if n_jobs > 1:
        #     pool= multiprocess.Pool(n_jobs)
        #     featureplot_x = np.concatenate( pool.map(lambda data: data.mean(axis=1) , np.array_split(feature_data_select,n_jobs )) )
        #     if 'mean_sd' == feature_plot_option:
        #         featureplot_y = np.concatenate( pool.map(lambda data: (data**2).mean(axis=1) - np.asarray(data.mean(axis=1))**2, np.array_split(feature_data_select,n_jobs )) )
        #     else:
        #         if len(selectind) == feature_data.shape[1]:
        #             featureplot_y = np.zeros(feature_data.shape[1])
        #         else:
        #             featureplot_y = featureplot_x-np.concatenate( pool.map(lambda data: data.mean(axis=1), np.array_split(feature_data.values[:,np.setdiff1d(np.arange(feature_data.shape[1]), selectind)],n_jobs)))

        #         current_sampleindex = selectind
        # else:
        featureplot_x = feature_data_select.mean(axis=1)
        if 'mean_sd' == feature_plot_option:
            featureplot_y = (feature_data_select ** 2).mean(axis=1) - np.asarray(feature_data_select.mean(axis=1)) ** 2
        else:
            if len(selectind) == feature_data.shape[1]:
                featureplot_y = np.zeros(feature_data.shape[1])
            else:
                featureplot_y = featureplot_x - feature_data.values[:,
                                                np.setdiff1d(np.arange(feature_data.shape[1]), selectind)].mean(axis=1)

            current_sampleindex = selectind

        rindex = feature_data.index.values
        if selectedGene:
            selected_rindex = [p['customdata'] for p in selectedGene['points'] if 'customdata' in p]
        else:
            selected_rindex = []

        if len(selected_rindex) > 0:
            rindex = np.intersect1d(rindex, selected_rindex)
        selectrind = match(rindex, feature_data.index.values)

        if 'mean_sd' == feature_plot_option:
            top10ind = np.argsort(-featureplot_y)[:10]
        else:
            top10ind = np.argsort(-np.abs(featureplot_y))[:10]

        figure = {
            'data': [
                go.Scatter(
                    x=featureplot_x[top10ind],
                    y=featureplot_y[top10ind],
                    text=feature_data.index.values[top10ind],
                    textposition='top center',
                    customdata=feature_data.index.values[top10ind],
                    mode='text',
                    hoverinfo='text',
                    marker=dict(
                        size=9,
                        line=dict(
                            color='rgba(217, 217, 217, 0.14)',
                            width=0
                        ),
                        opacity=0.8,
                        showscale=True
                    )
                ),
                dict({
                    'type': 'scattergl',
                    'x': featureplot_x, 'y': featureplot_y,
                    'customdata': feature_data.index.values,
                    'selectedpoints': selectrind,
                    'text': feature_data.index.values,
                    'hoverinfo': 'text',
                    'mode': 'markers',
                    'marker': {'size': 9, 'color': '#1f77b4'},
                    'selected': {
                        'marker': {
                            'color': FEATURECOLOR
                        },
                    },
                    'unselected': {
                        'marker': {
                            'color': '#bbbbbb'
                        }
                    }
                }),
            ],
            'layout': {
                'margin': {'l': 25, 'r': 0, 'b': 20, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'showlegend': False,
                'paper_bgcolor': BGCOLOR,
                'plot_bgcolor': BGCOLOR,
                'xaxis': {'title': 'Mean', 'automargin': True},
                'yaxis': {
                    'title': 'SD' if 'mean_sd' == feature_plot_option else 'Average Difference (Selected - Unselected)',
                    'automargin': True},
            }
        }

        return figure


    order_inds = None


    @app.callback(
        Output('dummy2', 'children'),
        [Input('scatter_3d', 'clickData')],
        [State('coexp_heatmap', 'figure'),
         State('show-options', 'value'),
         State('heatmap_precision_options', 'value'),
         State('heatmap_reference_options', 'value'),
         State('heatmap_checklist', 'value'),
         State('networkgene-dropdown', 'value'),
         State('network_bandwidth_dropdown', 'value'),
         State('network_min_radius_dropdown', 'value'),
         State('network_n_pcs', 'value')])
    def cluster_network(clickData, figure, show_options_value, heatmap_precision_options, heatmap_reference_options,
                        heatmap_checklist_value, networkgenes, bw, min_radius, n_pcs):
        global order_inds
        if show_options_value == 'show_network_options':
            gene_is = [network_data.index.get_loc(gene) for gene in networkgenes]

            if 'cell' == heatmap_reference_options:
                data_i = data.index.get_loc(clickData['points'][0]['customdata'])
            else:
                data_i = traj.index.get_loc(clickData['points'][0]['customdata'])

            if 'cell' == heatmap_reference_options:
                if n_pcs == 0:
                    cov, effective_N = utils.locCov(data.values[:, :ndim], data.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data.values.T[:, gene_is])
                else:
                    cov, effective_N = utils.locCov(data.values[:, :ndim], data.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data_pca_z[:, :n_pcs])
                    cov = np.dot(np.dot(network_data_pca.components_[:n_pcs, gene_is].T, cov),
                                 network_data_pca.components_[:n_pcs, gene_is])
            else:
                if n_pcs == 0:
                    cov, effective_N = utils.locCov(traj.values[:, :ndim], traj.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data.values.T[:, gene_is])
                else:
                    cov, effective_N = utils.locCov(traj.values[:, :ndim], traj.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data_pca_z[:, :n_pcs])
                    cov = np.dot(np.dot(network_data_pca.components_[:n_pcs, gene_is].T, cov),
                                 network_data_pca.components_[:n_pcs, gene_is])

            if 'show_precision' == heatmap_precision_options:
                precision_chol = _compute_precision_cholesky(cov[np.newaxis, :, :], 'full')
                precision = np.dot(precision_chol[0, :, :], precision_chol[0, :, :].T)
                h = precision
            elif 'show_covariance' == heatmap_precision_options:
                h = cov
            else:
                raise ValueError

            order_inds = np.asarray(
                sch.dendrogram(sch.linkage(squareform(pdist(h)), method='average', metric='euclidean'), no_plot=True)[
                    'leaves'])
            return []
        else:
            raise PreventUpdate


    @app.callback(
        Output('dummy3', 'children'),
        [Input('reset-heatmap-order-button', 'n_clicks')])
    def reset_order_inds(dummy):
        global order_inds
        order_inds = None
        return []


    @app.callback(
        Output('effective_n', 'children'),
        [Input('coexp_heatmap', 'figure')]
    )
    def update_effective_n(dummy):
        return 'Effective sample size: ' + '{: .2f}'.format(effective_N)


    @app.callback(
        Output('coexp_heatmap', 'figure'),
        [Input('scatter_3d', 'hoverData'),
         Input('dummy2', 'children'),
         Input('dummy3', 'children'),
         Input('heatmap_precision_options', 'value'),
         Input('heatmap_reference_options', 'value'),
         Input('heatmap_checklist', 'value'),
         Input('networkgene-dropdown', 'value'),
         Input('colorscale-picker', 'colorscale')],
        [
            State('coexp_heatmap', 'figure'),
            State('ndim_dropdown', 'value'),
            State('show-options', 'value'),
            State('network_bandwidth_dropdown', 'value'),
            State('network_min_radius_dropdown', 'value'),
            State('network_n_pcs', 'value')])
    def update_network(hoverData, dummy2, dummy3, heatmap_precision_options, heatmap_reference_options,
                       heatmap_checklist_value, networkgenes, colorscale,
                       figure, ndim, show_options_value, bw, min_radius, n_pcs):
        global order_inds
        global effective_N
        cm = list(zip(BINS, colorscale))
        if show_options_value == 'show_network_options':
            gene_is = [network_data.index.get_loc(gene) for gene in networkgenes]
            if 'cell' == heatmap_reference_options:
                data_i = data.index.get_loc(hoverData['points'][0]['customdata'])
            else:
                data_i = traj.index.get_loc(hoverData['points'][0]['customdata'])

            if 'cell' == heatmap_reference_options:
                if n_pcs == 0:
                    cov, effective_N = utils.locCov(data.values[:, :ndim], data.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data.values.T[:, gene_is])
                else:
                    cov, effective_N = utils.locCov(data.values[:, :ndim], data.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data_pca_z[:, :n_pcs])
                    cov = np.dot(np.dot(network_data_pca.components_[:n_pcs, gene_is].T, cov),
                                 network_data_pca.components_[:n_pcs, gene_is])
            else:
                if n_pcs == 0:
                    cov, effective_N = utils.locCov(traj.values[:, :ndim], traj.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data.values.T[:, gene_is])
                else:
                    cov, effective_N = utils.locCov(traj.values[:, :ndim], traj.values[[data_i], :ndim], bw,
                                                         min_radius, cov_data=network_data_pca_z[:, :n_pcs])
                    cov = np.dot(np.dot(network_data_pca.components_[:n_pcs, gene_is].T, cov),
                                 network_data_pca.components_[:n_pcs, gene_is])

            if 'show_precision' == heatmap_precision_options:
                precision_chol = _compute_precision_cholesky(cov[np.newaxis, :, :], 'full')
                precision = np.dot(precision_chol[0, :, :], precision_chol[0, :, :].T)
                h = precision
            elif 'show_covariance' == heatmap_precision_options:
                h = cov
            else:
                raise ValueError

            x = networkgenes
            y = x

            if order_inds is not None and len(order_inds) == len(x):
                x = [x[i] for i in order_inds]
                y = [y[i] for i in order_inds]
                h = h[order_inds, :][:, order_inds]
            else:
                order_inds = None
            x = [str(i) for i in x]
            y = [str(i) for i in y]

            if not 'show_diagonal' in heatmap_checklist_value:
                np.fill_diagonal(h, np.nan)

            # fffigure =  ff.create_annotated_heatmap(x=x, y=y, z=h, annotation_text=[["%.2f" % x for x in row] for row in h ], colorscale='Viridis')
            # figure['data'] = fffigure['data']
            cmax = np.max(np.abs(h))
            figure['data'] = [go.Heatmap(x=x, y=y, z=h, colorscale=cm, xgap=1 if h.shape[0] <= 50 else 0,
                                         ygap=1 if h.shape[0] <= 50 else 0, zmax=cmax, zmin=-cmax)]

            if 'show_values' in heatmap_checklist_value and h.shape[0] <= 15:
                figure['data'].append(go.Scatter(x=np.repeat(x, len(x)), y=np.tile(y, len(x)),
                                                 text=["%.2f" % x for row in h for x in row], hoverinfo='skip',
                                                 mode='text', textfont=dict(color='#ffffff')))
                # figure['layout']['annotations'] = fffigure['layout']['annotations']
            else:
                figure['layout']['annotations'] = None

            figure['layout']['paper_bgcolor'] = BGCOLOR
            figure['layout']['plot_bgcolor'] = BGCOLOR
            figure['data'][0]['showscale'] = 'show_legend' in heatmap_checklist_value

            return figure
        else:
            raise PreventUpdate


    # Handle file uploaders
    @app.callback(
        Output('upload_label', 'children'),
        [
            Input('upload', 'contents'),
            Input('upload', 'filename'),
        ], [State('upload_label', 'children')])
    def upload(content, filename, label):
        global data
        global traj
        global history
        global current_content
        global s
        global message
        global output_dict
        global with_user_input_data
        global with_annotation_data
        global with_velocity_data
        global with_velocity_input_data
        if content:
            try:
                _, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                if filename.endswith('.zip'):
                    zipinput = zipfile.ZipFile(io.BytesIO())
                    # TODO
                input_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', nrows=SAMPLELIMIT + 1,
                                         index_col=0)
                input_data = input_data.iloc[:SAMPLELIMIT, :]
                input_data = input_data / np.std(input_data.iloc[:, 0])
                with_user_input_data = True
                current_content = content
                if arguments['--networkdata'] == 'input':
                    network_data = input_data.T
                    with_network_data = True
            except:
                with_user_input_data = False
                message.append("Warning: cannot read input data.")
                return ['Input (optional)' + (u" \u2713" if with_user_input_data else "")]
            try:
                with_annotation_data = np.all(annotation_data.index == data.index)
            except:
                with_annotation_data = False

            if with_velocity_input_data:
                if np.any(input_data.shape != velocity_input_data.shape) or np.any(
                        input_data.index != velocity_input_data.index):
                    with_velocity_input_data = False
                    with_velocity_data = False
                    message.append('Warning: Velocity data does not match input data.')

        return ['Input (optional)' + (u" \u2713" if with_user_input_data else "")]


    @app.callback(
        Output('upload', 'style'),
        [Input('upload_label', 'children')],
        [State('upload', 'style')]
    )
    def upload_restyle(label, style):
        style['borderStyle'] = 'solid' if with_user_input_data else 'dashed'
        return style


    @app.callback(
        Output('upload_velocity_label', 'children'),
        [
            Input('upload_velocity', 'contents'),
            Input('upload_velocity', 'filename'),
        ], [State('upload_velocity_label', 'children'),
            ])
    def upload_velocity(content, filename, label):
        global velocity_input_data
        global velocity_data
        global message
        global with_velocity_data
        global with_velocity_input_data
        if content:
            try:
                _, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                if filename.endswith('.zip'):
                    zipinput = zipfile.ZipFile(io.BytesIO())
                    # TODO
                velocity_input_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t',
                                                  nrows=SAMPLELIMIT + 1, index_col=0)
                velocity_input_data = velocity_input_data.iloc[:SAMPLELIMIT, :]
                with_velocity_input_data = True
                current_content = content
            except:
                with_velocity_input_data = False
                message.append("Warning: cannot read velocity data.")
                return ['Velocity (optional)' + (u" \u2713" if with_user_input_data else "")]

            if np.any(input_data.shape != velocity_input_data.shape) or np.any(
                    input_data.index != velocity_input_data.index):
                with_velocity_input_data = False
                message.append('Warning: Velocity data does not match input data.')

            with_velocity_data = False

        return ['Velocity (optional)' + (u" \u2713" if with_velocity_input_data else "")]


    @app.callback(
        Output('upload_velocity', 'style'),
        [Input('upload_velocity_label', 'children')],
        [State('upload_velocity', 'style')]
    )
    def upload_velocity_restyle(label, style):
        style['borderStyle'] = 'solid' if with_velocity_input_data else 'dashed'
        return style


    # These callbacks handle inconsistencies caused by uploaded data
    @app.callback(
        Output('display-checklist', 'value'),
        [Input('upload_label', 'children'),
         Input('upload_annotation_label', 'children'),
         Input('upload_velocity_label', 'children'),
         Input('dummy_cl', 'children')],
        [State('display-checklist', 'value')],
    )
    def update_disabled_value(a, b, c, cl_name, values):
        if annotation_data.shape[1] == 0:
            values = [value for value in values if value != 'show_annotation']
        if not with_velocity_data:
            values = [value for value in values if value != 'show_annotation']

        if cl_name is not None and len(cl_name) > 0 and cl_name[0] in annotation_data.columns.values:
            values = values + ['show_annotation']
        return values


    @app.callback(
        Output('display-checklist', 'options'),
        [Input('display-checklist', 'value')],
        [State('display-checklist', 'options')]
    )
    def update_disabled(a, options):
        assert options[-1]['value'] == 'show_annotation'
        options[-1]['disabled'] = annotation_data.shape[1] == 0
        assert options[-3]['value'] == 'show_velocity'
        options[-3]['disabled'] = not with_velocity_data
        return options


    @app.callback(
        Output('ndim_dropdown', 'options'),
        [Input('display-checklist', 'value'),
         Input('dummy_dr', 'children')]
    )
    def update_ndim_dropdown_options(a, dummy_dr):
        return [{'label': str(i), 'value': i} for i in range(2, data.shape[1] + 1)]


    @app.callback(
        Output('ndim_dropdown', 'value'),
        [Input('ndim_dropdown', 'options')],
        [State('ndim_dropdown', 'value')]
    )
    def update_ndim_dropdown_value(a, value):
        return value if value <= data.shape[1] else data.shape[1]


    @app.callback(
        Output('upload_feature_label', 'children'),
        [
            Input('upload_feature', 'contents'),
        ], [State('upload_feature_label', 'children')])
    def upload_feature(content, label):
        global feature_data
        global input_data
        global current_feature_content
        global message
        global with_feature_data
        global feature_data_sd
        global with_network_data
        if content:
            try:
                _, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                feature_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', nrows=SAMPLELIMIT + 1,
                                           index_col=0)
                feature_data_sd = np.std(feature_data.values, axis=1)
                feature_data = feature_data.iloc[np.argsort(-feature_data_sd), :]

                with_feature_data = True
                if arguments['--networkdata'] == 'feature':
                    network_data = feature_data
                    with_network_data = True
            except:
                with_feature_data = False
                message.append("Warning: cannot read feature data.")
                return ['Feature ' + (u" \u2713" if with_feature_data else "")]

            if with_user_input_data:
                try:
                    assert len(np.intersect1d(feature_data.columns, data.index)) == len(data.index)
                except:
                    with_feature_data = False
                    message.append("Warning: Feature file column names does not match with Input file row names.")
            else:
                input_data = feature_data.T
                if arguments['--networkdata'] == 'input':
                    network_data = input_data.T
                    with_network_data = True

            return ['Feature ' + (u" \u2713" if with_feature_data else "")]
        else:
            return label


    @app.callback(
        Output('upload_feature', 'style'),
        [Input('upload_feature_label', 'children')],
        [State('upload_feature', 'style')]
    )
    def upload_feature_restyle(label, style):
        style['borderStyle'] = 'solid' if with_feature_data else 'dashed'
        return style


    @app.callback(
        Output('upload_annotation_label', 'children'),
        [
            Input('upload_annotation', 'contents'),
        ], [State('upload_annotation_label', 'children')])
    def upload_annotation(content, label):
        global annotation_data
        global current_annotation_content
        global message
        global with_annotation_data

        if content:
            try:
                _, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                read_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', low_memory=False,
                                        nrows=SAMPLELIMIT + 1, index_col=0)
                read_data = read_data.iloc[:SAMPLELIMIT, :]
                with_annotation_data = True
            except:
                with_annotation_data = False
                message.append("Warning: cannot read annotation data.")
                return ['Annotation (optional)' + (u" \u2713" if with_annotation_data else "")]

            try:
                assert np.all(read_data.index == data.index)
                annotation_data = read_data
            except:
                with_annotation_data = False
                message.append("Warning: annotation data row names does not match with input data row names.")

            return ['Annotation (optional)' + (u" \u2713" if with_annotation_data else "")]
        else:
            return label


    @app.callback(
        Output('upload_annotation', 'style'),
        [Input('upload_annotation_label', 'children')],
        [State('upload_annotation', 'style')]
    )
    def upload_annotation_restyle(label, style):
        style['borderStyle'] = 'solid' if with_annotation_data else 'dashed'
        return style


    @app.callback(
        Output('upload_trajectory_label', 'children'),
        [
            Input('upload_trajectory', 'contents'),
        ], [State('upload_trajectory_label', 'children')])
    def upload_trajectory(content, label):
        global traj
        global history
        global with_trajectory_data
        global message
        global with_trajectory_data
        if content:
            try:
                _, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                traj = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', nrows=SAMPLELIMIT + 1, index_col=0)
                history = [traj]
                with_trajectory_data = True
            except:
                with_trajectory_data = False
                message.append("Warning: cannot read trajectory data.")
                return ['Trajectory (optional)' + (u" \u2713" if with_trajectory_data else "")]

            return ['Trajectory (optional)' + (u" \u2713" if with_trajectory_data else "")]
        else:
            return label


    @app.callback(
        Output('upload_trajectory', 'style'),
        [Input('upload_trajectory_label', 'children')],
        [State('upload_trajectory', 'style')]
    )
    def upload_trajectory_restyle(label, style):
        style['borderStyle'] = 'solid' if with_trajectory_data else 'dashed'
        return style


    @app.callback(
        Output('dummy4', 'children'),
        [Input('subsample_button', 'n_clicks')],
        [State('subsample_dropdown', 'value'),
         State('njobs_dropdown', 'value')]
    )
    def subsample(n_clicks, n_samples, n_jobs):
        global traj
        # global data
        global bootstrap_trajs
        global bootstrap_traces

        if n_clicks == None:
            return []
        else:
            ind = s.inverse_density_sampling(traj.values, n_samples, n_jobs)
            # data = data.iloc[ind,:]
            traj = traj.iloc[ind, :]

            history = [traj.copy()]
            bootstrap_trajs = []
            bootstrap_traces = []
            return []


    @app.callback(
        Output('embedding-method-options', 'children'),
        [Input('dr_method', 'value')],
    )
    def display_proj_options(dr_method):
        return [
            html.Div(children=[
                html.Label('Number of input PCs'),
                dcc.Dropdown(
                    id='dr_N_PCs',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in range(3, MAX_PCS + 1)
                    ],
                    value=DEFAULT_PCS,
                    clearable=False,
                )], style={'display': 'block' if dr_method in ['graphdr', 'diffusion_map', 'umap'] else 'none'}),
            html.Div(children=[
                html.Label('Metric'),
                dcc.Dropdown(
                    id='dr_metric_dropdown',
                    options=[
                        {'label': i, 'value': i}
                        for i in ['euclidean',
                                  'chebyshev',
                                  'canberra',
                                  'braycurtis',
                                  'mahalanobis',
                                  'seuclidean',
                                  'cosine',
                                  'correlation',
                                  'hamming',
                                  'jaccard']
                    ],
                    value='euclidean',
                    clearable=False,
                ),
            ], style={'display': 'block' if dr_method in ['graphdr', 'diffusion_map', 'umap'] else 'none'}),
            html.Div(children=[
                html.Label('Number of Neighbors'),
                dcc.Dropdown(
                    id='dr_n_neighbors_dropdown',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in range(2, 201)
                    ],
                    value=DEFAULT_DR_K,
                    clearable=False,
                ),
            ], style={'display': 'block' if dr_method in ['graphdr', 'umap'] else 'none'}),
            html.Div(children=[
                html.Label('Output Dim'),
                dcc.Dropdown(
                    id='dr_dim_dropdown',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in range(1, MAX_PCS + 1)
                    ],
                    value=3,
                    clearable=False,
                ),
            ], style={'display': 'block' if dr_method in ['umap'] else 'none'}),
            html.Div(children=[
                html.Label('Min distance'),
                dcc.Dropdown(
                    id='dr_min_dist_dropdown',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in [0.001, 0.002, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5]
                    ],
                    value=0.1,
                    clearable=False,
                ),
            ], style={'display': 'block' if dr_method in ['umap'] else 'none'}),
            html.Div(children=[
                html.Label('Regularization (nonlinearity)'),
                dcc.Dropdown(
                    id='dr_lambda_dropdown',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in
                        [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 10000.0, 100000.0]
                    ],
                    value=DEFAULT_DR_LAMBDA,
                    clearable=False,
                ),
            ], style={'display': 'block' if dr_method in ['graphdr'] else 'none'}),

        ]


    @app.callback(
        Output('cluster-method-options', 'children'),
        [Input('cl_method', 'value')],
    )
    def display_cluster_options(cl_method):
        return [
            html.Div(children=[
                html.Label('Number of Neighbors'),
                dcc.Dropdown(
                    id='cl_n_neighbors_dropdown',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in range(2, 201)
                    ],
                    value=30,
                    clearable=False,
                ),
            ], style={'display': 'block' if cl_method in ['spectral'] else 'none'}),
            html.Div(children=[
                html.Label('Number of Clusters'),
                dcc.Dropdown(
                    id='cl_n_clusters_dropdown',
                    options=[
                        {'label': str(i), 'value': i}
                        for i in range(2, 201)
                    ],
                    value=20,
                    clearable=False,
                ),
            ], style={'display': 'block' if cl_method in ['spectral', 'kmeans', 'gmm'] else 'none'}),
            html.Div(children=[
                html.Label('Bandwidth'),
                dcc.Dropdown(
                    id='cl-meanshift-bandwidth',
                    options=[
                        {'label': '{: .2f}'.format(i), 'value': i}
                        for i in np.linspace(0, 5, 101)
                    ],
                    value=0.5,
                    clearable=False,
                ),
            ], style={'display': 'block' if cl_method in ['meanshift'] else 'none'}),
        ]


    if __name__ == '__main__':
        app.run_server(debug=True, port=arguments['--port'])
