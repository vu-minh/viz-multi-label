# Visualize the multi-label dataset.
# vu-minh, 2018-07-12

import json
from functools import partial
from itertools import combinations

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from server import app, cache


import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE

from common.dataset import dataset


@cache.memoize()
def get_data_with_all_labels(dataset_name):
    dataset.set_data_home("./data")
    return dataset.load_dataset_multi_label(dataset_name)


@cache.memoize()
def get_all_label_names(dataset_name):
    (data, multi_labels) = get_data_with_all_labels(dataset_name)
    return multi_labels  # [{"target": [], "target_names": []}]


@cache.memoize()
def get_one_label_names(dataset_name, label_name):
    multi_labels = get_all_label_names(dataset_name)
    return multi_labels[label_name]["target_names"]


@cache.memoize()
def get_embedding(dataset_name, perplexity):
    data, _ = get_data_with_all_labels(dataset_name)
    data = StandardScaler().fit_transform(data)

    tsne = MulticoreTSNE(perplexity=perplexity, n_iter=1000,
                         n_iter_without_progress=1000, min_grad_norm=1e-32,
                         verbose=1)
    Z = tsne.fit_transform(data)
    return Z


@app.callback(
    Output("select-label", "options"),
    [Input("select-dataset", "value")],
)
def load_multi_label_dataset_callback(dataset_name):
    return [{'label': lbl_name, 'value': lbl_name}
            for lbl_name in get_all_label_names(dataset_name)]


@app.callback(
    Output("scatter-graph", "figure"),
    [Input("select-dataset", "value"),
     Input("select-label", "value"),
     Input("select-perplexity", "value")]
)
def load_graph_callback(dataset_name, label_name, perplexity):
    if None in [dataset_name, label_name, perplexity]:
        raise PreventUpdate
    
    label_names = get_one_label_names(dataset_name, label_name)
    print(label_names)
    
    Z = get_embedding(dataset_name, float(perplexity))
    print(Z.shape)
    
    traces = []
    for name in np.unique(label_names):
        Z_by_label = Z[label_names == name]
        traces.append(go.Scatter(
            x=Z_by_label[:, 0],
            y=Z_by_label[:, 1],
            text=name,
            mode='markers',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=name,
        ))
    print(len(traces))
    
    return {
        'data': traces,
        'layout': go.Layout(
            autosize=True,
            width=800,
            height=500,
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest'
        )
    }

