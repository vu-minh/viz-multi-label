# Visualize the multi-label dataset.
# vu-minh, 2018-07-12

import json
from functools import partial
from itertools import combinations
from itertools import cycle

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


def get_custom_group_for_label():
    return {
        'Automobile_transformed': {
            'num_of_cylinders': {
                'one': "<= 4",
                'two': "<= 4",
                'three': "<= 4",
                'four': "<= 4",
                'five': "> 4",
                'six': "> 4",
                'seven': "> 4",
                'eight': "> 4",
            }
        }
    }


@cache.memoize()
def get_data_with_all_labels(dataset_name):
    dataset.set_data_home("./data")
    return dataset.load_dataset_multi_label(dataset_name)


@cache.memoize()
def get_all_label_names(dataset_name):
    (data, multi_labels) = get_data_with_all_labels(dataset_name)
    return multi_labels  # [{"target": [], "target_names": []}]


# @cache.memoize()
def get_one_label_names(dataset_name, label_name):
    multi_labels = get_all_label_names(dataset_name)
    label_names = multi_labels[label_name]['target_names']
    custom_group_name = get_custom_group_for_label()

    if dataset_name in custom_group_name:
        if label_name in custom_group_name[dataset_name]:
            custom_rule = custom_group_name[dataset_name][label_name]
            label_names = np.array([custom_rule[s] for s in label_names])
    return label_names


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
    [Output("select-label-color", "options"),
     Output("select-label-maker", "options")],
    [Input("select-dataset", "value")],
)
def load_multi_label_dataset_callback(dataset_name):
    multi_labels = get_all_label_names(dataset_name)
    label_options = []
    for label_name, labels in multi_labels.items():
        n_unique = len(np.unique(labels['target_names']))
        label_options.append({'label': f"{label_name} ({n_unique})",
                              'value': label_name})
    return (label_options, label_options)


@app.callback(
    Output("scatter-graph", "figure"),
    [Input("select-dataset", "value"),
     Input("select-label-color", "value"),
     Input("select-label-maker", "value"),
     Input("select-perplexity", "value")]
)
def load_graph_callback(dataset_name, label_color, label_maker, perplexity):
    if None in [dataset_name, label_color, label_maker, perplexity]:
        raise PreventUpdate
    
    label_color_names = get_one_label_names(dataset_name, label_color)
    label_maker_names = get_one_label_names(dataset_name, label_maker)
    
    Z = get_embedding(dataset_name, float(perplexity))
    list_color_codes = cycle(['#1f77b4', '#ff7f0e', '#2ca02c',
                              '#d62728', '#9467bd', '#8c564b',
                              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    traces = []
    for name_color, color_code in zip(np.unique(label_color_names), list_color_codes):        
        (idx_by_color, ) = np.where(label_color_names == name_color)
        for maker_symbol, name_maker in enumerate(np.unique(label_maker_names)):
            (idx_by_marker, ) = np.where(label_maker_names == name_maker)

            idx_by_color_maker = list(set(idx_by_color.tolist()) &
                                      set(idx_by_marker.tolist()))
            Z_by_color_maker = Z[idx_by_color_maker]
            name = f"{label_color}: {name_color} <br> {label_maker}: {name_maker}"

            traces.append(go.Scatter(
                x=Z_by_color_maker[:, 0],
                y=Z_by_color_maker[:, 1],
                text=name,
                mode='markers',
                opacity=0.8,
                marker={
                    'color': 'white',
                    'size': 12,
                    'symbol': maker_symbol,
                    'line': {'width': 2.0, 'color': color_code}
                },
                name=name,
            ))

    no_axes_config = {
        'showgrid': False,
        'showline': False,
        'zeroline': False,
        'showticklabels': False,
    }

    
    return {
        'data': traces,
        'layout': go.Layout(
            autosize=True,
            width=800,
            height=500,
            xaxis=no_axes_config,
            yaxis=no_axes_config,
            # xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            # yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest'
        )
    }

