# dash app to show the visualization of a multi-label dataset
# vu-minh 20190712

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from server import app, server


# import the separate callbacks
import viz_multi_label_app_callback


select_dataset = html.Div([
    html.Label("Dataset:"),
    dcc.Dropdown(
        id="select-dataset",
        options=[
            {'label': "Automobile", 'value': "Automobile_transformed"},
        ],
        value="Automobile_transformed",
    ),
])


select_label = html.Div([
    html.Label("Target label:"),
    dcc.Dropdown(
        id="select-label",
        options=[
            {'label': "", 'value': ""}
        ],
    ),
])


select_perplexity = html.Div([
    html.Label("Perplexity"),
    dcc.Dropdown(
        id="select-perplexity",
        options=[{'label': p, 'value': p}
                 for p in [5, 10, 20, 30, 50, 100]],
        value=30,
    )
])


scatter_graph = dcc.Graph(id="scatter-graph")


app.layout = dbc.Container([
    dbc.Row([
        html.Div(["Multi-label dataset visualization"])
    ]),
    dbc.Row([
        dbc.Col([select_dataset, select_label, select_perplexity], md=4),
        dbc.Col([scatter_graph], md=6),
        dbc.Col([], md=2),
    ]),
], fluid=True)


if __name__ == "__main__":
    import better_exceptions
    better_exceptions.MAX_LENGTH = None

    app.run_server(debug=True, threaded=True, host="0.0.0.0", processes=1)
