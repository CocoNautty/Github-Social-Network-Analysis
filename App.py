from dash import Dash, html, dcc, callback, Output, Input, no_update
import networkx as nx
import dash_bootstrap_components as dbc
import DataStructure as ds
import DataSampling as sp
import Visualization as vis
import random

# Load data
raw_G = ds.load_data('./sampled_data/sampled_edges.csv', './sampled_data/sampled_targets.csv', './sampled_data/sampled_features.json')
G = nx.Graph()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Github Social Network Visualization"), width=12)
    ], className="my-4"),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                # figure=vis.create_network_figure(sampled_G, nx.spring_layout(sampled_G)),
                id='network-graph',
                style={'height': '80vh'}
            ),
            width=9
        ),
        dbc.Col([
            html.H4("Sample Size"),
            html.P("Adjust the number of steps for the Reweighted Random Walk sampling algorithm"),
            # dbc.Button('Resample Graph', id='update-button', n_clicks=0, color="primary", className="mb-3"),
            dcc.Slider(
                id='sample-size-slider',
                min=100,
                max=3000,
                step=100,
                value=500,
                marks={i: str(i) for i in range(100, 5001, 500)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag',
                className="mb-3"
            )
        ], width=3)
    ])
], fluid=True)

@app.callback(
    Output('network-graph', 'figure'),
    # Input('update-button', 'n_clicks'),
    Input('sample-size-slider', 'value')
)
def update_graph(sample_size):
    start_node = random.choice(list(raw_G.nodes()))
    G = sp.reweighted_random_walk_sampling(raw_G, start_node, sample_size)

    pos = nx.spring_layout(G)

    fig = vis.create_network_figure(G, pos)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
