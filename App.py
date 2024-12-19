from dash import Dash, html, dcc, callback_context, Output, Input, no_update
import networkx as nx
import dash_bootstrap_components as dbc
import DataStructure as ds
import DataSampling as sp
import Visualization as vis
import random
import numpy as np
import community as community_louvain

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
            ),
            html.H4("Community Detection"),
            dbc.Button('Louvain', id='louvain-button', n_clicks=0, color="primary", className="mb-3"),
        ], width=3)
    ])
], fluid=True)

@app.callback(
    Output('network-graph', 'figure'),
    # Input('update-button', 'n_clicks'),
    Input('sample-size-slider', 'value'),
    Input('louvain-button', 'n_clicks'),
)
def update_graph(sample_size, louvain_clicks):
    start_node = random.choice(list(raw_G.nodes()))
    G = sp.reweighted_random_walk_sampling(raw_G, start_node, sample_size)

    partition = community_louvain.best_partition(G)

    pos = nx.spring_layout(G)

    # Define grid size based on the number of communities
    num_communities = len(set(partition.values()))
    grid_size = int(np.ceil(np.sqrt(num_communities)))

    # Create a grid of positions for the communities
    community_positions = {}
    for i, community in enumerate(set(partition.values())):
        row = i // grid_size
        col = i % grid_size
        community_positions[community] = np.array([row, col])

    # Apply the grid-based offset to the node positions
    for node, community in partition.items():
        pos[node] += community_positions[community]

    # Normalize positions to fit within the canvas
    min_pos = np.min(list(pos.values()), axis=0)
    max_pos = np.max(list(pos.values()), axis=0)
    for node in pos:
        pos[node] = (pos[node] - min_pos) / (max_pos - min_pos)

    fig = vis.create_network_figure(G, pos, partition)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
