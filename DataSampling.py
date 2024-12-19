import random
import pandas as pd
import DataStructure as ds
import json

def simple_random_walk_sampling(G, start_node, num_steps):
    """
    Perform a random walk on the graph and return the subgraph induced by the visited nodes

    Parameters
    ----------
    G : nx.Graph
        Graph to sample from
    start_node : int
        Node to start the random walk from
    num_steps : int
        Number of steps to take

    Returns
    -------
    nx.Graph
        Subgraph induced by the visited nodes
    """
    current_node = start_node
    visited_nodes = set([current_node])

    for _ in range(num_steps):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        visited_nodes.add(current_node)

    return G.subgraph(visited_nodes)

def reweighted_random_walk_sampling(G, start_node, num_steps):
    """
    Perform a random walk on the graph and return the subgraph induced by the visited nodes

    Parameters
    ----------
    G : nx.Graph
        Graph to sample from
    start_node : int
        Node to start the random walk from
    num_steps : int
        Number of steps to take

    Returns
    -------
    nx.Graph
        Subgraph induced by the visited nodes
    """
    current_node = start_node
    visited_nodes = set([current_node])

    for _ in range(num_steps):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        weights = [G.degree(n) for n in neighbors]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        current_node = random.choices(neighbors, weights=probabilities)[0]
        visited_nodes.add(current_node)

    return G.subgraph(visited_nodes)

def save_edges(G, path):
    """
    Save the edges of a graph to a csv file

    Parameters
    ----------
    G : nx.Graph
        Graph to save
    path : str
        Path to the csv file
    """
    edges = [(source, target) for source, target in G.edges]
    pd.DataFrame(edges, columns=['source', 'target']).to_csv(path, index=False)

def save_targets(G, path):
    """
    Save the targets of a graph to a csv file

    Parameters
    ----------
    G : nx.Graph
        Graph to save
    path : str
        Path to the csv file
    """
    targets = [(node, G.nodes[node]['name'], G.nodes[node]['target']) for node in G.nodes]
    pd.DataFrame(targets, columns=['id', 'name', 'ml_target']).to_csv(path, index=False)

def save_features(G, path):
    """
    Save the features of a graph to a json file

    Parameters
    ----------
    G : nx.Graph
        Graph to save
    path : str
        Path to the json file
    """
    features = {str(node): G.nodes[node]['features'] for node in G.nodes}
    with open(path, 'w') as f:
        json.dump(features, f)

if __name__ == "__main__":
    raw_G = ds.load_data('./raw_data/musae_git_edges.csv', './raw_data/musae_git_target.csv', './raw_data/musae_git_features.json')

    start_node = random.choice(list(raw_G.nodes()))
    sampled_G = simple_random_walk_sampling(raw_G, start_node, 3000)

    save_edges(sampled_G, './sampled_data/sampled_edges.csv')
    save_targets(sampled_G, './sampled_data/sampled_targets.csv')
    save_features(sampled_G, './sampled_data/sampled_features.json')