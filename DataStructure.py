import networkx as nx
import pandas as pd
import json

def load_edges(path) -> pd.DataFrame:
    """
    Load edges from a csv file

    Parameters
    ----------
    path : str
        Path to the csv file

    Returns
    -------
    pd.DataFrame
        Dataframe containing the edges
    """
    return pd.read_csv(path, names=['source', 'target'], header=0)

def load_targets(path) -> pd.DataFrame:
    """
    Load targets from a csv file

    Parameters
    ----------
    path : str
        Path to the csv file

    Returns
    -------
    pd.DataFrame
        Dataframe containing the targets
    """
    return pd.read_csv(path)

def load_features(path) -> json:
    """
    Load features from a json file

    Parameters
    ----------
    path : str
        Path to the json file

    Returns
    -------
    json
        Dictionary containing the features
    """
    with open(path, 'r') as f:
        return json.load(f)

def load_graph(edges: pd.DataFrame, targets: pd.DataFrame, features: json) -> nx.Graph:
    """
    Load a graph from the edges, targets and features

    Parameters
    ----------
    edges : pd.DataFrame
        Dataframe containing the edges
    targets : pd.DataFrame
        Dataframe containing the targets
    features : json
        Dictionary containing the features

    Returns
    -------
    nx.Graph
        Graph containing the edges, targets and features
    """
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'])
    for node, attrs in features.items():
        G.nodes[int(node)]['features'] = attrs
    for _, row in targets.iterrows():
        G.nodes[row['id']]['target'] = row['ml_target']
        G.nodes[row['id']]['name'] = row['name']
    return G

def load_data(edges_path, targets_path, features_path) -> nx.Graph:
    """
    Load a graph from the edges, targets and features

    Parameters
    ----------
    edges_path : str
        Path to the csv file containing the edges
    targets_path : str
        Path to the csv file containing the targets
    features_path : str
        Path to the json file containing the features

    Returns
    -------
    nx.Graph
        Graph containing the edges, targets and features
    """
    edges = load_edges(edges_path)
    targets = load_targets(targets_path)
    features = load_features(features_path)
    return load_graph(edges, targets, features)

if __name__ == '__main__':
    G = load_data('./raw_data/musae_git_edges.csv', './raw_data/musae_git_target.csv', './raw_data/musae_git_features.json')
    print(G.nodes[0]['features'])