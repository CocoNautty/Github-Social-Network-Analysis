Github Social Network Analysis Project

Description:
This project involves analyzing a large social network of GitHub developers. The data was collected from the public API in June 2019. Nodes represent developers who have starred at least 10 repositories, and edges represent mutual follower relationships between them. The vertex features are extracted based on the location, repositories starred, employer, and e-mail address. The primary task is binary node classification, predicting whether a GitHub user is a web or machine learning developer based on their job title.

Data source:
https://snap.stanford.edu/data/github-social.html

Project Structure:
- raw_data/
  - musae_git_edges.csv: Contains the edges of the graph
  - musae_git_target.csv: Contains the target labels for the nodes
  - musae_git_features.json: Contains the features for the nodes
- sampled_data/: Contains the sampled data generated by the sampling scripts
- DataStructure.py: Contains functions to load edges, targets, features, and construct the graph
- DataSampling.py: Contains functions for sampling the graph and saving the sampled data
- Visualization.py: Contains functions to create visualizations of the graph
- App.py: Contains the Dash application for visualizing the GitHub social network

Usage:
1. Load the data and construct the graph:

    import DataStructure as ds
    G = ds.load_data('./raw_data/musae_git_edges.csv', './raw_data/musae_git_target.csv', './raw_data/musae_git_features.json')

2. Perform random walking sampling:

    import DataSampling as sp
    sampled_G = sp.simple_random_walk_sampling(G, start_node, num_steps)

3. Save the sampled data:

    sp.save_edges(sampled_G, './sampled_data/sampled_edges.csv')
    sp.save_targets(sampled_G, './sampled_data/sampled_targets.csv')
    sp.save_features(sampled_G, './sampled_data/sampled_features.json')

4. Run the Dash application:

    python App.py

Make sure to install the required packages before running the scripts:

    pip install -r requirements.txt