import warnings
import pandas as pd
import networkx as nx
from causallearn.search.ScoreBased.GES import ges
# import pygraphviz as pgv
# import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout

warnings.filterwarnings('ignore')

def process_adjacency_matrix(adjacency_matrix, forbidden_edges, required_edges):
    n = adjacency_matrix.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == -1:
                adjacency_matrix[j, i] = 0
                adjacency_matrix[i, j] = 0
            elif adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == 0:
                adjacency_matrix[j, i] = 1
            elif adjacency_matrix[i, j] == 0 and adjacency_matrix[j, i] == -1:
                adjacency_matrix[i, j] = 1
    adjacency_matrix[adjacency_matrix == -1] = 0

    # Apply forbidden and required edges
    for (i, j) in forbidden_edges:
        adjacency_matrix[i, j] = 0
    for (i, j) in required_edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 0

    return adjacency_matrix

def build_causal_graph(data, score_func='local_score_BIC', maxP=None, parameters=None):
    record = ges(data, score_func=score_func, maxP=maxP, parameters=parameters)
    adjacency_matrix = record['G'].graph
    return adjacency_matrix

def save_graph_as_png(adjacency_matrix, column_names, output_path='causal_graph.png'):
    # processed_matrix = process_adjacency_matrix(adjacency_matrix, [], [])  # Placeholder for forbidden and required edges
    G = nx.DiGraph()

    for var in column_names:
        G.add_node(var)

    for i, var1 in enumerate(column_names):
        for j, var2 in enumerate(column_names):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(var1, var2)

    A = nx.nx_agraph.to_agraph(G)
    A.layout(prog='dot')

    A.draw(output_path)

def parse_background_knowledge(file_path, variable_names):
    forbidden_edges = []
    required_edges = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if parts[0].strip() == 'forbidden':
                forbidden_edges.append((variable_names.index(parts[1].strip()), variable_names.index(parts[2].strip())))
            elif parts[0].strip() == 'required':
                required_edges.append((variable_names.index(parts[1].strip()), variable_names.index(parts[2].strip())))
    return forbidden_edges, required_edges

def ges_causal_graph(normal_df, background_file):
    del_columns = [normal_df.columns[0], 'anomaly']
    normal_df = normal_df.drop(columns=del_columns)
    variable_names = list(normal_df.columns)
    # choices=['local_score_BIC', 'local_score_BDeu', 'local_score_cv_general', 'local_score_marginal_general', 'local_score_cv_multi', 'local_score_marginal_multi']
    score_func = 'local_score_BIC'
    output_file = 'ges_causal_graph.png'

    forbidden_edges, required_edges = parse_background_knowledge(background_file, variable_names)
    
    data = normal_df.values

    adjacency_matrix = build_causal_graph(data, score_func=score_func)
    processed_matrix = process_adjacency_matrix(adjacency_matrix, forbidden_edges, required_edges)
    save_graph_as_png(processed_matrix, variable_names, output_path=output_file)

    graph_df = pd.DataFrame(processed_matrix, index=variable_names, columns=variable_names)

    return graph_df
