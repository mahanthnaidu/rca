import warnings
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
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

def print_adjacency_matrix(graph, column_names):
    adjacency_matrix = graph
    processed_matrix = process_adjacency_matrix(adjacency_matrix)
    df = pd.DataFrame(processed_matrix, index=column_names, columns=column_names)
    return df

def run_pc_algorithm(data, alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name='MV_Crtn_Fisher_Z', verbose=False, show_progress=True):
    data_values = data.values
    cg = pc(data_values, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, mvpc=mvpc, correction_name=correction_name, verbose=verbose, show_progress=show_progress)
    return cg

def visualize_processed_graph(processed_matrix, variable_names, output_file='pc_causal_graph.png'):
    G = nx.DiGraph()

    for var in variable_names:
        G.add_node(var)

    for i, var1 in enumerate(variable_names):
        for j, var2 in enumerate(variable_names):
            if processed_matrix[i, j] == 1:
                G.add_edge(var1, var2)

    A = nx.nx_agraph.to_agraph(G)
    A.layout(prog='dot')

    A.draw(output_file)

def parse_background_knowledge(background_file, variable_names):
    forbidden_edges = []
    required_edges = []
    with open(background_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if parts[0].strip() == 'forbidden':
                forbidden_edges.append((variable_names.index(parts[1].strip()), variable_names.index(parts[2].strip())))
            elif parts[0].strip() == 'required':
                required_edges.append((variable_names.index(parts[1].strip()), variable_names.index(parts[2].strip())))
    return forbidden_edges, required_edges

def pc_causal_graph(normal_df, background_file):
    del_columns = [normal_df.columns[0], 'anomaly']
    normal_df = normal_df.drop(columns=del_columns)
    variable_names = list(normal_df.columns)
    indep_test = 'chisq'
    
    forbidden_edges, required_edges = parse_background_knowledge(background_file, variable_names)
    
    cg = run_pc_algorithm(data=normal_df, indep_test=indep_test)
    processed_matrix = process_adjacency_matrix(cg.G.graph, forbidden_edges, required_edges)
    
    output_file = 'pc_causal_graph.png'
    visualize_processed_graph(processed_matrix=processed_matrix, variable_names=variable_names, output_file=output_file)
    graph_df = pd.DataFrame(processed_matrix, index=variable_names, columns=variable_names)
    
    return graph_df
