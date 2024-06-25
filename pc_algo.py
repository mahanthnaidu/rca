import pandas as pd
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ConstraintBased.PC import pc

def process_adjacency_matrix(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == -1:
                adjacency_matrix[j, i] = 0
                adjacency_matrix[i, j] = 1
            elif adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == 0:
                adjacency_matrix[j, i] = 1
            elif adjacency_matrix[i, j] == 0 and adjacency_matrix[j, i] == -1:
                adjacency_matrix[i, j] = 1              
    adjacency_matrix[adjacency_matrix == -1] = 0
    return adjacency_matrix

def print_adjacency_matrix(graph, column_names):
    adjacency_matrix = graph
    processed_matrix = process_adjacency_matrix(adjacency_matrix)
    df = pd.DataFrame(processed_matrix, index=column_names, columns=column_names)
    return df

# “fisherz” “chisq” “gsq” “kci” “mv_fisherz”
def run_pc_algorithm(data, alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2, mvpc=False, correction_name='MV_Crtn_Fisher_Z', verbose=False, show_progress=True):
    """ Run the PC algorithm with the specified parameters """
    data_values = data.values
    cg = pc(data_values, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, mvpc=mvpc, correction_name=correction_name, verbose=verbose, show_progress=show_progress)
    return cg

def visualize_causal_graph(cg, variable_names, output_file='causal_graph.png'):
    """ Visualize the causal graph using pydot and save it as a PNG file """
    pyd = GraphUtils.to_pydot(cg.G, labels=variable_names)
    pyd.write_png(output_file)

def pc_causal_graph(normal_df):
    del_columns = [normal_df.columns[0],'anomaly']
    normal_df = normal_df.drop(columns = del_columns)
    variable_names = list(normal_df.columns)
    indep_test = 'fisherz'
    # “fisherz” “chisq” “gsq” “kci” “mv_fisherz”
    cg = run_pc_algorithm(data=normal_df,indep_test=indep_test)
    output_file = 'pc_causal_graph.png'
    visualize_causal_graph(cg=cg, variable_names=variable_names,output_file=output_file)
    graph_df = print_adjacency_matrix(cg.G.graph,variable_names)
    return graph_df
