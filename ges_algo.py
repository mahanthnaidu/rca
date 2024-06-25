import warnings
import pandas as pd
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

warnings.filterwarnings('ignore')

def build_causal_graph(data, score_func='local_score_BIC', maxP=None, parameters=None):
    print('g')
    record = ges(data, score_func=score_func, maxP=maxP, parameters=parameters)
    print('h')
    return record

def save_graph_as_png(record, column_names, output_path='causal_graph.png'):
    pyd = GraphUtils.to_pydot(record['G'], labels=column_names)
    pyd.write_png(output_path)
    return

def process_adjacency_matrix(adjacency_matrix):
    adjacency_matrix[adjacency_matrix == -1] = 0
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == -1:
                adjacency_matrix[j, i] = 0
    return adjacency_matrix

def print_adjacency_matrix(record, column_names):
    adjacency_matrix = record['G'].graph
    print('e')
    processed_matrix = process_adjacency_matrix(adjacency_matrix)
    print('f')
    df = pd.DataFrame(processed_matrix, index=column_names, columns=column_names)
    return df

def main_func(normal_df, score_func='local_score_BIC', maxP=3, output_path='causal_graph.png',parameters=None):
    data = normal_df.values
    column_names = normal_df.columns
    print('b')
    record = build_causal_graph(data, score_func=score_func, maxP=maxP, parameters=parameters)
    print('c')
    save_graph_as_png(record, column_names, output_path=output_path)
    print('d')
    adjacency_df = print_adjacency_matrix(record, column_names)
    return adjacency_df
    

def ges_causal_graph(normal_df):
    score_func = 'local_score_BIC'
    # choices=['local_score_BIC', 'local_score_BDeu', 'local_score_cv_general', 'local_score_marginal_general', 'local_score_cv_multi', 'local_score_marginal_multi']
    output_path = '../ges_causal_graph.png'
    del_columns = [normal_df.columns[0],'anomaly']
    normal_df = normal_df.drop(columns = del_columns)
    maxP = None
    params = None
    print('a')
    graph_df = main_func(normal_df=normal_df,score_func=score_func,maxP=maxP,output_path=output_path,parameters=params)
    return graph_df
