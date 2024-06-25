import warnings
import pandas as pd
from pyrca.analyzers.ht import HT, HTConfig

warnings.filterwarnings('ignore')

def run_ht_rca(graph_df, anomalous_metrics, abnormal_df,root_cause_top_k):
    config = HTConfig(graph=graph_df, aggregator='max')
    ht_rca = HT(config)
    ht_rca.train(graph_df)
    for index, row in abnormal_df.iterrows():
        single_row_df = pd.DataFrame([row])
        root_causes = ht_rca.find_root_causes(abnormal_df=single_row_df, anomalous_metrics=anomalous_metrics,root_cause_top_k=root_cause_top_k)
        print(f"HT RCA Root Causes for row {index} are:")
        for node in root_causes.root_cause_nodes:
            print(node)

def ht_rca_func(graph_df,abnormal_df,root_cause_top_k):
    del_columns = [abnormal_df.columns[0],'anomaly']
    abnormal_df = abnormal_df.drop(columns = del_columns)
    anomalous_metrics = abnormal_df.columns[0]
    run_ht_rca(graph_df=graph_df,anomalous_metrics=anomalous_metrics,abnormal_df=abnormal_df,root_cause_top_k=root_cause_top_k)
    return 
