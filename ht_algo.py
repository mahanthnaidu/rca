import warnings
import pandas as pd
from pyrca.analyzers.ht import HT, HTConfig

warnings.filterwarnings('ignore')

def run_ht_rca(graph_df, normal_df,anomalous_metrics, abnormal_df,root_cause_top_k,dt_hr_df):	
    config = HTConfig(graph=graph_df, aggregator='sum')
    ht_rca = HT(config=config)
    ht_rca.train(normal_df)
    for index, row in abnormal_df.iterrows():
        single_row_df = pd.DataFrame([row])
        root_causes = ht_rca.find_root_causes(abnormal_df=single_row_df, anomalous_metrics=anomalous_metrics,root_cause_top_k=root_cause_top_k,adjustment=True)
        print(f"Date-Hour: {dt_hr_df['dt_hr'].iloc[index]}, HT RCA Root Causes for row {index} are:")
        for node in root_causes.root_cause_nodes:
            print('Root Cause: ' + node[0] + ', Root Cause Value: ' + str(node[1]))
        # print(root_causes)

def ht_rca_func(graph_df,normal_df,abnormal_df,root_cause_top_k):
    del_columns = [abnormal_df.columns[0],'anomaly']
    dt_hr_df = abnormal_df[['dt_hr']]
    abnormal_df = abnormal_df.drop(columns = del_columns)
    normal_df = normal_df.drop(columns = del_columns)
    anomalous_metrics = str(abnormal_df.columns[0])
    # print('Anomalous Metrics : ' + str(anomalous_metrics))
    run_ht_rca(graph_df=graph_df,normal_df=normal_df,anomalous_metrics=anomalous_metrics,abnormal_df=abnormal_df,root_cause_top_k=root_cause_top_k,dt_hr_df=dt_hr_df)
    return 
