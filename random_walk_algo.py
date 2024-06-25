import warnings
import numpy as np
import pandas as pd
from pyrca.analyzers.random_walk import RandomWalkConfig, RandomWalk

warnings.filterwarnings('ignore')

class CustomRandomWalk(RandomWalk):
    def _random_walk(self, graph, start, num_steps, num_repeats, random_seed=0):
        if random_seed is not None:
            np.random.seed(random_seed)
        scores = {}
        for _ in range(num_repeats):
            node = start
            for _ in range(num_steps):
                probs = graph[node]["probs"]
                probs = np.nan_to_num(probs)
                probs = probs / sum(probs)
                index = np.random.choice(list(range(len(probs))), p=probs)
                node = graph[node]["nodes"][index]
                scores[node] = scores.get(node, 0) + 1
        for node in scores:
            scores[node] /= num_repeats * num_steps
        return scores
    
    def _build_weighted_graph(self, df, anomalies, rho):
        metrics = df.columns
        node_weights = self._node_weight(df, anomalies)
        self_weights = self._self_weight(df, anomalies, node_weights)
        node_ws = {metric: max(values) for metric, values in node_weights.items()}
        self_ws = {metric: max(values) for metric, values in self_weights.items()}
        graph = {m: {"nodes": [], "weights": [], "probs": None} for m in metrics}

        for metric in metrics:
            for p in self.graph.predecessors(metric):
                graph[metric]["nodes"].append(p)
                graph[metric]["weights"].append(node_ws[p])
            for p in self.graph.successors(metric):
                graph[metric]["nodes"].append(p)
                graph[metric]["weights"].append(node_ws[p] * rho)
            graph[metric]["nodes"].append(metric)
            graph[metric]["weights"].append(self_ws[metric])

        for metric in graph.keys():
            w = np.array(graph[metric]["weights"])
            w = np.where(w == 0, 1e-10, w)
            graph[metric]["probs"] = w / sum(w)
        return graph

def run_random_walk_rca(graph_df, anomalous_metrics, abnormal_df,root_cause_top_k):
    config = RandomWalkConfig(graph=graph_df)
    rw_rca = CustomRandomWalk(config)
    for index, row in abnormal_df.iterrows():
        single_row_df = pd.DataFrame([row])
        root_causes = rw_rca.find_root_causes(anomalous_metrics=anomalous_metrics, df=single_row_df,root_cause_top_k=root_cause_top_k)
        print(f"Random Walk RCA Root Causes for row {index} are:")
        for node in root_causes.root_cause_nodes:
            print(node)

def random_walk_func(graph_df,abnormal_df,root_cause_top_k):
    del_columns = [abnormal_df.columns[0],'anomaly']
    abnormal_df = abnormal_df.drop(columns = del_columns)
    anomalous_metrics = [abnormal_df.columns[0]]
    run_random_walk_rca(graph_df=graph_df,anomalous_metrics=anomalous_metrics,abnormal_df=abnormal_df,root_cause_top_k=root_cause_top_k)
    return 
    