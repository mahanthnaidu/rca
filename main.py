import warnings
import pandas as pd
from bn_rca.bn_algo import bn_rca_func  
from ht_rca.ht_algo import ht_rca_func
from pc_graph.pc_algo import pc_causal_graph
from ges_graph.ges_algo import ges_causal_graph
from random_walk_rca.random_walk_algo import random_walk_func
from background_graph.causal_graph import domain_knowledge_graph
from epsilon_diagnosis.epsilon_diagnosis import epsilon_diagnosis_function

warnings.filterwarnings('ignore')

def add_anomaly_column(input_file, dev=0.07):
    df = pd.read_csv(input_file)
    df['anomaly'] = 0
    normal_df = df.iloc[:24].copy()
    abnormal_df = pd.DataFrame(columns=df.columns)
    abs_dev_percentages = []
    for i in range(24, len(df)):
        current_row = df.iloc[i:i+1]
        last_24_avg = normal_df.iloc[-24:, 1].mean()
        abs_dev_percent = (abs(current_row.iloc[0, 1] - last_24_avg) / last_24_avg) * 100
        abs_dev_percentages.append(abs_dev_percent)

        if abs_dev_percent >= dev * 100:
            df.at[i, 'anomaly'] = 1
            abnormal_df = pd.concat([abnormal_df, current_row], ignore_index=True)
        else:
            normal_df = pd.concat([normal_df, current_row], ignore_index=True)
       
    normal_df.to_csv('datasets/normal_dataset.csv', index=False)
    abnormal_df.to_csv('datasets/abnormal_dataset.csv', index=False)
    df.to_csv(input_file[:-4] + '_with_anomaly_label.csv', index=False)
    
    return df, normal_df, abnormal_df, abs_dev_percentages

class RCAFramework:
    def __init__(self, dataset, k, background_knowledge):
        self.model_type = None
        self.algorithm_type1 = None
        self.algorithm_type2 = None
        self.dataset = dataset
        self.num_root_causes = k
        self.epsilon_diagnosis_type = None
        self.background_knowledge = background_knowledge
    
    def select_model(self):
        print("Select RCA Model:")
        print("1. One-Phase Model")
        print("2. Two-Phase Model")
        choice = int(input("Enter your choice (1 or 2): "))
        if choice == 1:
            self.model_type = 'One-Phase'
            self.select_one_phase_algorithm()
        elif choice == 2:
            self.model_type = 'Two-Phase'
            self.select_two_phase_algorithm()
        else:
            print("Invalid choice. Please select 1 or 2.")
            self.select_model()
    
    def select_one_phase_algorithm(self):
        print("Select Algorithm for One-Phase Model:")
        print("1. RCD (Root Causal Discovery)")
        print("2. Epsilon Diagnosis")
        choice = int(input("Enter your choice (1 or 2): "))
        if choice == 1:
            self.algorithm_type1 = 'RCD'
        elif choice == 2:
            self.algorithm_type1 = 'Epsilon Diagnosis'
        else:
            print("Invalid choice. Please select 1 or 2.")
            self.select_one_phase_algorithm()
    
    def select_two_phase_algorithm(self):
        print("Select Algorithm for Graph Generation in Two-Phase Model:")
        print("1. Domain Knowledge Graph")
        print("2. GES (Greedy Equivalence Search)")
        print("3. PC (Peter and Clark Algorithm)")
        choice1 = int(input("Enter your choice (1, 2, or 3): "))
        if choice1 == 1:
            self.algorithm_type1 = 'Domain Knowledge Graph'
        elif choice1 == 2:
            self.algorithm_type1 = 'GES'
        elif choice1 == 3:
            self.algorithm_type1 = 'PC'
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            self.select_two_phase_algorithm()

        print("Select Algorithm for Finding Root Causes in Two-Phase Model:")
        print("1. Random Walk")
        print("2. Hypothesis Testing")
        print("3. Bayesian Network")
        choice2 = int(input("Enter your choice (1, 2, or 3): "))
        if choice2 == 1:
            self.algorithm_type2 = 'Random Walk'
        elif choice2 == 2:
            self.algorithm_type2 = 'Hypothesis Testing'
        elif choice2 == 3:
            self.algorithm_type2 = 'Bayesian Network'
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            self.select_two_phase_algorithm()
    
    def run_rca(self):
        print("Running RCA with the following parameters:")
        print(f"Model Type: {self.model_type}")
        if self.model_type == 'One-Phase':
            print(f"Algorithm Type: {self.algorithm_type1}")
            if self.algorithm_type1 == 'RCD':
                print("Implementing RCD algorithm for one-phase RCA.")
                df, normal_df, abnormal_df, abs_dev_percentages = add_anomaly_column(self.dataset)
                # root_causal_discovery(normal_df=normal_df, abnormal_df=abnormal_df, k=self.num_root_causes)
                
            elif self.algorithm_type1 == 'Epsilon Diagnosis':
                print("Implementing Epsilon Diagnosis algorithm for one-phase RCA.")
                print("Select Statistical Test Type:")
                print("1. ks")
                print("2. levene")
                print("3. ttest_ind_from_stats")
                print("4. ks_2samp")
                print("5. ranksums")
                print("6. kruskal")
                choice = int(input("Enter your choice (1, 2, 3, 4, 5, or 6): "))
                if choice == 1:
                    self.epsilon_diagnosis_type = 'ks'
                elif choice == 2:
                    self.epsilon_diagnosis_type = 'levene'
                elif choice == 3:
                    self.epsilon_diagnosis_type = 'ttest_ind_from_stats'
                elif choice == 4:
                    self.epsilon_diagnosis_type = 'ks_2samp'
                elif choice == 5:
                    self.epsilon_diagnosis_type = 'ranksums'
                elif choice == 6:
                    self.epsilon_diagnosis_type = 'kruskal'
                else:
                    print("Invalid choice. Please select 1, 2, 3, 4, 5, or 6.")
                df, normal_df, abnormal_df, abs_dev_percentages = add_anomaly_column(self.dataset)
                epsilon_diagnosis_function(df=df, test=self.epsilon_diagnosis_type, k=self.num_root_causes, dev=abs_dev_percentages)

        elif self.model_type == 'Two-Phase':
            print(f"Algorithm Type for Graph Generation: {self.algorithm_type1}")
            print(f"Algorithm Type for Finding Root Causes: {self.algorithm_type2}")
            graph_df = None
            df, normal_df, abnormal_df, abs_dev_percentages = add_anomaly_column(self.dataset)
            if self.algorithm_type1 == 'Domain Knowledge Graph':
                print("Generating causal graph using domain knowledge graph.")
                graph_df = domain_knowledge_graph(background_file=self.background_knowledge, variable_names=df.columns[1:])

            elif self.algorithm_type1 == 'GES':
                print("Generating causal graph using GES algorithm.")
                graph_df = ges_causal_graph(normal_df=normal_df, background_file=self.background_knowledge)

            elif self.algorithm_type1 == 'PC':
                print("Generating causal graph using PC algorithm.")
                graph_df = pc_causal_graph(normal_df=normal_df, background_file=self.background_knowledge)
            
            if self.algorithm_type2 == 'Random Walk':
                print("Implementing random walk algorithm for two-phase RCA.")
                random_walk_func(graph_df=graph_df, normal_df=normal_df, abnormal_df=abnormal_df, root_cause_top_k=self.num_root_causes)

            elif self.algorithm_type2 == 'Hypothesis Testing':
                print("Implementing hypothesis testing algorithm for two-phase RCA.")
                ht_rca_func(graph_df=graph_df, normal_df=normal_df, abnormal_df=abnormal_df, root_cause_top_k=self.num_root_causes)

            elif self.algorithm_type2 == 'Bayesian Network':
                print("Implementing Bayesian Network algorithm for two-phase RCA.")
                bn_rca_func(graph_df=graph_df, normal_df=normal_df, abnormal_df=abnormal_df, root_cause_top_k=self.num_root_causes)

        print("RCA process completed.")
    
if __name__ == "__main__":
    dataset = 'datasets/final_24.csv'
    background_knowledge = 'datasets/background_knowledge.csv'
    k = 10
    rca = RCAFramework(dataset, k, background_knowledge)
    rca.select_model()
    rca.run_rca()
