import pandas as pd
from ges_graph.ges_algo import ges_causal_graph
from rcd.rcd import root_causal_discovery
from epsilon_diagnosis.epsilon_diagnosis import epsilon_diagnosis_function

def add_anomaly_column(input_file, dev=0.3):
    df = pd.read_csv(input_file)
    df['anomaly'] = 0
    normal_df = df.iloc[:7].copy()
    abnormal_df = pd.DataFrame(columns=df.columns)
    for i in range(7, len(df)):
        current_row = df.iloc[i:i+1]
        last_7_avg = normal_df.iloc[-7:, 1].mean()   
        if abs(current_row.iloc[0, 1] - last_7_avg) / last_7_avg >= dev:
            df.at[i, 'anomaly'] = 1
            abnormal_df = pd.concat([abnormal_df, current_row], ignore_index=True)
        else:
            normal_df = pd.concat([normal_df, current_row], ignore_index=True)
    normal_df.to_csv('normal_dataset.csv', index=False)
    abnormal_df.to_csv('abnormal_dataset.csv', index=False)
    df.to_csv(input_file[:-4] + '_with_anomaly_label.csv', index=False)
    return df,normal_df,abnormal_df

class RCAFramework:
    def __init__(self,dataset,k):
        self.model_type = None
        self.algorithm_type1 = None
        self.algorithm_type2 = None
        self.dataset = dataset
        self.num_root_causes = k
        self.epsilon_diagnosis_type = None
    
    def select_model(self):
        print("Select RCA Model:")
        print("1. One-Phase Model")
        print("2. Two-Phase Model")
        choice = int(input("Enter your choice (1 or 2): "))
        if choice == 1:
            self.model_type = 'one-phase'
            self.select_one_phase_algorithm()
        elif choice == 2:
            self.model_type = 'two-phase'
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
            self.algorithm_type1 = 'rcd'
        elif choice == 2:
            self.algorithm_type1 = 'epsilon'
        else:
            print("Invalid choice. Please select 1 or 2.")
            self.select_one_phase_algorithm()
    
    def select_two_phase_algorithm(self):
        print("Select Algorithm for Graph Generation in Two-Phase Model:")
        print("1. GES (Greedy Equivalence Search)")
        print("2. PC (Peter and Clark Algorithm)")
        choice1 = int(input("Enter your choice (1 or 2): "))
        if choice1 == 1:
            self.algorithm_type1 = 'ges'
        elif choice1 == 2:
            self.algorithm_type1 = 'PC'
        else:
            print("Invalid choice. Please select 1 or 2.")
            self.select_two_phase_algorithm()
        
        print("Select Algorithm for Finding Root Causes in Two-Phase Model:")
        print("1. Random Walk")
        print("2. Hypothesis Testing")
        choice2 = int(input("Enter your choice (1 or 2): "))
        if choice2 == 1:
            self.algorithm_type2 = 'X'
        elif choice2 == 2:
            self.algorithm_type2 = 'Y'
        else:
            print("Invalid choice. Please select 1 or 2.")
            self.select_two_phase_algorithm()
    
    def run_rca(self):
        print("Running RCA with the following parameters:")
        print(f"Model Type: {self.model_type}")
        if self.model_type == 'one-phase':
            print(f"Algorithm Type: {self.algorithm_type1}")
            if self.algorithm_type1 == 'rcd':
                print("Implementing RCD algorithm for one-phase RCA.")
                df,normal_df,abnormal_df = add_anomaly_column(self.dataset)
                root_causal_discovery(normal_df_date=normal_df,anomalous_df_date=abnormal_df,k=self.num_root_causes)
                
            elif self.algorithm_type1 == 'epsilon':
                print("Implementing Epsilon Diagnosis algorithm for one-phase RCA.")
                print("Select Statistical Test Type:")
                print("1. ks")
                print("2. levene")
                print("3. kruskal")
                choice = int(input("Enter your choice (1 or 2 or 3): "))
                if choice == 1:
                    self.epsilon_diagnosis_type = 'ks'
                elif choice == 2:
                    self.epsilon_diagnosis_type = 'levene'
                elif choice == 3:
                    self.epsilon_diagnosis_type = 'kruskal'
                else:
                    print("Invalid choice. Please select 1 or 2 or 3.")
                df,normal_df,abnormal_df = add_anomaly_column(self.dataset)
                epsilon_diagnosis_function(df=df,test=self.epsilon_diagnosis_type,k=self.num_root_causes)

        elif self.model_type == 'two-phase':
            print(f"Algorithm Type for Graph Generation: {self.algorithm_type1}")
            print(f"Algorithm Type for Finding Root Causes: {self.algorithm_type2}")
            graph_df = None
            if self.algorithm_type1 == 'ges':
                print("Generating causal graph using GES algorithm.")
                df,normal_df,abnormal_df = add_anomaly_column(self.dataset)
                graph_df = ges_causal_graph(normal_df=normal_df)
                # print(graph_df)
                # print("Heheheheheh")
            elif self.algorithm_type1 == 'PC':
                print("Generating causal graph using PC algorithm.")
                # Implement PC logic
            
            # Implement root cause finding logic based on self.algorithm_type2
        
        print("RCA process completed.")
    
    def epsilon_diagnosis(self, dataset_file, test_type, k):
        df = pd.read_csv(dataset_file)
        epsilon_diagnosis_function(df, test_type, k)
    
if __name__ == "__main__":
    dataset = 'main_dataset.csv'
    k = 5
    rca = RCAFramework(dataset,k)
    rca.select_model()
    rca.run_rca()
