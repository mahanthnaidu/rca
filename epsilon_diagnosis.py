import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, chi2_contingency, levene, anderson_ksamp, shapiro, wilcoxon, kruskal, bartlett

def epsilon_diagnosis(normal_df, abnormal_df, test='ks'):
    root_causes = {}
    metrics = normal_df.columns
    
    for metric in metrics:
        normal_values = normal_df[metric].values
        abnormal_values = abnormal_df[metric].values
        
        if test == 'ks':
            statistic, p_value = ks_2samp(normal_values, abnormal_values)
        elif test == 't':
            statistic, p_value = ttest_ind(normal_values, abnormal_values)
        elif test == 'mw':
            statistic, p_value = mannwhitneyu(normal_values, abnormal_values)
        elif test == 'chi2':
            contingency_table = pd.crosstab(normal_values, abnormal_values)
            statistic, p_value, _, _ = chi2_contingency(contingency_table)
        elif test == 'levene':
            statistic, p_value = levene(normal_values, abnormal_values)
        elif test == 'ad':
            ad_result = anderson_ksamp([normal_values, abnormal_values])
            p_value = ad_result.significance_level[0]
        elif test == 'shapiro':
            _, p_value = shapiro(normal_values)
        elif test == 'wilcoxon':
            statistic, p_value = wilcoxon(normal_values, abnormal_values)
        elif test == 'kruskal':
            statistic, p_value = kruskal(normal_values, abnormal_values)
        elif test == 'bartlett':
            statistic, p_value = bartlett(normal_values, abnormal_values)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        root_causes[metric] = p_value
    
    return root_causes

def epsilon_diagnosis_function(df, test='ks', k=5):
    normal_df = df[df['anomaly'] == 0].drop(columns=['anomaly','Date'])
    abnormal_df = df[df['anomaly'] == 1].drop(columns=['anomaly','Date'])
    normal_df = normal_df.drop(columns=[normal_df.columns[0]])
    abnormal_df = abnormal_df.drop(columns=[abnormal_df.columns[0]])

    # normal_df.to_csv('normal_data.csv', index=False)
    # abnormal_df.to_csv('abnormal_data.csv', index=False)
    
    for i in range(len(abnormal_df)):
        current_abnormal_row = abnormal_df.iloc[[i]]
        root_causes = epsilon_diagnosis(normal_df, current_abnormal_row, test=test)
        
        sorted_root_causes = sorted(root_causes.items(), key=lambda item: item[1])
        
        print(f"Row {i} root top {k} causes are:")
        for metric, value in sorted_root_causes[:k]:
            print(f"  Metric: {metric}, Root Cause Value: {value}")