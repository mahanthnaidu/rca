import warnings
import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_1samp, ttest_ind, ttest_ind_from_stats, ttest_rel, chisquare, power_divergence, 
    kstest, ks_1samp, ks_2samp, epps_singleton_2samp, mannwhitneyu, ranksums, wilcoxon, 
    kruskal, friedmanchisquare, brunnermunzel, combine_pvalues, chi2_contingency, levene, 
    anderson_ksamp, shapiro, bartlett
)

warnings.filterwarnings('ignore')

def epsilon_diagnosis(normal_df, abnormal_df, test='ks'):
    root_causes = {}
    metrics = normal_df.columns
    
    for metric in metrics:
        normal_values = normal_df[metric].values
        abnormal_values = abnormal_df[metric].values
        
        if test == 'ks':
            statistic, p_value = kstest(normal_values, abnormal_values)
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
            p_value = ad_result.significance_level
        elif test == 'shapiro':
            _, p_value = shapiro(normal_values)
        elif test == 'wilcoxon':
            statistic, p_value = wilcoxon(normal_values, abnormal_values)
        elif test == 'kruskal':
            statistic, p_value = kruskal(normal_values, abnormal_values)
        elif test == 'bartlett':
            statistic, p_value = bartlett(normal_values, abnormal_values)
        elif test == 'ttest_1samp':
            statistic, p_value = ttest_1samp(normal_values, np.mean(abnormal_values))
        elif test == 'ttest_rel':
            statistic, p_value = ttest_rel(normal_values, abnormal_values)
        elif test == 'ttest_ind_from_stats':
            mean1, std1, nobs1 = np.mean(normal_values), np.std(normal_values), len(normal_values)
            mean2, std2, nobs2 = np.mean(abnormal_values), np.std(abnormal_values), len(abnormal_values)
            statistic, p_value = ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
        elif test == 'chisquare':
            statistic, p_value = chisquare(normal_values, abnormal_values)
        elif test == 'power_divergence':
            statistic, p_value = power_divergence(normal_values, abnormal_values)
        elif test == 'ks_1samp':
            statistic, p_value = ks_1samp(normal_values, abnormal_values)
        elif test == 'ks_2samp':
            statistic, p_value = ks_2samp(normal_values, abnormal_values)
        elif test == 'epps_singleton_2samp':
            statistic, p_value = epps_singleton_2samp(normal_values, abnormal_values)
        elif test == 'ranksums':
            statistic, p_value = ranksums(normal_values, abnormal_values)
        elif test == 'friedmanchisquare':
            statistic, p_value = friedmanchisquare(normal_values, abnormal_values)
        elif test == 'brunnermunzel':
            statistic, p_value = brunnermunzel(normal_values, abnormal_values)
        elif test == 'combine_pvalues':
            # For combine_pvalues, we need a list of p-values from different tests
            _, p_values = zip(
                *[kstest(normal_values, abnormal_values),
                  ttest_ind(normal_values, abnormal_values),
                  mannwhitneyu(normal_values, abnormal_values)]
            )
            statistic, p_value = combine_pvalues(p_values)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        root_causes[metric] = p_value
    
    return root_causes

def compute_significance_scores(p_values):
    p_values = np.array(p_values)
    p_values[p_values == 0] = np.finfo(float).eps
    significance_scores = 1 / p_values
    return significance_scores

def normalize_scores(scores):
    total = np.sum(scores)
    if total == 0:
        return np.zeros_like(scores)
    normalized_scores = scores / total
    return normalized_scores

def allocate_changes(overall_change, weights):
    changes = overall_change * weights
    return changes

def quantify_root_causes(root_causes, overall_change):
    root_causes_dict = {metric: p_value for metric, p_value in root_causes}
    
    p_values = list(root_causes_dict.values())
    significance_scores = compute_significance_scores(p_values)
    weights = normalize_scores(significance_scores)
    changes = allocate_changes(overall_change, weights)
    
    quantified_root_causes = {}
    for i, (metric, _) in enumerate(root_causes):
        change = changes[i]
        quantified_root_causes[metric] = (change, weights[i] * 100)
    
    return quantified_root_causes

def epsilon_diagnosis_function(df, test='ks', k=20, dev=[]):
    normal_df = df[df['anomaly'] == 0].drop(columns=['anomaly', 'dt_hr'])
    abnormal_df = df[df['anomaly'] == 1]
    dt_hr_df = abnormal_df[['dt_hr']]
    abnormal_df = abnormal_df.drop(columns=['anomaly', 'dt_hr'])
    normal_df = normal_df.drop(columns=[normal_df.columns[0]])
    abnormal_df = abnormal_df.drop(columns=[abnormal_df.columns[0]])
    
    results = []
    for i in range(len(abnormal_df)):
        current_abnormal_row = abnormal_df.iloc[[i]]
        root_causes = epsilon_diagnosis(normal_df, current_abnormal_row, test=test)
        
        sorted_root_causes = sorted(root_causes.items(), key=lambda item: item[1])
        overall_change = dev[i] * 100.0  
        quantified_root_causes = quantify_root_causes(sorted_root_causes[:k], overall_change)
        # print(overall_change)
        results.append((i, quantified_root_causes))
        
        print(f"Date-Hour: {dt_hr_df['dt_hr'].iloc[i]},row {i} top {k} root causes are:")
        for metric, (value, percent) in quantified_root_causes.items():
            print(f"  Metric: {metric}, Quantified Cause: {value} ({percent}%)")
    
    return results
