import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Extract information from fine_tuning_log.txt
def parse_fine_tuning_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex to find eval_accuracy and eval_macro_f1 after each epoch
    eval_acc_pattern = re.compile(r"'eval_accuracy': ([\d\.]+),")
    eval_macro_f1_pattern = re.compile(r"'eval_macro_f1': ([\d\.]+),")
    
    eval_acc = eval_acc_pattern.findall(content)
    eval_macro_f1 = eval_macro_f1_pattern.findall(content)
    
    return list(map(float, eval_acc)), list(map(float, eval_macro_f1))

# Step 2: Extract the RF and GBDT test results from combined_results.csv
def parse_combined_results(file_path):
    df = pd.read_csv(file_path)
    
    # Filter to only include rows where the dataset is 'test'
    rf_data = df[(df['method'] == 'Random Forest') & (df['dataset'] == 'test')]
    gbdt_data = df[(df['method'] == 'GBDT') & (df['dataset'] == 'test')]
    
    rf_acc = rf_data['accuracy'].values
    gbdt_acc = gbdt_data['accuracy'].values
    
    rf_f1 = rf_data['macro_f1'].values
    gbdt_f1 = gbdt_data['macro_f1'].values
    
    return rf_acc, gbdt_acc, rf_f1, gbdt_f1

# Step 3: Plotting the comparison
def plot_results(eval_acc, eval_macro_f1, rf_acc, gbdt_acc, rf_f1, gbdt_f1):
    models = ['Random Forest', 'GBDT', 'GF-6L-ft-1-ep', 'GF-6L-ft-2-ep', 'GF-6L-ft-3-ep']
    
    # Calculate means and standard deviations for RF and GBDT
    rf_acc_mean, rf_acc_std = np.mean(rf_acc), np.std(rf_acc)
    gbdt_acc_mean, gbdt_acc_std = np.mean(gbdt_acc), np.std(gbdt_acc)
    
    rf_f1_mean, rf_f1_std = np.mean(rf_f1), np.std(rf_f1)
    gbdt_f1_mean, gbdt_f1_std = np.mean(gbdt_f1), np.std(gbdt_f1)
    
    # Accuracy plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Colors for each model
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Test Accuracy Plot
    acc_values = [rf_acc_mean, gbdt_acc_mean] + eval_acc
    acc_errors = [rf_acc_std, gbdt_acc_std] + [0, 0, 0]  # No error bars for Geneformer models
    
    ax[0].bar(models, acc_values, yerr=acc_errors, capsize=5, color=colors)
    ax[0].set_title('Test Accuracy vs Eval Accuracy', fontsize=14)
    ax[0].set_ylabel('Accuracy', fontsize=12)
    ax[0].set_ylim([0.5, 1.0])
    ax[0].tick_params(axis='x', labelsize=6)
    ax[0].tick_params(axis='y', labelsize=6)
    
    # Macro F1 Plot
    f1_values = [rf_f1_mean, gbdt_f1_mean] + eval_macro_f1
    f1_errors = [rf_f1_std, gbdt_f1_std] + [0, 0, 0]  # No error bars for Geneformer models
    
    ax[1].bar(models, f1_values, yerr=f1_errors, capsize=5, color=colors)
    ax[1].set_title('Test Macro F1 vs Eval Macro F1', fontsize=14)
    ax[1].set_ylabel('Macro F1', fontsize=12)
    ax[1].set_ylim([0.0, 1.0])
    ax[1].tick_params(axis='x', labelsize=6)
    ax[1].tick_params(axis='y', labelsize=6)
    
    # plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_output/baselines_geneformer_compare.png')

# Step 4: Main execution
if __name__ == "__main__":
    fine_tuning_log_path = 'model_output/fine_tuning_log.txt'
    combined_results_path = 'model_output/combined_results.csv'
    
    # Parse data
    eval_acc, eval_macro_f1 = parse_fine_tuning_log(fine_tuning_log_path)
    rf_acc, gbdt_acc, rf_f1, gbdt_f1 = parse_combined_results(combined_results_path)
    
    # Plot results
    plot_results(eval_acc, eval_macro_f1, rf_acc, gbdt_acc, rf_f1, gbdt_f1)
