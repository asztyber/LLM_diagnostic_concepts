# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%

def string_to_sets(mso_string):
    """Convert MSO string to set of frozensets, handling NaN values"""
    if pd.isna(mso_string):
        return set()
    return {frozenset(s.split(', ')) for s in mso_string.split('\n')}

def calculate_metrics(true_mso, pred_mso):
    """Calculate TP, FP, FN, precision, recall, and F1 for a pair of MSOs"""
    true_sets = string_to_sets(true_mso)
    pred_sets = string_to_sets(pred_mso)
    
    tp = len(true_sets & pred_sets)
    fp = len(pred_sets - true_sets)
    fn = len(true_sets - pred_sets)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

def count_relations(example_content):
    """Count the number of relations in the example content"""
    if pd.isna(example_content):
        return 0
    try:
        r_start = example_content.find('r = [')
        if r_start == -1:
            return 0
        r_content = example_content[r_start:]
        r_list_start = r_content.find('[') + 1
        r_list_end = r_content.find(']')
        relations_str = r_content[r_list_start:r_list_end]
        relations = [rel.strip().strip("'").strip('"') for rel in relations_str.split(',\n')]
        return sum(1 for rel in relations)
    except:
        return 0

def process_dataframe(df):
    """Process dataframe to calculate all metrics"""
    # Add relation count
    df['relation_count'] = df['Example content'].apply(count_relations)
    
    # Process all Generated MSO columns
    mso_columns = ['Generated MSO'] + [f'Generated MSO.{i}' for i in range(1, 10)]
    
    for col in mso_columns:
        if col in df.columns:
            metrics = df.apply(
                lambda row: calculate_metrics(row['MSO'], row[col]),
                axis=1
            )
            
            # Add new columns with metrics
            prefix = col.replace('Generated MSO', 'MSO')
            df[f'{prefix}_TP'] = metrics.apply(lambda x: x['TP'])
            df[f'{prefix}_FP'] = metrics.apply(lambda x: x['FP'])
            df[f'{prefix}_FN'] = metrics.apply(lambda x: x['FN'])
            df[f'{prefix}_Precision'] = metrics.apply(lambda x: x['Precision'])
            df[f'{prefix}_Recall'] = metrics.apply(lambda x: x['Recall'])
            df[f'{prefix}_F1'] = metrics.apply(lambda x: x['F1'])
    
    return df

def plot_metrics(df, title, save_path=None):
    """Create plot for a single dataset and optionally save it"""
    # Get F1 columns
    f1_columns = [col for col in df.columns if col.endswith('_F1')]
    
    # For datasets with only one MSO version, std will be 0
    f1_means = df[f1_columns].mean(axis=1) if len(f1_columns) > 1 else df[f1_columns[0]]
    f1_stds = df[f1_columns].std(axis=1) if len(f1_columns) > 1 else pd.Series(0, index=df.index)
    
    # Sort by relation count
    sort_idx = df['relation_count'].argsort()
    sorted_means = f1_means.iloc[sort_idx]
    sorted_stds = f1_stds.iloc[sort_idx]
    sorted_relations = df['relation_count'].iloc[sort_idx]
    sorted_names = df['Example name'].iloc[sort_idx]
    
    # Create plot
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    
    # Plot F1 scores
    line1 = ax1.plot(range(len(sorted_means)), sorted_means, 'bo-',
                     linewidth=2, markersize=6, label='F1 Score')
    
    # Add standard deviation only if there are multiple versions
    if len(f1_columns) > 1:
        fill = ax1.fill_between(range(len(sorted_means)),
                               sorted_means - sorted_stds,
                               sorted_means + sorted_stds,
                               alpha=0.2, color='blue', label='±1 std dev')
    
    # Add relation counts
    ax2 = ax1.twinx()
    line2 = ax2.plot(range(len(sorted_relations)), sorted_relations, 'r--',
                     alpha=0.5, label='Number of Relations')
    
    # Customize plot
    ax1.set_xlabel('Examples (sorted by number of relations)')
    ax1.set_ylabel('F1 Score')
    ax2.set_ylabel('Number of Relations', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_ylim(max(0, sorted_means.min() - 0.1), 1.1)
    plt.title(title)
    
    # Add x-tick labels
    ax1.set_xticks(range(len(sorted_means)))
    ax1.set_xticklabels(sorted_names, rotation=90, ha='center', fontsize=8)
    
    # Add legend
    if len(f1_columns) > 1:
        lines = line1 + [fill] + line2
        labels = ['F1 Score', '±1 std dev', 'Number of Relations']
    else:
        lines = line1 + line2
        labels = ['F1 Score', 'Number of Relations']
    ax1.legend(lines, labels, loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_aggregated_metrics(dfs, save_path=None):
    """Create aggregated plot showing F1, Precision, and Recall for all datasets"""
    plt.figure(figsize=(8, 5))
    
    # Position of bars
    datasets = list(dfs.keys())
    x = np.arange(len(datasets))
    width = 0.25  # Width of bars
    
    # Collect metrics for each dataset
    f1_means = []
    f1_stds = []
    precision_means = []
    precision_stds = []
    recall_means = []
    recall_stds = []
    
    for df in dfs.values():
        # Get metric columns
        f1_cols = [col for col in df.columns if col.endswith('_F1')]
        precision_cols = [col for col in df.columns if col.endswith('_Precision')]
        recall_cols = [col for col in df.columns if col.endswith('_Recall')]
        
        # Calculate means and stds
        f1_means.append(df[f1_cols].mean().mean())
        f1_stds.append(df[f1_cols].mean().std())
        precision_means.append(df[precision_cols].mean().mean())
        precision_stds.append(df[precision_cols].mean().std())
        recall_means.append(df[recall_cols].mean().mean())
        recall_stds.append(df[recall_cols].mean().std())
    
    # Create bars
    plt.bar(x - width, f1_means, width, label='F1', color='blue', 
            yerr=f1_stds, capsize=5, alpha=0.7)
    plt.bar(x, precision_means, width, label='Precision', color='green',
            yerr=precision_stds, capsize=5, alpha=0.7)
    plt.bar(x + width, recall_means, width, label='Recall', color='red',
            yerr=recall_stds, capsize=5, alpha=0.7)
    
    # Customize plot
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('MSO Generation - Comparison of Metrics Across Models')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels on top of bars
    def add_value_labels(x, values, stds):
        for i, (v, s) in enumerate(zip(values, stds)):
            plt.text(x[i], v + s + 0.02, f'{v:.2f}±{s:.2f}', 
                    ha='center', va='bottom', rotation=0, fontsize=8)
    
    add_value_labels(x - width, f1_means, f1_stds)
    add_value_labels(x, precision_means, precision_stds)
    add_value_labels(x + width, recall_means, recall_stds)
    
    plt.ylim(0, 1.2)  # Set y-axis limit to accommodate labels
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

# Read and process all files
files = ['results/Karol/results_article/4o_mini/MSO.xlsx',
         'results/Karol/results_article/o1/MSO.xlsx',
         'results/Karol/results_article/o3_mini/MSO.xlsx']
names = ['4o_mini', 'o1', 'o3']

dfs = {}
for file, name in zip(files, names):
    df = pd.read_excel(file, skiprows=2, usecols='B:BO')
    df = df.drop(23)
    dfs[name] = process_dataframe(df)

# Create plots for each dataset
for name, df in dfs.items():
    plot_metrics(df, f'MSO Generation - F1 Scores by Example Complexity - {name}',
                save_path=f'pic/mso_metrics_{name}.pdf')

# Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)
for name, df in dfs.items():
    f1_cols = [col for col in df.columns if col.endswith('_F1')]
    mean_f1 = df[f1_cols].mean().mean()
    std_f1 = df[f1_cols].mean().std()
    
    print(f"\nDataset: {name}")
    print(f"Average F1 score: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Average number of relations: {df['relation_count'].mean():.1f}")
    print(f"Correlation between F1 and relations: {df[f1_cols[0]].corr(df['relation_count']):.3f}")

# %%

# Create and save aggregated plot
print("\nCreating aggregated metrics plot...")
plot_aggregated_metrics(dfs, save_path='pic/mso_metrics_aggregated.pdf')

# %%
