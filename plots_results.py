# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import ast
import re

# %%

def string_to_sets(mso_string):
    if pd.isna(mso_string):
        return set()
    return {frozenset(s.split(', ')) for s in mso_string.split('\n')}

def calculate_metrics(true_str, pred_str):
    """Calculate metrics between true and predicted sets"""
    # Convert strings to sets
    true_set = set(true_str.split(',')) if true_str else set()
    pred_set = set(pred_str.split(',')) if pred_str else set()
    
    # Remove empty strings
    true_set = {x.strip() for x in true_set if x.strip()}
    pred_set = {x.strip() for x in pred_set if x.strip()}
    
    # Calculate metrics
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1
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

def process_dataframe(df, true_col, pred_col):
    """Process dataframe to calculate all metrics"""
    # Add relation count
    df['relation_count'] = df['Example content'].apply(count_relations)
    
    # Find all prediction columns (base name and its variants with .1, .2, etc.)
    pred_columns = [col for col in df.columns if col.startswith(pred_col) or col == pred_col]
    
    # Calculate metrics for each prediction column
    for i, pred_col_name in enumerate(pred_columns):
        metrics = df.apply(
            lambda row: calculate_metrics(
                str(row[true_col]) if pd.notna(row[true_col]) else "",
                str(row[pred_col_name]) if pd.notna(row[pred_col_name]) else ""
            ),
            axis=1
        )
        
        suffix = f"_{i}" if i > 0 else ""
        df[f'TP{suffix}'] = metrics.apply(lambda x: x['TP'])
        df[f'FP{suffix}'] = metrics.apply(lambda x: x['FP'])
        df[f'FN{suffix}'] = metrics.apply(lambda x: x['FN'])
        df[f'Precision{suffix}'] = metrics.apply(lambda x: x['Precision'])
        df[f'Recall{suffix}'] = metrics.apply(lambda x: x['Recall'])
        df[f'F1{suffix}'] = metrics.apply(lambda x: x['F1'])
    
    return df

def process_metrics_stats(df, f1_columns):
    """Helper function to process metrics based on number of versions"""
    grouped = df.groupby('relation_count')
    
    if len(f1_columns) == 1:
        # For single version (o1), just use the values directly
        f1_means = grouped[f1_columns[0]].mean()
        f1_stds = grouped[f1_columns[0]].std().fillna(0)  # Fill NaN with 0 for single values
    else:
        # For multiple versions, compute stats across all versions
        grouped_stats = grouped.agg({
            col: ['mean', 'std'] for col in f1_columns
        })
        f1_means = grouped_stats[f1_columns].xs('mean', axis=1, level=1).mean(axis=1)
        f1_stds = grouped_stats[f1_columns].xs('std', axis=1, level=1).mean(axis=1)
    
    return f1_means, f1_stds

def plot_metrics(df, title, save_path):
    """Plot showing metrics by example complexity"""
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    f1_columns = [col for col in df.columns if 'F1' in col]
    f1_means, f1_stds = process_metrics_stats(df, f1_columns)
    
    plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                fmt='o', capsize=3, capthick=1.5,
                ecolor='gray', markersize=8,
                color='#2E86C1',
                alpha=0.7,
                elinewidth=1.5,
                markeredgecolor='white',
                markeredgewidth=1.5)
    
    # Add trend line
    z = np.polyfit(f1_means.index, f1_means, 1)
    p = np.poly1d(z)
    x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
    y_new = p(x_new)
    plt.plot(x_new, y_new, color='#E74C3C', linestyle='--', 
             alpha=0.8, linewidth=2, label='Trend')
    
    plt.xlabel('Number of Relations', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.ylim(-0.1, 1.2)
    plt.xlim(f1_means.index.min() - 0.5, f1_means.index.max() + 0.5)
    
    plt.xticks(f1_means.index, fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_metrics(dfs, save_path):
    """Create an aggregated plot comparing different versions"""
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    plt.ylim(-0.05, 1.05)
    
    colors = ['#2E86C1', '#28B463', '#E74C3C']
    markers = ['o', 's', 'D']
    
    name_mapping = {
        'karol_4o_mini_mso': '4o-mini',
        'karol_o1_mso': 'o1',
        'karol_o3_mini_mso': 'o3-mini',
        'maxence_gpt_4o_mini_mso': '4o-mini-gpt',
        'maxence_o1_mso': 'o1-gpt',
        'maxence_o3_mini_mso': 'o3-mini-gpt',
        '4o_mini_conflicts': '4o-mini',
        '4o_mini_diagnoses': '4o-mini',
        'o1_conflicts': 'o1',
        'o1_diagnoses': 'o1',
        'o3_mini_conflicts': 'o3-mini',
        'o3_mini_diagnoses': 'o3-mini'
    }
    
    for (name, df), color, marker in zip(dfs.items(), colors, markers):
        # Group by relation count first
        grouped = df.groupby('relation_count')
        f1_columns = [col for col in df.columns if 'F1' in col]
        
        if len(f1_columns) == 1:
            # For single version, just use the values directly
            f1_means = grouped[f1_columns[0]].mean()
            f1_stds = grouped[f1_columns[0]].std().fillna(0)
        else:
            # For multiple versions, compute stats across all versions
            grouped_stats = grouped.agg({
                col: ['mean', 'std'] for col in f1_columns
            })
            f1_means = grouped_stats[f1_columns].xs('mean', axis=1, level=1).mean(axis=1)
            f1_stds = grouped_stats[f1_columns].xs('std', axis=1, level=1).mean(axis=1)
        
        simple_name = name_mapping.get(name, name)
        
        plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                    label=simple_name, color=color, marker=marker,
                    fmt='o', capsize=3, capthick=1.5,
                    alpha=0.7, markersize=8,
                    elinewidth=1.5,
                    markeredgecolor='white',
                    markeredgewidth=1.5)
        
        # Trend line
        z = np.polyfit(f1_means.index, f1_means, 1)
        p = np.poly1d(z)
        x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
        y_new = p(x_new)
        y_new = np.clip(y_new, 0, 1)
        plt.plot(x_new, y_new, 
                color=color, linestyle='--', alpha=0.5,
                linewidth=2)
    
    plt.xlabel('Number of Relations', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.legend(fontsize=10, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def count_solutions(solution_str):
    """Count number of solutions in a newline-separated string"""
    if pd.isna(solution_str) or not solution_str:
        return 0
    return len([s for s in str(solution_str).split('\n') if s.strip()])

def process_metrics_stats_by_solutions(df, f1_columns, true_col):
    """Helper function to process metrics based on solution count"""
    # Compute number of solutions
    df['solution_count'] = df[true_col].apply(count_solutions)
    
    # Group by solution count
    grouped = df.groupby('solution_count')
    
    if len(f1_columns) == 1:
        # For single version, just use the values directly
        f1_means = grouped[f1_columns[0]].mean()
        f1_stds = grouped[f1_columns[0]].std().fillna(0)
    else:
        # For multiple versions, compute stats across all versions
        grouped_stats = grouped.agg({
            col: ['mean', 'std'] for col in f1_columns
        })
        f1_means = grouped_stats[f1_columns].xs('mean', axis=1, level=1).mean(axis=1)
        f1_stds = grouped_stats[f1_columns].xs('std', axis=1, level=1).mean(axis=1)
    
    return f1_means, f1_stds

def plot_metrics_by_solutions(df, save_path, true_col, x_label):
    """Create a plot showing metrics sorted by number of solutions"""
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    f1_columns = [col for col in df.columns if 'F1' in col]
    f1_means, f1_stds = process_metrics_stats_by_solutions(df, f1_columns, true_col)
    
    plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                fmt='o', capsize=3, capthick=1.5,
                ecolor='gray', markersize=8,
                color='#2E86C1',
                alpha=0.7,
                elinewidth=1.5,
                markeredgecolor='white',
                markeredgewidth=1.5)
    
    # Trend line
    z = np.polyfit(f1_means.index, f1_means, 1)
    p = np.poly1d(z)
    x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
    y_new = p(x_new)
    plt.plot(x_new, y_new, color='#E74C3C', linestyle='--', 
             alpha=0.8, linewidth=2, label='Trend')
    
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.ylim(0, 1.0)
    plt.xlim(f1_means.index.min() - 0.5, f1_means.index.max() + 0.5)
    plt.xticks(f1_means.index, fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_metrics_by_solutions(dfs, save_path, true_col, x_label):
    """Create an aggregated plot comparing different versions, sorted by number of solutions"""
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    
    plt.ylim(-0.05, 1.05)
    
    colors = ['#2E86C1', '#28B463', '#E74C3C']
    markers = ['o', 's', 'D']
    
    # Simplified version names mapping
    name_mapping = {
        'karol_4o_mini_mso': '4o-mini',
        'karol_o1_mso': 'o1',
        'karol_o3_mini_mso': 'o3-mini',
        'maxence_gpt_4o_mini_mso': '4o-mini-gpt',
        'maxence_o1_mso': 'o1-gpt',
        'maxence_o3_mini_mso': 'o3-mini-gpt',
        '4o_mini_conflicts': '4o-mini',
        '4o_mini_diagnoses': '4o-mini',
        'o1_conflicts': 'o1',
        'o1_diagnoses': 'o1',
        'o3_mini_conflicts': 'o3-mini',
        'o3_mini_diagnoses': 'o3-mini'
    }
    
    for (name, df), color, marker in zip(dfs.items(), colors, markers):
        f1_columns = [col for col in df.columns if 'F1' in col]
        f1_means, f1_stds = process_metrics_stats_by_solutions(df, f1_columns, true_col)
        
        # Clip the error bars to stay within [0, 1] range
        f1_stds = np.minimum(f1_stds, np.minimum(f1_means - 0, 1 - f1_means))
        
        # Use simplified name for legend
        simple_name = name_mapping.get(name, name)
        
        plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                    label=simple_name, color=color, marker=marker,
                    fmt='o', capsize=3, capthick=1.5,
                    alpha=0.7, markersize=8,
                    elinewidth=1.5,
                    markeredgecolor='white',
                    markeredgewidth=1.5)
        
        z = np.polyfit(f1_means.index, f1_means, 1)
        p = np.poly1d(z)
        x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
        y_new = p(x_new)
        # Clip trend line values to [0, 1] range
        y_new = np.clip(y_new, 0, 1)
        plt.plot(x_new, y_new, 
                color=color, linestyle='--', alpha=0.5,
                linewidth=2)
    
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.legend(fontsize=10, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_bars_mso(dfs, save_path, title):
    """Create a bar plot showing aggregated F1, Precision, and Recall scores for MSO results"""
    plt.figure(figsize=(6, 3))
    sns.set_style("whitegrid")
    
    # Store results for each model and method
    results = {}
    
    # Model mapping
    model_mapping = {
        'karol_4o_mini_mso': ('4o-mini', 'Karol'),
        'karol_o1_mso': ('o1', 'Karol'),
        'karol_o3_mini_mso': ('o3-mini', 'Karol'),
        'maxence_gpt_4o_mini_mso': ('4o-mini', 'Maxence'),
        'maxence_o1_mso': ('o1', 'Maxence'),
        'maxence_o3_mini_mso': ('o3-mini', 'Maxence')
    }
    
    # Process each model's results
    for model_name, df in dfs.items():
        model, method = model_mapping.get(model_name, (model_name, 'Unknown'))
        if model not in results:
            results[model] = {'Karol': {}, 'Maxence': {}}
        
        # Get metric columns
        f1_cols = [col for col in df.columns if 'F1' in col]
        precision_cols = [col for col in df.columns if 'Precision' in col]
        recall_cols = [col for col in df.columns if 'Recall' in col]
        
        # Calculate mean and std for each metric
        results[model][method]['F1'] = df[f1_cols].values.mean()
        results[model][method]['Precision'] = df[precision_cols].values.mean()
        results[model][method]['Recall'] = df[recall_cols].values.mean()
        results[model][method]['F1_std'] = df[f1_cols].values.std()
        results[model][method]['Precision_std'] = df[precision_cols].values.std()
        results[model][method]['Recall_std'] = df[recall_cols].values.std()
    
    # Setup the plot
    metrics = ['F1', 'Precision', 'Recall']
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.12
    
    colors = {
        '4o-mini': '#2E86C1',
        'o1': '#28B463',
        'o3-mini': '#E74C3C'
    }
    
    n_models = len(models)
    
    # Plot bars for each model and method
    for i, model in enumerate(models):
        base_offset = (i - n_models/2) * (width * 2.2)
        
        # Get values and clip error bars
        karol_values = [results[model]['Karol'][m] for m in metrics]
        karol_errors = [results[model]['Karol'][f'{m}_std'] for m in metrics]
        maxence_values = [results[model]['Maxence'][m] for m in metrics]
        maxence_errors = [results[model]['Maxence'][f'{m}_std'] for m in metrics]
        
        # Clip error bars
        karol_errors = np.minimum(karol_errors, 
                                np.minimum(karol_values, np.subtract(1, karol_values)))
        maxence_errors = np.minimum(maxence_errors, 
                                  np.minimum(maxence_values, np.subtract(1, maxence_values)))
        
        # Plot Karol's results
        plt.bar(x + base_offset, karol_values,
                width, label=f'{model} (dedicated)',
                color=colors[model], alpha=0.7)
        
        # Plot Maxence's results
        plt.bar(x + base_offset + width, maxence_values,
                width, label=f'{model} (broad)',
                color=colors[model], alpha=0.7,
                hatch='///')
        
        # Add error bars
        plt.errorbar(x + base_offset, karol_values,
                    yerr=karol_errors,
                    fmt='none', color='black', capsize=3)
        plt.errorbar(x + base_offset + width, maxence_values,
                    yerr=maxence_errors,
                    fmt='none', color='black', capsize=3)
    

    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    
    plt.xticks(x, metrics)
    
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
              ncol=3, fontsize=8)  # Smaller font
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_bars_other(dfs, save_path, title):
    """Create a bar plot showing aggregated F1, Precision, and Recall scores for conflicts/diagnoses"""
    plt.figure(figsize=(6, 3))
    sns.set_style("whitegrid")
    
    # Store results for each model
    results = {}
    
    # Model mapping (simplified for conflicts/diagnoses)
    name_mapping = {
        '4o_mini_conflicts': '4o-mini',
        '4o_mini_diagnoses': '4o-mini',
        'o1_conflicts': 'o1',
        'o1_diagnoses': 'o1',
        'o3_mini_conflicts': 'o3-mini',
        'o3_mini_diagnoses': 'o3-mini'
    }
    
    # Process each model's results
    for model_name, df in dfs.items():
        model = name_mapping.get(model_name, model_name)
        
        # Get metric columns
        f1_cols = [col for col in df.columns if 'F1' in col]
        precision_cols = [col for col in df.columns if 'Precision' in col]
        recall_cols = [col for col in df.columns if 'Recall' in col]
        
        # Calculate mean and std for each metric
        if model not in results:
            results[model] = {}
        
        results[model]['F1'] = df[f1_cols].values.mean()
        results[model]['Precision'] = df[precision_cols].values.mean()
        results[model]['Recall'] = df[recall_cols].values.mean()
        results[model]['F1_std'] = df[f1_cols].values.std()
        results[model]['Precision_std'] = df[precision_cols].values.std()
        results[model]['Recall_std'] = df[recall_cols].values.std()
        
        print(f"\nAggregated Metrics for {title}:")
        for model, metrics_data in results.items():
            print(f"  Model: {model}")
            for metric, value in metrics_data.items():
                if '_std' not in metric:
                    std_value = metrics_data.get(f'{metric}_std', 0)
                    print(f"    {metric}: Mean = {value:.3f}, Std = {std_value:.3f}")
    
    # Setup the plot
    metrics = ['F1', 'Precision', 'Recall']
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25 
    
    colors = {
        '4o-mini': '#2E86C1',
        'o1': '#28B463',
        'o3-mini': '#E74C3C'
    }
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_offset = (i - 1) * width
        
        # Get values and clip error bars
        values = [results[model][m] for m in metrics]
        errors = [results[model][f'{m}_std'] for m in metrics]
        
        # Clip error bars
        errors = np.minimum(errors, 
                          np.minimum(values, np.subtract(1, values)))
        
        # Plot bars
        plt.bar(x + model_offset, values,
                width, label=model,
                color=colors[model], alpha=0.7)
        
        # Add error bars
        plt.errorbar(x + model_offset, values,
                    yerr=errors,
                    fmt='none', color='black', capsize=3)
    
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)  # Changed back to 1.1
    
    plt.xticks(x, metrics)
    
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
              ncol=3, fontsize=8)  # Smaller font
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_example_info(content):
    """Extract information from example content string"""
    if pd.isna(content):
        return None
        
    # Split the content into lines
    lines = content.split('\n')
    
    # Extract title
    title = lines[0].split("'")[1]
    
    # Extract variables using ast.literal_eval
    x_vars = ast.literal_eval(lines[1].split('=')[1].strip())
    f_vars = ast.literal_eval(lines[2].split('=')[1].strip())
    z_vars = ast.literal_eval(lines[3].split('=')[1].strip())
    
    # Count relations by finding number of items in r list
    relations = count_relations(content)
    
    return {
        'Example': title,
        'X_vars': len(x_vars),
        'F_vars': len(f_vars),
        'Z_vars': len(z_vars),
        'Relations': relations
    }

def process_examples():
    # Get dataframes from group_dfs
    mso_df = group_dfs['mso']['karol_4o_mini_mso']
    conflicts_df = group_dfs['conflicts']['4o_mini_conflicts']
    diagnoses_df = group_dfs['diagnoses']['4o_mini_diagnoses']
    
    # Process each unique example
    examples_info = []
    
    # Get unique examples and remove NaN values
    unique_examples = mso_df['Example content'].dropna().unique()
    
    for example_content in unique_examples:
        # Get basic example info
        info = extract_example_info(example_content)
        if info is None:
            continue
            
        example_name = info['Example']
        
        # Count MSOs
        mso_count = mso_df[mso_df['Example content'] == example_content]['Total Number of MSO'].values[0]
        info['MSO_count'] = mso_count
        
        # Get conflicts info
        conflicts_counts = conflicts_df[conflicts_df['Example content'] == example_content]['Total Number of Conflicts']
        info['Min_conflicts'] = conflicts_counts.min()
        info['Max_conflicts'] = conflicts_counts.max()
        
        # Get diagnoses info
        diagnoses_counts = diagnoses_df[diagnoses_df['Example content'] == example_content]['Total Number of Diagnoses']
        info['Min_diagnoses'] = diagnoses_counts.min()
        info['Max_diagnoses'] = diagnoses_counts.max()
        
        examples_info.append(info)
    
    # Create dataframe
    result_df = pd.DataFrame(examples_info)
    
    # Reorder columns
    column_order = ['Example', 'X_vars', 'Z_vars', 'F_vars', 'Relations', 
                   'MSO_count', 'Min_conflicts', 'Max_conflicts', 
                   'Min_diagnoses', 'Max_diagnoses']
    result_df = result_df[column_order]
    
    return result_df

base_path = 'results'
versions = {
    'Karol': {
        'path': f'{base_path}/Karol/results_article',
        'versions': ['4o_mini', 'o1', 'o3_mini']
    },
    'Maxence': {
        'path': f'{base_path}/results_DX2025_Maxence',
        'versions': ['gpt_4o_mini', 'o1', 'o3_mini']
    }
}

file_groups = {
    'mso': {
        'files': [],  # Will be populated dynamically
        'names': [],  # Will be populated dynamically
        'true_col': 'MSO',
        'pred_col': 'Generated MSO'
    },
    'diagnoses': {
        'files': [f'{versions["Karol"]["path"]}/{ver}/MINIMAL_DIAGNOSES.xlsx' for ver in versions["Karol"]["versions"]],
        'names': [f'{ver}_diagnoses' for ver in versions["Karol"]["versions"]],
        'true_col': 'Minimal Diagnoses',
        'pred_col': 'Generated  Minimal Diagnoses'  # Note the double space
    },
    'conflicts': {
        'files': [f'{versions["Karol"]["path"]}/{ver}/MINIMAL_CONFLICTS.xlsx' for ver in versions["Karol"]["versions"]],
        'names': [f'{ver}_conflicts' for ver in versions["Karol"]["versions"]],
        'true_col': 'Minimal Conflicts',
        'pred_col': 'Generated  Minimal Conflicts'  # Assuming similar pattern
    }
}

# Add MSO files from both sources
for source, config in versions.items():
    for ver in config['versions']:
        file_path = f'{config["path"]}/{ver}/MSO.{"csv" if source == "Maxence" else "xlsx"}'
        file_groups['mso']['files'].append(file_path)
        file_groups['mso']['names'].append(f'{source.lower()}_{ver}_mso')

def read_file(file_path, skiprows=2):
    """Read either Excel or CSV file"""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, skiprows=skiprows, usecols='B:BO')
    else:
        return pd.read_csv(file_path, sep=';', encoding='utf-8')

group_dfs = defaultdict(dict)
for group_name, config in file_groups.items():
    for file, name in zip(config['files'], config['names']):
        df = read_file(file)
        
        # Fill NaN values in 'Example content' with previous non-empty value
        df['Example content'] = df['Example content'].ffill()
        
        if file.endswith('.xlsx'):
            df = df.drop(df.index[-1])  # Drop last row only for Excel files
        
        # Verify columns exist
        if config['true_col'] not in df.columns:
            raise KeyError(f"True column '{config['true_col']}' not found in {file}")
        if config['pred_col'] not in df.columns:
            raise KeyError(f"Predicted column '{config['pred_col']}' not found in {file}")
            
        group_dfs[group_name][name] = process_dataframe(
            df, 
            config['true_col'],
            config['pred_col']
        )

# %%
# Create plots for each group
for group_name, dfs in group_dfs.items():
    # Individual plots for each version in the group
    for name, df in dfs.items():
        plot_metrics(df, f'{group_name.upper()} - Metrics by Example Complexity - {name}',
                    save_path=f'pic/metrics_{name}.pdf')
    
    # Aggregated plot for the group
    plot_aggregated_metrics(dfs, save_path=f'pic/metrics_{group_name}_aggregated.pdf')
    
    # Print summary statistics for the group
    print(f"\nSummary Statistics for {group_name.upper()}:")
    print("-" * 50)
    for name, df in dfs.items():
        mean_f1 = df['F1'].mean()
        std_f1 = df['F1'].std()
        
        print(f"\nVersion: {name}")
        print(f"Average F1 score: {mean_f1:.3f} ± {std_f1:.3f}")
        print(f"Average number of relations: {df['relation_count'].mean():.1f}")
        print(f"Correlation between F1 and relations: {df['F1'].corr(df['relation_count']):.3f}")

# Update the solution columns dictionary to use true column names
solution_columns = {
    'mso': 'MSO',
    'conflicts': 'Minimal Conflicts',
    'diagnoses': 'Minimal Diagnoses'
}

# Update the x-axis labels
x_labels = {
    'mso': 'Number of MSOs',
    'conflicts': 'Number of Conflicts',
    'diagnoses': 'Number of Diagnoses'
}

# Create additional plots sorted by number of solutions
for group_name, dfs in group_dfs.items():
    true_col = solution_columns[group_name]
    x_label = x_labels[group_name]
    
    # Individual plots for each version
    for name, df in dfs.items():
        plot_metrics_by_solutions(
            df, 
            f'pic/metrics_by_solutions_{name}.pdf',
            true_col,
            x_label
        )
    
    # Aggregated plot for the group
    plot_aggregated_metrics_by_solutions(
        dfs, 
        f'pic/metrics_by_solutions_{group_name}_aggregated.pdf',
        true_col,
        x_label
    )

# Update the plotting calls
for group_name, dfs in group_dfs.items():
    if group_name == 'mso':
        plot_aggregated_bars_mso(
            dfs,
            f'pic/metrics_bars_{group_name}.pdf',
            f'{group_name.upper()} Generation - Comparison of Metrics Across Models'
        )
    else:
        plot_aggregated_bars_other(
            dfs,
            f'pic/metrics_bars_{group_name}.pdf',
            f'{group_name.upper()} Generation - Comparison of Metrics Across Models'
        )

# %%
examples_df = process_examples()
print(examples_df)
