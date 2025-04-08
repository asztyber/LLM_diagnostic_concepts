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
    """Create a plot showing metrics by example complexity with improved aesthetics"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Get all F1 columns
    f1_columns = [col for col in df.columns if 'F1' in col]
    f1_means, f1_stds = process_metrics_stats(df, f1_columns)
    
    # Create the scatter plot with error bars
    plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                fmt='o', capsize=3, capthick=1.5,
                ecolor='gray', markersize=8,
                color='#2E86C1',  # Nice blue color
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
    
    # Customize the plot
    plt.xlabel('Number of Relations', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with some padding
    plt.ylim(-0.1, 1.2)
    plt.xlim(f1_means.index.min() - 0.5, f1_means.index.max() + 0.5)
    
    # Customize ticks
    plt.xticks(f1_means.index, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_metrics(dfs, save_path):
    """Create an aggregated plot comparing different versions with improved aesthetics"""
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    colors = ['#2E86C1', '#28B463', '#E74C3C']  # Blue, Green, Red
    markers = ['o', 's', 'D']  # Circle, Square, Diamond
    
    for (name, df), color, marker in zip(dfs.items(), colors, markers):
        # Get all F1 columns
        f1_columns = [col for col in df.columns if 'F1' in col]
        f1_means, f1_stds = process_metrics_stats(df, f1_columns)
        
        # Scatter plot with error bars
        plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                    label=name, color=color, marker=marker,
                    fmt='o', capsize=3, capthick=1.5,
                    alpha=0.7, markersize=8,
                    elinewidth=1.5,
                    markeredgecolor='white',
                    markeredgewidth=1.5)
        
        # Trend line
        z = np.polyfit(f1_means.index, f1_means, 1)
        p = np.poly1d(z)
        x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
        plt.plot(x_new, p(x_new), 
                color=color, linestyle='--', alpha=0.5,
                linewidth=2)
    
    # Customize the plot
    plt.xlabel('Number of Relations', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title('Comparison of F1 Scores Across Versions', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits
    plt.ylim(-0.1, 1.2)
    
    # Customize ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add legend with better positioning
    plt.legend(fontsize=10, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def count_solutions(solution_str):
    """Count number of solutions in a comma-separated string"""
    if pd.isna(solution_str) or not solution_str:
        return 0
    return len([s for s in str(solution_str).split(',') if s.strip()])

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

def plot_metrics_by_solutions(df, title, save_path, true_col):
    """Create a plot showing metrics sorted by number of solutions"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Get all F1 columns
    f1_columns = [col for col in df.columns if 'F1' in col]
    f1_means, f1_stds = process_metrics_stats_by_solutions(df, f1_columns, true_col)
    
    # Create the scatter plot with error bars
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
    
    # Customize the plot
    plt.xlabel('Number of Solutions', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with some padding
    plt.ylim(-0.1, 1.2)
    plt.xlim(f1_means.index.min() - 0.5, f1_means.index.max() + 0.5)
    
    # Customize ticks
    plt.xticks(f1_means.index, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_metrics_by_solutions(dfs, save_path, true_col):
    """Create an aggregated plot comparing different versions, sorted by number of solutions"""
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    colors = ['#2E86C1', '#28B463', '#E74C3C']
    markers = ['o', 's', 'D']
    
    for (name, df), color, marker in zip(dfs.items(), colors, markers):
        f1_columns = [col for col in df.columns if 'F1' in col]
        f1_means, f1_stds = process_metrics_stats_by_solutions(df, f1_columns, true_col)
        
        plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                    label=name, color=color, marker=marker,
                    fmt='o', capsize=3, capthick=1.5,
                    alpha=0.7, markersize=8,
                    elinewidth=1.5,
                    markeredgecolor='white',
                    markeredgewidth=1.5)
        
        z = np.polyfit(f1_means.index, f1_means, 1)
        p = np.poly1d(z)
        x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
        plt.plot(x_new, p(x_new), 
                color=color, linestyle='--', alpha=0.5,
                linewidth=2)
    
    plt.xlabel('Number of Solutions', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title('Comparison of F1 Scores by Number of Solutions', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.1, 1.2)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.legend(fontsize=10, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Read and process all files
base_path = 'results/Karol/results_article'
versions = ['4o_mini', 'o1', 'o3_mini']

# Configure file groups with correct column names
file_groups = {
    'mso': {
        'files': [f'{base_path}/{ver}/MSO.xlsx' for ver in versions],
        'names': [f'{ver}_mso' for ver in versions],
        'true_col': 'MSO',
        'pred_col': 'Generated MSO'
    },
    'diagnoses': {
        'files': [f'{base_path}/{ver}/MINIMAL_DIAGNOSES.xlsx' for ver in versions],
        'names': [f'{ver}_diagnoses' for ver in versions],
        'true_col': 'Minimal Diagnoses',
        'pred_col': 'Generated  Minimal Diagnoses'  # Note the double space
    },
    'conflicts': {
        'files': [f'{base_path}/{ver}/MINIMAL_CONFLICTS.xlsx' for ver in versions],
        'names': [f'{ver}_conflicts' for ver in versions],
        'true_col': 'Minimal Conflicts',
        'pred_col': 'Generated  Minimal Conflicts'  # Assuming similar pattern
    }
}

# Process each group separately
group_dfs = {}
for group_name, config in file_groups.items():
    group_dfs[group_name] = {}
    for file, name in zip(config['files'], config['names']):
        df = pd.read_excel(file, skiprows=2, usecols='B:BO')
        # Fill NaN values in 'Example content' with previous non-empty value
        df['Example content'] = df['Example content'].ffill()
        df = df.drop(df.index[-1])  # Drop last row from each file
        print(df.columns)
        print(file)
        group_dfs[group_name][name] = process_dataframe(
            df, 
            config['true_col'],  # Pass the true column name
            config['pred_col']   # Pass the predicted column name
        )

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

# Create additional plots sorted by number of solutions
for group_name, dfs in group_dfs.items():
    true_col = solution_columns[group_name]
    
    # Individual plots for each version
    for name, df in dfs.items():
        plot_metrics_by_solutions(
            df, 
            f'{group_name.upper()} - Metrics by Number of Solutions - {name}',
            f'pic/metrics_by_solutions_{name}.pdf',
            true_col
        )
    
    # Aggregated plot for the group
    plot_aggregated_metrics_by_solutions(
        dfs, 
        f'pic/metrics_by_solutions_{group_name}_aggregated.pdf',
        true_col
    )

# %%
