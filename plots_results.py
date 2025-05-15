# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import ast

# --- Utility Functions ---
def string_to_sets(mso_string):
    if pd.isna(mso_string):
        return set()
    return {frozenset(s.split(', ')) for s in mso_string.split('\n')}

def calculate_metrics(true_str, pred_str):
    true_set = {x.strip() for x in str(true_str).split(',') if x.strip()} if true_str else set()
    pred_set = {x.strip() for x in str(pred_str).split(',') if x.strip()} if pred_str else set()
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}

def count_relations(example_content):
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

def count_solutions(solution_str):
    if pd.isna(solution_str) or not solution_str:
        return 0
    return len([s for s in str(solution_str).split('\n') if s.strip()])

# --- Data Processing Functions ---
def process_dataframe(df, true_col, pred_col):
    df['relation_count'] = df['Example content'].apply(count_relations)
    pred_columns = [col for col in df.columns if col.startswith(pred_col)]
    for i, pred_col_name in enumerate(pred_columns):
        metrics = df.apply(
            lambda row: calculate_metrics(
                str(row[true_col]) if pd.notna(row[true_col]) else "",
                str(row[pred_col_name]) if pd.notna(row[pred_col_name]) else ""
            ), axis=1
        )
        suffix = f"_{i}" if i > 0 else ""
        for metric in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']:
            df[f'{metric}{suffix}'] = metrics.apply(lambda x: x[metric])
    return df

def process_metrics_stats(df, f1_columns):
    grouped = df.groupby('relation_count')
    if len(f1_columns) == 1:
        f1_means = grouped[f1_columns[0]].mean()
        f1_stds = grouped[f1_columns[0]].std().fillna(0)
    else:
        grouped_stats = grouped.agg({col: ['mean', 'std'] for col in f1_columns})
        f1_means = grouped_stats[f1_columns].xs('mean', axis=1, level=1).mean(axis=1)
        f1_stds = grouped_stats[f1_columns].xs('std', axis=1, level=1).mean(axis=1)
    return f1_means, f1_stds

def process_metrics_stats_by_solutions(df, f1_columns, true_col):
    df['solution_count'] = df[true_col].apply(count_solutions)
    grouped = df.groupby('solution_count')
    if len(f1_columns) == 1:
        f1_means = grouped[f1_columns[0]].mean()
        f1_stds = grouped[f1_columns[0]].std().fillna(0)
    else:
        grouped_stats = grouped.agg({col: ['mean', 'std'] for col in f1_columns})
        f1_means = grouped_stats[f1_columns].xs('mean', axis=1, level=1).mean(axis=1)
        f1_stds = grouped_stats[f1_columns].xs('std', axis=1, level=1).mean(axis=1)
    return f1_means, f1_stds

# --- Plotting Functions ---
def plot_metrics(df, title, save_path):
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    f1_columns = [col for col in df.columns if 'F1' in col]
    f1_means, f1_stds = process_metrics_stats(df, f1_columns)
    plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                fmt='o', capsize=3, capthick=1.5,
                ecolor='gray', markersize=8,
                color='#2E86C1', alpha=0.7, elinewidth=1.5,
                markeredgecolor='white', markeredgewidth=1.5)
    z = np.polyfit(f1_means.index, f1_means, 1)
    p = np.poly1d(z)
    x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
    y_new = p(x_new)
    plt.plot(x_new, y_new, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2, label='Trend')
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

def plot_metrics_by_solutions(df, save_path, true_col, x_label):
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    f1_columns = [col for col in df.columns if 'F1' in col]
    f1_means, f1_stds = process_metrics_stats_by_solutions(df, f1_columns, true_col)
    plt.errorbar(f1_means.index, f1_means, yerr=f1_stds,
                fmt='o', capsize=3, capthick=1.5,
                ecolor='gray', markersize=8,
                color='#2E86C1', alpha=0.7, elinewidth=1.5,
                markeredgecolor='white', markeredgewidth=1.5)
    z = np.polyfit(f1_means.index, f1_means, 1)
    p = np.poly1d(z)
    x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
    y_new = p(x_new)
    plt.plot(x_new, y_new, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2, label='Trend')
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

def plot_aggregated_metrics(dfs, save_path):
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
        grouped = df.groupby('relation_count')
        f1_columns = [col for col in df.columns if 'F1' in col]
        if len(f1_columns) == 1:
            f1_means = grouped[f1_columns[0]].mean()
            f1_stds = grouped[f1_columns[0]].std().fillna(0)
        else:
            grouped_stats = grouped.agg({col: ['mean', 'std'] for col in f1_columns})
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
        z = np.polyfit(f1_means.index, f1_means, 1)
        p = np.poly1d(z)
        x_new = np.linspace(f1_means.index.min(), f1_means.index.max(), 100)
        y_new = np.clip(p(x_new), 0, 1)
        plt.plot(x_new, y_new, color=color, linestyle='--', alpha=0.5, linewidth=2)
    plt.xlabel('Number of Relations', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_metrics_by_solutions(dfs, save_path, true_col, x_label):
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
        f1_columns = [col for col in df.columns if 'F1' in col]
        f1_means, f1_stds = process_metrics_stats_by_solutions(df, f1_columns, true_col)
        f1_stds = np.minimum(f1_stds, np.minimum(f1_means - 0, 1 - f1_means))
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
        y_new = np.clip(p(x_new), 0, 1)
        plt.plot(x_new, y_new, color=color, linestyle='--', alpha=0.5, linewidth=2)
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_bars_mso(dfs, save_path, title):
    plt.figure(figsize=(6, 3))
    sns.set_style("whitegrid")
    results = {}
    model_mapping = {
        'karol_4o_mini_mso': ('4o-mini', 'Karol'),
        'karol_o1_mso': ('o1', 'Karol'),
        'karol_o3_mini_mso': ('o3-mini', 'Karol'),
        'maxence_gpt_4o_mini_mso': ('4o-mini', 'Maxence'),
        'maxence_o1_mso': ('o1', 'Maxence'),
        'maxence_o3_mini_mso': ('o3-mini', 'Maxence')
    }
    for model_name, df in dfs.items():
        model, method = model_mapping.get(model_name, (model_name, 'Unknown'))
        if model not in results:
            results[model] = {'Karol': {}, 'Maxence': {}}
        f1_cols = [col for col in df.columns if 'F1' in col]
        precision_cols = [col for col in df.columns if 'Precision' in col]
        recall_cols = [col for col in df.columns if 'Recall' in col]
        # Use all F1/Precision/Recall columns for direct mean/std (flattened)
        results[model][method]['F1'] = df[f1_cols].values.flatten().mean()
        results[model][method]['Precision'] = df[precision_cols].values.flatten().mean()
        results[model][method]['Recall'] = df[recall_cols].values.flatten().mean()
        results[model][method]['F1_std'] = df[f1_cols].values.flatten().std()
        results[model][method]['Precision_std'] = df[precision_cols].values.flatten().std()
        results[model][method]['Recall_std'] = df[recall_cols].values.flatten().std()
    metrics = ['F1', 'Precision', 'Recall']
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.12
    colors = {'4o-mini': '#2E86C1', 'o1': '#28B463', 'o3-mini': '#E74C3C'}
    n_models = len(models)
    for i, model in enumerate(models):
        base_offset = (i - n_models/2) * (width * 2.2)
        for method, hatch in zip(['Karol', 'Maxence'], [None, '///']):
            values = [results[model][method][m] for m in metrics]
            errors = [results[model][method][f'{m}_std'] for m in metrics]
            errors = np.minimum(errors, np.minimum(values, np.subtract(1, values)))
            plt.bar(x + base_offset + (width if method == 'Maxence' else 0), values,
                    width, label=f'{model} ({"dedicated" if method=="Karol" else "broad"})',
                    color=colors[model], alpha=0.7, hatch=hatch)
            plt.errorbar(x + base_offset + (width if method == 'Maxence' else 0), values,
                        yerr=errors, fmt='none', color='black', capsize=3)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(x, metrics)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_bars_other(dfs, save_path, title):
    plt.figure(figsize=(6, 3))
    sns.set_style("whitegrid")
    results = {}
    name_mapping = {
        '4o_mini_conflicts': '4o-mini',
        '4o_mini_diagnoses': '4o-mini',
        'o1_conflicts': 'o1',
        'o1_diagnoses': 'o1',
        'o3_mini_conflicts': 'o3-mini',
        'o3_mini_diagnoses': 'o3-mini'
    }
    for model_name, df in dfs.items():
        model = name_mapping.get(model_name, model_name)
        if model not in results:
            results[model] = {}
        f1_cols = [col for col in df.columns if 'F1' in col]
        precision_cols = [col for col in df.columns if 'Precision' in col]
        recall_cols = [col for col in df.columns if 'Recall' in col]
        # Use all F1/Precision/Recall columns for direct mean/std (flattened)
        results[model]['F1'] = df[f1_cols].values.flatten().mean()
        results[model]['Precision'] = df[precision_cols].values.flatten().mean()
        results[model]['Recall'] = df[recall_cols].values.flatten().mean()
        results[model]['F1_std'] = df[f1_cols].values.flatten().std()
        results[model]['Precision_std'] = df[precision_cols].values.flatten().std()
        results[model]['Recall_std'] = df[recall_cols].values.flatten().std()
    metrics = ['F1', 'Precision', 'Recall']
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.25
    colors = {'4o-mini': '#2E86C1', 'o1': '#28B463', 'o3-mini': '#E74C3C'}
    for i, model in enumerate(models):
        model_offset = (i - 1) * width
        values = [results[model][m] for m in metrics]
        errors = [results[model][f'{m}_std'] for m in metrics]
        errors = np.minimum(errors, np.minimum(values, np.subtract(1, values)))
        plt.bar(x + model_offset, values, width, label=model, color=colors[model], alpha=0.7)
        plt.errorbar(x + model_offset, values, yerr=errors, fmt='none', color='black', capsize=3)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(x, metrics)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- Example Info Extraction ---
def extract_example_info(content):
    if pd.isna(content):
        return None
    lines = content.split('\n')
    title = lines[0].split("'")[1]
    x_vars = ast.literal_eval(lines[1].split('=')[1].strip())
    f_vars = ast.literal_eval(lines[2].split('=')[1].strip())
    z_vars = ast.literal_eval(lines[3].split('=')[1].strip())
    relations = count_relations(content)
    return {'Example': title, 'X_vars': len(x_vars), 'F_vars': len(f_vars), 'Z_vars': len(z_vars), 'Relations': relations}

def process_examples():
    mso_df = group_dfs['mso']['karol_4o_mini_mso']
    conflicts_df = group_dfs['conflicts']['4o_mini_conflicts']
    diagnoses_df = group_dfs['diagnoses']['4o_mini_diagnoses']
    examples_info = []
    unique_examples = mso_df['Example content'].dropna().unique()
    for example_content in unique_examples:
        info = extract_example_info(example_content)
        if info is None:
            continue
        info['MSO_count'] = mso_df[mso_df['Example content'] == example_content]['Total Number of MSO'].values[0]
        conflicts_counts = conflicts_df[conflicts_df['Example content'] == example_content]['Total Number of Conflicts']
        info['Min_conflicts'] = conflicts_counts.min()
        info['Max_conflicts'] = conflicts_counts.max()
        diagnoses_counts = diagnoses_df[diagnoses_df['Example content'] == example_content]['Total Number of Diagnoses']
        info['Min_diagnoses'] = diagnoses_counts.min()
        info['Max_diagnoses'] = diagnoses_counts.max()
        examples_info.append(info)
    column_order = ['Example', 'X_vars', 'Z_vars', 'F_vars', 'Relations', 'MSO_count', 'Min_conflicts', 'Max_conflicts', 'Min_diagnoses', 'Max_diagnoses']
    return pd.DataFrame(examples_info)[column_order]

# --- File/Group Setup ---
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
    'mso': {'files': [], 'names': [], 'true_col': 'MSO', 'pred_col': 'Generated MSO'},
    'diagnoses': {
        'files': [f'{versions["Karol"]["path"]}/{ver}/MINIMAL_DIAGNOSES.xlsx' for ver in versions["Karol"]["versions"]],
        'names': [f'{ver}_diagnoses' for ver in versions["Karol"]["versions"]],
        'true_col': 'Minimal Diagnoses',
        'pred_col': 'Generated  Minimal Diagnoses'
    },
    'conflicts': {
        'files': [f'{versions["Karol"]["path"]}/{ver}/MINIMAL_CONFLICTS.xlsx' for ver in versions["Karol"]["versions"]],
        'names': [f'{ver}_conflicts' for ver in versions["Karol"]["versions"]],
        'true_col': 'Minimal Conflicts',
        'pred_col': 'Generated  Minimal Conflicts'
    }
}
for source, config in versions.items():
    for ver in config['versions']:
        file_path = f'{config["path"]}/{ver}/MSO.{"csv" if source == "Maxence" else "xlsx"}'
        file_groups['mso']['files'].append(file_path)
        file_groups['mso']['names'].append(f'{source.lower()}_{ver}_mso')

def read_file(file_path, skiprows=2):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, skiprows=skiprows, usecols='B:BO')
    else:
        return pd.read_csv(file_path, sep=';', encoding='utf-8')

group_dfs = defaultdict(dict)
for group_name, config in file_groups.items():
    for file, name in zip(config['files'], config['names']):
        df = read_file(file)
        df['Example content'] = df['Example content'].ffill()
        if file.endswith('.xlsx'):
            df = df.drop(df.index[-1])  # Drop last row only for Excel files
        
        # Verify columns exist
        if config['true_col'] not in df.columns:
            raise KeyError(f"True column '{config['true_col']}' not found in {file}")
        if config['pred_col'] not in df.columns:
            raise KeyError(f"Predicted column '{config['pred_col']}' not found in {file}")
        group_dfs[group_name][name] = process_dataframe(df, config['true_col'], config['pred_col'])

# --- Plotting and Reporting ---
solution_columns = {'mso': 'MSO', 'conflicts': 'Minimal Conflicts', 'diagnoses': 'Minimal Diagnoses'}
x_labels = {'mso': 'Number of MSOs', 'conflicts': 'Number of Conflicts', 'diagnoses': 'Number of Diagnoses'}
for group_name, dfs in group_dfs.items():
    for name, df in dfs.items():
        plot_metrics(df, f'{group_name.upper()} - Metrics by Example Complexity - {name}', save_path=f'pic/metrics_{name}.pdf')
    plot_aggregated_metrics(dfs, save_path=f'pic/metrics_{group_name}_aggregated.pdf')
    print(f"\nSummary Statistics for {group_name.upper()}:")
    print("-" * 50)
    for name, df in dfs.items():
        f1_cols = [col for col in df.columns if 'F1' in col]
        mean_f1 = df[f1_cols].values.flatten().mean()
        std_f1 = df[f1_cols].values.flatten().std()
        print(f"\nVersion: {name}")
        print(f"Average F1 score: {mean_f1:.3f} ± {std_f1:.3f}")
        print(f"Average number of relations: {df['relation_count'].mean():.1f}")
        print(f"Correlation between F1 and relations: {df['F1'].corr(df['relation_count']):.3f}")
for group_name, dfs in group_dfs.items():
    true_col = solution_columns[group_name]
    x_label = x_labels[group_name]
    for name, df in dfs.items():
        plot_metrics_by_solutions(df, f'pic/metrics_by_solutions_{name}.pdf', true_col, x_label)
    plot_aggregated_metrics_by_solutions(dfs, f'pic/metrics_by_solutions_{group_name}_aggregated.pdf', true_col, x_label)
for group_name, dfs in group_dfs.items():
    if group_name == 'mso':
        plot_aggregated_bars_mso(dfs, f'pic/metrics_bars_{group_name}.pdf', f'{group_name.upper()} Generation - Comparison of Metrics Across Models')
    else:
        plot_aggregated_bars_other(dfs, f'pic/metrics_bars_{group_name}.pdf', f'{group_name.upper()} Generation - Comparison of Metrics Across Models')
# %%
examples_df = process_examples()
print(examples_df)

# %%
