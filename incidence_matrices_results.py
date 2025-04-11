#%%
import json
from typing import Dict, List, Set
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#%%
def read_examples_from_file(filepath):
    examples = {}
    current_name = None
    current_content = ""
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Name: "):
                # If we have a previous example, save it
                if current_name is not None:
                    try:
                        examples[current_name] = json.loads(current_content)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON for example {current_name}")
                
                # Start new example
                current_name = line.split("Name: ")[1]
                current_content = ""
            elif line and not line.isspace() and current_name is not None:
                current_content += line
                
    # Don't forget to add the last example
    if current_name is not None and current_content:
        try:
            examples[current_name] = json.loads(current_content)
        except json.JSONDecodeError:
            print(f"Error parsing JSON for example {current_name}")
    
    return examples

# %%
def load_correct_solution(filepath: str) -> Dict[str, List[str]]:
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_equation_accuracy(generated: Dict[str, List[str]], 
                           correct: Dict[str, List[str]]) -> float:
    """
    Compute percentage of equations that match exactly (as sets)
    """
    if not generated or not correct:
        return 0.0
    
    correct_count = 0
    total_equations = len(correct)
    
    for eq_name, correct_vars in correct.items():
        if eq_name in generated:
            if set(correct_vars) == set(generated[eq_name]):
                correct_count += 1
    
    return (correct_count / total_equations) * 100

def compute_precision_recall(generated: Dict[str, List[str]], 
                           correct: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision and recall for each equation
    If both sets are empty, precision and recall are set to 1.0
    """
    results = {}
    
    for eq_name, correct_vars in correct.items():
        correct_set = set(correct_vars)
        
        if eq_name not in generated:
            results[eq_name] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            continue
            
        generated_set = set(generated[eq_name])
        
        # If both sets are empty, consider it a perfect match
        if len(correct_set) == 0 and len(generated_set) == 0:
            results[eq_name] = {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            continue
        
        # True positives are variables that appear in both sets
        true_positives = len(correct_set.intersection(generated_set))
        
        # Precision = true positives / total predicted
        precision = true_positives / len(generated_set) if generated_set else 0.0
        
        # Recall = true positives / total actual
        recall = true_positives / len(correct_set) if correct_set else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[eq_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results

#%%
# Process results from all three folders
folders = {
    '4o_mini': "results/results_DX2025_Maxence/gpt_4o_mini/IM/",
    'o3_mini': "results/results_DX2025_Maxence/o3_mini/IM/",
    'o1': "results/results_DX2025_Maxence/o1/IM/"
}

all_model_results = {}

for model_name, results_dir in folders.items():
    print(f"\nProcessing model: {model_name}")
    result_files = glob.glob(os.path.join(results_dir, "*.txt"))
    
    all_file_results = {}
    
    for result_file in result_files:
        file_name = os.path.basename(result_file)
        print(f"Processing {file_name}")
        
        # Read the examples from this file
        examples = read_examples_from_file(result_file)
        
        # Process all examples from 2 to 24
        file_results = {}
        
        for example_num in range(2, 25):
            correct_solution_path = f'examples/incidence_matrices_only_x/incidence_model_definition_{example_num}.json'
            
            # Skip if file doesn't exist
            if not os.path.exists(correct_solution_path):
                continue
                
            # Load correct solution
            correct_solution = load_correct_solution(correct_solution_path)
            
            # Get generated solution
            example_key = str(example_num)
            if example_key not in examples:
                print(f"Warning: Example {example_key} not found in generated solutions")
                continue
                
            generated_solution = examples[example_key]
            
            # Compute metrics
            accuracy = compute_equation_accuracy(generated_solution, correct_solution)
            metrics = compute_precision_recall(generated_solution, correct_solution)
            
            # Calculate averages for this example
            avg_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
            avg_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
            avg_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
            
            file_results[example_num] = {
                'accuracy': accuracy,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1
            }
        
        # Calculate overall averages for this file
        if file_results:  # Only if we have results
            overall_avg = {
                'accuracy': sum(r['accuracy'] for r in file_results.values()) / len(file_results),
                'precision': sum(r['avg_precision'] for r in file_results.values()) / len(file_results),
                'recall': sum(r['avg_recall'] for r in file_results.values()) / len(file_results),
                'f1': sum(r['avg_f1'] for r in file_results.values()) / len(file_results)
            }
            
            all_file_results[file_name] = {
                'results': file_results,
                'overall_avg': overall_avg
            }
    
    all_model_results[model_name] = all_file_results

#%%
# Print summary of results for all models
print("\nOverall results for all models:")
print("Model | File | Accuracy | Precision | Recall | F1")
print("-" * 100)

for model_name, file_results in sorted(all_model_results.items()):
    print(f"\n{model_name}:")
    print("-" * 100)
    
    model_avgs = []
    for file_name, results in sorted(file_results.items()):
        avg = results['overall_avg']
        print(f"{model_name:8s} | {file_name[:30]:30s} | {avg['accuracy']:8.2f} | {avg['precision']:9.2f} | {avg['recall']:6.2f} | {avg['f1']:.2f}")
        model_avgs.append(avg)
    
    # Calculate and print average for this model
    if model_avgs:
        model_final_avg = {
            'accuracy': sum(avg['accuracy'] for avg in model_avgs) / len(model_avgs),
            'precision': sum(avg['precision'] for avg in model_avgs) / len(model_avgs),
            'recall': sum(avg['recall'] for avg in model_avgs) / len(model_avgs),
            'f1': sum(avg['f1'] for avg in model_avgs) / len(model_avgs)
        }
        print("-" * 100)
        print(f"{model_name:8s} | {'AVERAGE':30s} | {model_final_avg['accuracy']:8.2f} | {model_final_avg['precision']:9.2f} | {model_final_avg['recall']:6.2f} | {model_final_avg['f1']:.2f}")

#%%
# Store full results
all_model_results

# %%
def plot_model_results(all_model_results, save_path='pic/im_bars.pdf'):
    """Create a bar plot showing aggregated F1, Precision, Recall, and Accuracy scores"""
    plt.figure(figsize=(12, 5))
    sns.set_style("whitegrid")
    
    # Colors for different models
    colors = ['#2E86C1', '#28B463', '#E74C3C']  # Blue, Green, Red
    
    # Width of each bar and positions
    bar_width = 0.25
    metrics = ['F1', 'Precision', 'Recall', 'Accuracy']
    x = np.arange(len(metrics))
    
    # Model names mapping for legend
    model_labels = {
        '4o_mini': '4o-mini',
        'o1': 'o1',
        'o3_mini': 'o3-mini'
    }
    
    # Calculate metrics and their standard deviations for each model
    results = {}
    for model_name, file_results in all_model_results.items():
        # Collect all values for each metric and example
        all_values = {
            'F1': [],
            'Precision': [],
            'Recall': [],
            'Accuracy': []
        }
        
        # Gather values across all files and examples
        for file_data in file_results.values():
            for example_results in file_data['results'].values():
                all_values['F1'].append(example_results['avg_f1'])
                all_values['Precision'].append(example_results['avg_precision'])
                all_values['Recall'].append(example_results['avg_recall'])
                all_values['Accuracy'].append(example_results['accuracy'] / 100)  # Convert to same scale
        
        # Calculate mean and std for each metric
        results[model_name] = {}
        for metric in metrics:
            values = np.array(all_values[metric])
            results[model_name][metric] = np.mean(values)
            results[model_name][f'{metric}_std'] = np.std(values)
    
    # Set y-axis limits
    plt.ylim(-0.05, 1.05)
    
    # Plot bars for each model
    models = ['4o_mini', 'o1', 'o3_mini']
    for i, (model, color) in enumerate(zip(models, colors)):
        if model not in results:
            continue
            
        offset = (i - 1) * bar_width
        values = [results[model][m] for m in metrics]
        errors = [results[model][m + '_std'] for m in metrics]
        
        # Clip error bars to stay within plot limits
        errors = [min(e, 1 - v) for e, v in zip(errors, values)]
        
        bars = plt.bar(x + offset, values, bar_width,
                      yerr=errors,
                      color=color, alpha=0.7,
                      label=model_labels[model],  # Use the mapped label
                      capsize=3,
                      error_kw={'elinewidth': 1.5})
        
        # Add annotations at the middle of each bar
        for j, (v, e) in enumerate(zip(values, errors)):
            plt.text(x[j] + offset, v/2,  # Position at middle of bar
                    f'{v:.2f}±{e:.2f}',
                    ha='center', va='center',
                    fontsize=8)
    
    # Customize the plot
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, metrics, fontsize=10)
    
    # Add legend
    plt.legend(fontsize=10, loc='upper center',
              bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=True)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#%%
# Create the plot
plot_model_results(all_model_results)

# %%
