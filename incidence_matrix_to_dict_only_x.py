import faultdiagnosistoolbox as fdt
import sympy as sym
import json
import os
import numpy as np

input_folder = 'examples/fdt'
output_folder = 'examples/incidence_matrices_only_x'

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def create_incidence_dict(model, var_names):
    """Convert incidence matrices to dictionary format"""
    # Concatenate all incidence matrices horizontally [X|F|Z]
    combined_matrix = model.X
    
    # Combine all variable names
    all_vars = var_names['x']
    
    # Create dictionary mapping equations to variables
    incidence_dict = {}
    for eq_idx in range(combined_matrix.shape[0]):
        # Get indices where this equation has connections (1s in the matrix)
        var_indices = np.where(combined_matrix[eq_idx, :] == 1)[0]
        # Map these indices to variable names
        connected_vars = [all_vars[i] for i in var_indices]
        # Add to dictionary
        incidence_dict[f"eq{eq_idx}"] = connected_vars
    
    return incidence_dict

# Process all JSON files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"incidence_{filename}")
        
        # Read model definition
        with open(input_file, 'r') as file:
            model_def = json.load(file)

        # Define symbolic variables
        sym.var(model_def['x'])

        # Create the diagnosis model
        model = fdt.DiagnosisModel(model_def, name=filename)
        model.Lint()

        # Convert incidence matrices to dictionary
        incidence_dict = create_incidence_dict(model, model_def)

        # Save the results
        with open(output_file, 'w') as file:
            json.dump(incidence_dict, file, indent=4)

        print(f"Processed {filename}, results saved to {output_file}")
