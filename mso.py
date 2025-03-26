import faultdiagnosistoolbox as fdt
import sympy as sym
import json
import os

input_folder = 'examples/fdt'
output_folder = 'examples/msos'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"msos_{filename.split('_')[-1].split('.')[0]}.json")
        
        with open(input_file, 'r') as file:
            model_def = json.load(file)

        # Define symbolic variables
        sym.var(model_def['x'])
        sym.var(model_def['f'])
        sym.var(model_def['z'])

        # Create the diagnosis model
        model = fdt.DiagnosisModel(model_def, name=filename)
        model.Lint()

        # Compute MSOs
        msos = model.MSO()
        fsm = model.FSM(msos)

        # Save the results
        with open(output_file, 'w') as file:
            json.dump(msos, file, indent=4)

        print(f"Processed {filename}, results saved to {output_file}")