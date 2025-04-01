#%%
import faultdiagnosistoolbox as fdt
import sympy as sym
import json


# %%
filename = 'model_definition_2.json'    
input_file = 'examples/fdt/' + filename
with open(input_file, 'r') as file:
    model_def = json.load(file)

# Define symbolic variables
sym.var(model_def['x'])
sym.var(model_def['f'])
sym.var(model_def['z'])

new_rels = []
for rel in model_def['rels']:
    # Split by common operators
    tokens = rel.replace('*', ' ').replace('+', ' ').replace('-', ' ').replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('&', ' ').replace('|', ' ').replace('~', ' ')
    # Split into words and remove empty strings
    variables = [token.strip() for token in tokens.split() if token.strip()]
    # Remove numeric values
    variables = [var for var in variables if not var.replace('.', '').isdigit()]
    new_rels.append(variables)
    
model_def['rels'] = new_rels
model = fdt.DiagnosisModel(model_def, name=filename)
model.Lint()
# %%
# Here we have incidence matrices
# rels to x
print(model.X)
# %%
# rels to f
print(model.F)
# %%
# x to z
print(model.Z)
# %%
