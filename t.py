#%%
import faultdiagnosistoolbox as fdt
import sympy as sym
import json

#%%
filename = 'model_definition_2.json'    
input_file = 'examples/fdt/' + filename
with open(input_file, 'r') as file:
    model_def = json.load(file)

# Define symbolic variables
sym.var(model_def['x'])
sym.var(model_def['f'])
sym.var(model_def['z'])

# Create the diagnosis model
model = fdt.DiagnosisModel(model_def, name=filename)
model.Lint()

#%%
# Here we have incidence matrices
# rels to x
model.X
# %%
# rels to f
model.F
# %%
# x to z
model.Z
# %%
