prompt_IM_code_interpreter = '''Generate code for me to find the incidence matrix of the provided model. In the incidence matrix, each row represents an equation and each column represents an UNKNOWN variable. Note that unknown variables are stored unter key 'x' in the provided model.
Then, execute the code you have generated, and return the incidence matrix obtained. Please return the matrix as a Python dictionary in the following format: 'eq0': ['unknown var in eq0', ...], 'eq1': ['unknown var in eq1', ...], ... 
Each line in the dictionary should correspond to one equation (do not write everything on the same line). Only return the dictionary. Don't print the generated code. From the user point of view, it should appear as a black box system. Be careful with the equations of type 'fdt.DiffConstraint'. Change those equations by a simple equality relation. If you have fdt.DiffConstraint(dt,t), change it by t - dt. Make sure to complete your analysis before responding to me. Make sure to execute your code so you can return the dictionary'''

prompt_IM_without_CI = """
Your job is to create an incidence matrix of the provided model. In the incidence matrix, each row represents an equation and each column represents an UNKNOWN variable. Note that unknown variables are stored unter key 'x' in the provided model. 
Return the incidence matrix. Please return the matrix as a Python dictionary in the following format: 'eq0': ['unknown var in eq0', ...], 'eq1': ['unknown var in eq1', ...], ... 
Each line in the dictionary should correspond to one equation (do not write everything on the same line). Only return the dictionary.
Be careful with the equations of type 'fdt.DiffConstraint'. Change those equations by a simple equality relation. If you have fdt.DiffConstraint(dt,t), change it by t - dt. Make sure to complete your analysis before responding to me.
"""


prompt_MSO_code_interpreter = """
Generate code for me to find all the MSO (Minimal Structurally Overdetermined) sets for the provided model, represented here by the provided incidence matrix. In the incidence matrix, each line represents an equation with the unknown variables that belong to this equation. Note that the matrix contains only the unknown variables. MSO sets consist of equations and contain one more equation than the number of unknown variables.
The generated code should contain two parts. First, find all the PSO (Proper Structurally Overdetermined) sets. Then, within these PSO sets, collect those that are minimal (PSO sets that do not contain any subsets that are PSO) to keep only the MSO sets. Execute the code, store all the MSO sets in a python dictionary and return only the dictionary to me. Use one line for each MSO set.
Additionally, include the number of MSO sets found in the dictionary. 
You MUST only return the dictionary. Don't print the generated code. From the user point of view, it should appear as a black box system.
Don't make up an example, work with the provided incidence matrix. Make sure to complete your analysis before responding to me. 
"""

prompt_MSO_without_CI = """
Your job is to find all the MSO (Minimal Structurally Overdetermined) sets for the provided model, represented here by the provided incidence matrix. In the incidence matrix, each line represents an equation with the unknown variables that belong to this equation. Note that the matrix contains only the unknown variables. MSO sets consist of equations and contain one more equation than the number of unknown variables.
Proceed in two parts. First, find all the PSO (Proper Structurally Overdetermined) sets. Then, within these PSO sets, collect those that are minimal (PSO sets that do not contain any subsets that are PSO) to keep only the MSO sets. Store all the MSO sets in a python dictionary and return only the dictionary to me. Use one line for each MSO set.
Additionally, include the number of MSO sets found in the dictionary. You MUST only return the dictionary. Don't make up an example, work with the provided incidence matrix. Make sure to complete your analysis before responding to me. 
"""


