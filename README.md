# Are diagnostic concepts within the reach of LLMs?

This repo accompanies the paper:

A. Sztyber-Betley, E. Chanthery, L. Travé-Massuyès, S. Merkelbach, K. Kukla, M. Glotin, A. Diedrich, and O. Niggemann Are diagnostic concepts within the reach of LLMs? submitted to the 36th International Conference on Principles of Diagnosis and Resilient Systems (DX’25) 

It includes prompts, all results, scripts used to produce the summary of results and helper scripts for format conversions.

## Directory Structure

```
.
├── examples/
│   ├── fdt/                      # FDT model definitions (model_definition_N.json)
│   ├── incidence_matrices/       # Correct incidence matrices
│   ├── incidence_matrices_only_x/ # Correct incidence matrices (only X)
│   ├── logic_and_arithmetic/     # Input model definitions (.txt)
│   ├── msos/                     # Correct MSO sets (msos_N.json)
│   ├── msos_with_comp_names/     # Named correct MSO sets (msos_N_with_names.json)
│   └── rel_comp_maps/            # Component mapping files (rel_comp_map_N.json)
├── pic/                          # Directory for plots and images
├── prompts/                      # Directory for LLM prompts
│   ├── broad/
│   └── dedicated/
├── results/                      # Directory for experiment results
│   ├── Karol/
│   └── results_DX2025_Maxence/
├── incidence_matrices_results.py       # Processing of incidence matrices results
├── incidence_matrix_to_dict_only_x.py  # Correct incidence matrix generation (only X variables)
├── incidence_matrix_to_dict.py         # Correct incidence matrix generation
├── mso_format_conversion.py            # Preprocessing of results from broad approach
├── mso.py                              # Generation of correct MSOs
├── msos_to_comp_names.py               # MSO with component names instead of numbers
├── plots_results.py                    # Results summary and plots
├── README.md
└── to_fdt.py                           # Conversion on .txt files into fdt format
```



## Overview of format processing scripts

### 1. to_fdt.py
Converts logical model definitions to the Fault Diagnosis Toolbox (FDT) format.

- **Input**: Text files in `examples/logic_and_arithmetic/` with model definitions
- **Output**: 
  - FDT model definitions in `examples/fdt/model_definition_N.json`
  - Component mappings in `examples/rel_comp_maps/rel_comp_map_N.json`
- **Function**: Parses text files containing model definitions (variables, faults, relations) and converts them to the FDT JSON format

### 2. mso.py
Processes FDT model definitions to generate MSO sets using the Fault Diagnosis Toolbox.

- **Input**: FDT model definitions from `examples/fdt/`
- **Output**: MSO sets in `examples/msos/msos_N.json`
- **Function**: 
  - Creates diagnosis models using FDT
  - Computes Minimal Structural Overdetermined sets
  - Saves results as numbered JSON files

### 3. msos_to_comp_names.py
Translates numerical MSO representations to human-readable component names.

- **Input**: 
  - MSO sets from `examples/msos/`
  - Component mappings from `examples/rel_comp_maps/`
- **Output**: Named MSO sets in `examples/msos_with_comp_names/msos_N_with_names.json`
- **Function**: Maps numerical identifiers to component names for better readability


### 4. incidence_matrix_to_dict.py
Converts the full incidence matrix (X, F, Z) of a model to a dictionary format.

- **Input**: FDT model definitions from `examples/fdt/`
- **Output**: JSON files in `examples/incidence_matrices/` containing the incidence matrices as dictionaries.
- **Function**: For each model, it computes the X, F, and Z matrices and saves them as a dictionary in a JSON file.

### 5. incidence_matrix_to_dict_only_x.py
Converts only the X part (analytical redundancy relations vs unknown variables) of the incidence matrix to a dictionary format.

- **Input**: FDT model definitions from `examples/fdt/`
- **Output**: JSON files in `examples/incidence_matrices_only_x/` containing the X matrix as a dictionary.
- **Function**: For each model, it computes the X matrix and saves it as a dictionary in a JSON file.


