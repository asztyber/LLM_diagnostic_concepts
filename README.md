# Fault Diagnosis Model Processing Pipeline

This repository contains a set of Python scripts for processing and analyzing fault diagnosis models. The pipeline converts logical models to MSO (Minimal Structural Overdetermined) sets and translates them to human-readable component names.

## Scripts Overview

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

### 4. t.py
A testing/debugging script for examining FDT model properties.

- **Function**: Allows inspection of incidence matrices (X, F, Z) for a specific model


## Directory Structure

```
examples/
├── logic_and_arithmetic/     # Input model definitions (.txt)
├── fdt/                      # FDT model definitions (model_definition_N.json)
├── rel_comp_maps/            # Component mapping files (rel_comp_map_N.json)
├── msos/                     # Generated MSO sets (msos_N.json)
└── msos_with_comp_names/     # Named MSO sets (msos_N_with_names.json)
```


## Usage

1. Place your logical model definitions in `examples/logic_and_arithmetic/`
2. Run the scripts in the following order:
   ```bash
   python to_fdt.py
   python mso.py
   python msos_to_comp_names.py
   ```

## Dependencies
- faultdiagnosistoolbox
- sympy
- json
- os
- glob
- re

