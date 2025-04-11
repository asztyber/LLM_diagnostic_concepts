# %%
import pandas as pd
import json
import os
import glob

# %%
def read_mso_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split by "Name: " and filter out empty strings
    sections = [s for s in content.split('Name: ') if s.strip()]
    
    mso_dict = {}
    for section in sections:
        # Get the example number and json content
        lines = section.strip().split('\n', 1)
        example_num = int(lines[0].strip())
        json_content = lines[1].strip()
        json_data = json.loads(json_content)
        
        # Convert different formats to a standard format
        if 'MSO_sets' in json_data:
            # Format from gpt_4o_mini or o3_mini rep2
            if isinstance(json_data['MSO_sets'], dict):
                mso_sets = json_data['MSO_sets']
            else:  # list format
                mso_sets = {f'set{i+1}': s for i, s in enumerate(json_data['MSO_sets'])}
        elif 'MSO_sets' in json_data and isinstance(json_data['MSO_sets'], dict):
            # Another variant with MSO_sets
            mso_sets = json_data['MSO_sets']
        elif 'msos' in json_data:
            # Format from o3_mini rep5
            mso_sets = {k.replace('mso', 'set'): v for k, v in json_data['msos'].items()}
        elif 'MSO_set_1' in json_data:
            # Format from o3_mini rep0, rep1, rep3, rep6, rep9
            mso_sets = {}
            i = 1
            while f'MSO_set_{i}' in json_data:
                mso_sets[f'set{i}'] = json_data[f'MSO_set_{i}']
                i += 1
        elif 'MSO_1' in json_data:
            # Format from o3_mini rep4
            mso_sets = {}
            i = 1
            while f'MSO_{i}' in json_data:
                mso_sets[f'set{i}'] = json_data[f'MSO_{i}']
                i += 1
        elif 'MSO_set1' in json_data:
            # Format from o3_mini rep7
            mso_sets = {}
            i = 1
            while f'MSO_set{i}' in json_data:
                mso_sets[f'set{i}'] = json_data[f'MSO_set{i}']
                i += 1
        elif 'MSO0' in json_data:
            # Format from o1
            mso_sets = {}
            i = 0
            while f'MSO{i}' in json_data:
                mso_sets[f'set{i+1}'] = json_data[f'MSO{i}']
                i += 1
        else:
            print(f"Warning: Unknown format in example {example_num}")
            print(json_data)
            mso_sets = {}
        
        mso_dict[example_num] = {'MSO_sets': mso_sets}
    
    return mso_dict

def transform_eq_to_components(mso_dict, example_num):
    # Read the mapping file for the specific example
    mapping_file = f'examples/rel_comp_maps/rel_comp_map_{example_num}.json'
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Get the MSO sets for this example
    mso_data = mso_dict[example_num]
    transformed_sets = {}
    
    # Transform each set
    for set_name, equations in mso_data['MSO_sets'].items():
        # Convert each equation number to component name
        components = [mapping[eq[2:]] for eq in equations]  # eq[2:] removes "eq" prefix
        transformed_sets[set_name] = components
    
    return {
        "MSO_sets": transformed_sets,
    }


# %%
# Read the Excel file
excel_path = 'results/Karol/results_article/4o_mini/MSO.xlsx'
df_excel = pd.read_excel(excel_path, skiprows=2, usecols='B:BO')
df_excel['Example content'] = df_excel['Example content'].ffill()
df_excel = df_excel.drop(df_excel.index[-1])

# %%

def format_mso_sets(mso_data):
    """Convert MSO sets to string format similar to the Excel file"""
    sets = mso_data['MSO_sets']
    if not sets:  # If empty
        return ""
    
    # Convert each set to comma-separated string
    formatted_sets = []
    for components in sets.values():
        formatted_sets.append(", ".join(components))
    
    # Join all sets with newline
    return "\n".join(formatted_sets)


# %%
def process_mso_file(filepath):
    """Process a single MSO file and return a dictionary with formatted results"""
    mso_dict = read_mso_file(filepath)
    
    data = []
    for example_num, mso_data in mso_dict.items():
        # Transform equation names to component names if mapping exists
        mapping_file = f'examples/rel_comp_maps/rel_comp_map_{example_num}.json'
        if os.path.exists(mapping_file):
            transformed_data = transform_eq_to_components(mso_dict, example_num)
        else:
            print(f"Warning: No mapping file found for example {example_num}")
            transformed_data = mso_data
            
        data.append({
            'Example name': f'example_{example_num}',
            'MSO': format_mso_sets(transformed_data)
        })
    
    return pd.DataFrame(data)

# %%
# Define directories to process
directories = [
    'results/results_DX2025_Maxence/gpt_4o_mini/MSOs/',
    'results/results_DX2025_Maxence/o3_mini/MSOs/',
    'results/results_DX2025_Maxence/o1/MSOs/'
]

# Process each directory
for directory in directories:
    print(f"\nProcessing directory: {directory}")
    
    # Get all MSO files and process them
    mso_files = glob.glob(f'{directory}/*.txt')
    mso_files.sort()

    all_results = []
    for i, filepath in enumerate(mso_files):
        print(f"Processing file {i+1}/{len(mso_files)}: {os.path.basename(filepath)}")
        df = process_mso_file(filepath)
        # Rename the MSO column to include the file number
        column_name = 'Generated MSO' if i == 0 else f'Generated MSO.{i}'
        df = df.rename(columns={'MSO': column_name})
        all_results.append(df)

    if all_results:
        # Merge all results on Example name
        final_df = all_results[0]
        for df in all_results[1:]:
            final_df = final_df.merge(df, on='Example name', how='outer')

        # Sort by Example name
        final_df = final_df.sort_values('Example name')

        # Clean up and merge with Excel data
        excel_df = df_excel[['Example name', 'Example content', 'MSO']].copy()

        # Merge with our processed results
        final_df = final_df.merge(excel_df, on='Example name', how='outer')

        # Reorder columns to have Example content and Original MSO first
        cols = ['Example name', 'Example content', 'MSO'] + [col for col in final_df.columns if col not in ['Example name', 'Example content', 'MSO']]
        final_df = final_df[cols]

        # Save the final DataFrame
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(directory)))
        output_path = f'results/results_DX2025_Maxence/{dir_name}/MSO.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
        print(f"Saved results to: {output_path}")

        # Display first few rows
        print("\nFirst few rows of the processed data:")
        print(final_df.head())
    else:
        print(f"No files found in {directory}")

# %%
