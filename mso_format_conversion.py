# %%
import pandas as pd
import json
import os
import glob

# --- MSO File Reading and Conversion ---
def read_mso_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    sections = [s for s in content.split('Name: ') if s.strip()]
    mso_dict = {}
    for section in sections:
        lines = section.strip().split('\n', 1)
        example_num = int(lines[0].strip())
        json_content = lines[1].strip()
        json_data = json.loads(json_content)
        # Standardize MSO set formats
        if 'MSO_sets' in json_data:
            if isinstance(json_data['MSO_sets'], dict):
                mso_sets = json_data['MSO_sets']
            else:
                mso_sets = {f'set{i+1}': s for i, s in enumerate(json_data['MSO_sets'])}
        elif 'msos' in json_data:
            mso_sets = {k.replace('mso', 'set'): v for k, v in json_data['msos'].items()}
        elif 'MSO_set_1' in json_data:
            mso_sets = {f'set{i}': json_data[f'MSO_set_{i}'] for i in range(1, 100) if f'MSO_set_{i}' in json_data}
        elif 'MSO_1' in json_data:
            mso_sets = {f'set{i}': json_data[f'MSO_{i}'] for i in range(1, 100) if f'MSO_{i}' in json_data}
        elif 'MSO_set1' in json_data:
            mso_sets = {f'set{i}': json_data[f'MSO_set{i}'] for i in range(1, 100) if f'MSO_set{i}' in json_data}
        elif 'MSO0' in json_data:
            mso_sets = {f'set{i+1}': json_data[f'MSO{i}'] for i in range(0, 100) if f'MSO{i}' in json_data}
        else:
            print(f"Warning: Unknown format in example {example_num}")
            print(json_data)
            mso_sets = {}
        mso_dict[example_num] = {'MSO_sets': mso_sets}
    return mso_dict

def transform_eq_to_components(mso_dict, example_num):
    mapping_file = f'examples/rel_comp_maps/rel_comp_map_{example_num}.json'
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    mso_data = mso_dict[example_num]
    transformed_sets = {set_name: [mapping[eq[2:]] for eq in equations] for set_name, equations in mso_data['MSO_sets'].items()}
    return {"MSO_sets": transformed_sets}

def format_mso_sets(mso_data):
    sets = mso_data['MSO_sets']
    if not sets:
        return ""
    return "\n".join(", ".join(components) for components in sets.values())

def process_mso_file(filepath):
    mso_dict = read_mso_file(filepath)
    data = []
    for example_num, mso_data in mso_dict.items():
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

# --- Excel Reference Data ---
excel_path = 'results/Karol/results_article/4o_mini/MSO.xlsx'
df_excel = pd.read_excel(excel_path, skiprows=2, usecols='B:BO')
df_excel['Example content'] = df_excel['Example content'].ffill()
df_excel = df_excel.drop(df_excel.index[-1])

# --- Directory Processing ---
directories = [
    'results/results_DX2025_Maxence/gpt_4o_mini/MSOs/',
    'results/results_DX2025_Maxence/o3_mini/MSOs/',
    'results/results_DX2025_Maxence/o1/MSOs/'
]
for directory in directories:
    print(f"\nProcessing directory: {directory}")
    mso_files = sorted(glob.glob(f'{directory}/*.txt'))
    all_results = []
    for i, filepath in enumerate(mso_files):
        print(f"Processing file {i+1}/{len(mso_files)}: {os.path.basename(filepath)}")
        df = process_mso_file(filepath)
        column_name = 'Generated MSO' if i == 0 else f'Generated MSO.{i}'
        df = df.rename(columns={'MSO': column_name})
        all_results.append(df)
    if all_results:
        final_df = all_results[0]
        for df in all_results[1:]:
            final_df = final_df.merge(df, on='Example name', how='outer')
        final_df = final_df.sort_values('Example name')
        excel_df_sub = df_excel[['Example name', 'Example content', 'MSO']].copy()
        final_df = final_df.merge(excel_df_sub, on='Example name', how='outer')
        cols = ['Example name', 'Example content', 'MSO'] + [col for col in final_df.columns if col not in ['Example name', 'Example content', 'MSO']]
        final_df = final_df[cols]
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(directory)))
        output_path = f'results/results_DX2025_Maxence/{dir_name}/MSO.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
        print(f"Saved results to: {output_path}")
        print("\nFirst few rows of the processed data:")
        print(final_df.head())
    else:
        print(f"No files found in {directory}")
# %%
