import json
import os
import glob

def translate_msos_to_comp_names():
    # Create output directory if it doesn't exist
    output_dir = "examples/msos_with_comp_names"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MSO files
    mso_files = glob.glob("examples/msos/msos_*.json")
    
    for mso_file in mso_files:
        # Extract the number from the filename
        file_number = mso_file.split('msos_')[1].split('.')[0]
        
        # Read the MSOs file
        with open(mso_file, "r") as f:
            msos = json.load(f)
        
        # Read the corresponding component mapping
        comp_map_file = f"examples/rel_comp_maps/rel_comp_map_{file_number}.json"
        with open(comp_map_file, "r") as f:
            comp_map = json.load(f)
        
        # Translate integers to component names
        translated_msos = []
        for mso in msos:
            translated_mso = [comp_map[str(num)] for num in mso]
            translated_msos.append(translated_mso)
        
        # Save the translated result
        output_filename = f"msos_{file_number}_with_names.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w") as f:
            json.dump(translated_msos, f, indent=4)
        
        print(f"Processed file: msos_{file_number}.json")

if __name__ == "__main__":
    translate_msos_to_comp_names()
