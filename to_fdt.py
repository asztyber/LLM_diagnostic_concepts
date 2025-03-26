import re
import json
import os

def parse_input_file(input_file):
    with open(input_file, 'r') as file:
        content = file.read()
    
    title = re.search(r"title\s*=\s*'(.*?)'", content).group(1)
    x = re.search(r"x\s*=\s*\[(.*?)\]", content).group(1).replace("'", "").split(", ")
    f = re.search(r"f\s*=\s*\[(.*?)\]", content).group(1).replace("'", "").split(", ")
    z = re.search(r"z\s*=\s*\[(.*?)\]", content).group(1).replace("'", "").split(", ")
    
    r_match = re.search(r"r\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if r_match:
        r = [rel.strip().replace("'", "") for rel in r_match.group(1).split('\n') if rel.strip()]
    else:
        r = []
    
    o_match = re.search(r"o\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if o_match:
        o = [list(map(int, row.strip().replace("[", "").replace("]", "").split(", "))) for row in o_match.group(1).split('\n') if row.strip()]
    else:
        o = []
    
    # Create relation to component mapping separately
    rel_to_comp = {}
    for idx, rel in enumerate(r):
        match = re.match(r"^(\w+):", rel)
        if match:
            rel_to_comp[idx] = match.group(1)
    
    return {
        "title": title,
        "x": x,
        "f": f,
        "z": z,
        "r": r,
        "o": o
    }, rel_to_comp

def convert_to_fdt_format(parsed_data):
    fdt_format = {
        "type": "Symbolic",
        "x": [x for x in parsed_data["x"] if x not in parsed_data["z"]],
        "f": [f"f_{fault}" for fault in parsed_data["f"]],
        "z": parsed_data["z"],
        "rels": []
    }
    
    for rel in parsed_data["r"]:
        match = re.match(r"^(\w+):\s*(.*)", rel)
        if match:
            fault_indicator = match.group(1)
            equation = match.group(2).replace("=", "-") + f" + f_{fault_indicator}"
        else:
            equation = rel.replace("=", "-")
        fdt_format["rels"].append(equation.strip().replace(",", ""))
    
    return fdt_format

def write_output_file(output_file, fdt_format):
    with open(output_file, 'w') as file:
        json.dump(fdt_format, file, indent=4)

def write_relation_mapping(output_file, rel_to_comp):
    with open(output_file, 'w') as file:
        json.dump(rel_to_comp, file, indent=4)

def main():
    input_folder = 'examples/logic_and_arithmetic'
    output_folder = 'examples/fdt'
    maps_output_folder = 'examples/rel_comp_maps'
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"model_definition_{filename.split('_')[-1].split('.')[0]}.json")
            maps_file = os.path.join(maps_output_folder, f"rel_comp_map_{filename.split('_')[-1].split('.')[0]}.json")
            parsed_data, rel_to_comp = parse_input_file(input_file)
            fdt_format = convert_to_fdt_format(parsed_data)
            
            write_output_file(output_file, fdt_format)
            write_relation_mapping(maps_file, rel_to_comp)

if __name__ == "__main__":
    main()