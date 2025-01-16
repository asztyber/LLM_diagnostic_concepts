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
    
    return {
        "title": title,
        "x": x,
        "f": f,
        "z": z,
        "r": r,
        "o": o
    }

def convert_to_fdt_format(parsed_data):
    fdt_format = {
        "type": "Symbolic",
        "x": parsed_data["x"],
        "f": [f"f_{fault}" for fault in parsed_data["f"]],
        "z": [f"u_{var}" for var in parsed_data["z"]],
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
    
    for var in parsed_data["z"]:
        fdt_format["rels"].append(f"-{var} + u_{var}")
    
    return fdt_format

def write_output_file(output_file, fdt_format):
    with open(output_file, 'w') as file:
        json.dump(fdt_format, file, indent=4)

def main():
    input_folder = '/home/ania/LLM_and_diagnosisv2/examples/logic'
    output_folder = '/home/ania/LLM_and_diagnosisv2/examples/fdt'
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"model_definition_{filename.split('_')[-1].split('.')[0]}.json")
            
            parsed_data = parse_input_file(input_file)
            fdt_format = convert_to_fdt_format(parsed_data)
            write_output_file(output_file, fdt_format)

if __name__ == "__main__":
    main()