GPT_INSTRUCTION_PART_1 = """
Perform an analysis of the equation system equations and identify all Minimal Structurally Overdetermined (MSO) sets.
Definitions:
1. Unknown variables: These are only the variables that start with "x".
2. Structural redundancy: This is the difference between the number of equations and the number of unique unknown variables in those equations: R=(number of equations)−(number of unique unknowns).
3. PSO (Properly Structurally Observable): A subset is PSO if its structural redundancy is greater than 0.
Conditions:
1. The selected set of equations must have exactly one structural redundancy.
2. None of its proper subsets can be PSO (all must have redundancy ≤ 0).
Output:
JSON format: { "mso": [...] }, where each inner list represents a valid MSO set.
RESPONSE MUST BE IN JSON FORMAT. DO NOT GENERATE CODE AND RETURN IT IN RESPONSE.
<example>
<input>
equations = {
'M1': 'a * c = x01',
'M2': 'b * d = x02',
'M3': 'c * e = x03',
'A2': 'x01 + x02 = f',
'A1': 'x02 + x03 = g',
}
</input>
<output>
{
"mso" = [['M1', 'M2', 'A1'], ['M2', 'M3', 'A2'], ['M1', 'M3', 'A1', 'A2']]
}
</output>
</example>
"""
GPT_INSTRUCTION_PART_2 = """
Imagine you are an engineer specialized in fault diagnosis.
Create a class with the following components:
Initialize the class with three parameters:
    A list of lists containing equation names (call it 'mso')
    A dictionary where keys are equation names and values are equation strings (call it 'equations')
    A dictionary of observed data (call it 'data')
    Store these as instance variables with the same names.
    Create an additional instance variable called 'parsed_equations' by calling a method that defines and parses the equations.
Implement a private method that defines and parses equations:
    Initialize an empty dictionary called 'parsed'
    Iterate through each item in the 'equations' dictionary
    For each equation:
        Split the equation string by '=' into 'left_side' and 'right_side'
        Strip whitespace from both sides
        Create a 'symbols_dict' using dictionary comprehensions:
            For the left side: {var: symbols(var) for var in left_side.split() if var.isidentifier()}
            Update it with right side: {var: symbols(var) for var in right_side.split() if var.isidentifier()}
        Replace observed data in the symbols_dict:
            For each item in 'data', if the key is in symbols_dict, replace the value
        Build the equation based on the operator:
            Check for '&', '|', '^', '~^', '~', '~&', '~|' in the left_side
                For your information what means these symbols :
                    NAND ('~&')
                    NOR ('~|')
                    XNOR ('~^')
                    XOR ('^')
                    NOT ('~')
                    AND ('&')
                    OR ('|')
            For each case, create a parsed_expression using SymPy functions (Not, And, Or, Xor)
            Use list comprehensions to get the relevant symbols from symbols_dict
            Convert to equality using Eq(parsed_expression, symbols_dict[right_side])
        For arithmetic:
            If '+' in left_side: split by '+', use sum() with a generator expression
            If '' in left_side: split by '', use a loop to multiply terms
            Otherwise, assume it's a single term
            Create an Eq object with left_expr and right_expr
        Add the parsed expression to the 'parsed' dictionary
    Return the 'parsed' dictionary
Implement a private method to check if a set of equations is contradictory:
    Check if equations are logical or arithmetic:
        Use: any(char in equation for equation in self.equations.values() for char in ['~', '&', '|', '^'])
    For logical equations:
        Get all variables: {var for eq in equations.values() for var in eq.free_symbols}
        Generate all True/False combinations: product([True, False], repeat=len(variables))
        For each combination:
            Create an assignment dictionary using dict(zip(variables, combination))
            Check if all equations are satisfied using all() and a generator expression
            If any combination satisfies all equations, return False
        If no satisfying combination is found, return True
    For arithmetic equations:
        Convert equations.values() to a list
        Use SymPy's solve function: solve(eqs, dict=True)
        Return True if the solution list is empty, False otherwise
Implement a public method to get minimal conflicts:
    Initialize an empty list called 'minimal_conflicts'
    Iterate through each group in self.mso
    For each group:
        Create a dictionary 'group_equations' using a dictionary comprehension:
        {name: self.parsed_equations[name] for name in group if name in self.parsed_equations}
        Check if the group is contradictory using the method from step 3
        If contradictory, append the sorted group to 'minimal_conflicts'
Return 'minimal_conflicts' sorted using: sorted(minimal_conflicts, key=lambda x: (len(x), x))

HINTS:
1. To check whether the equation is logical, check the presence of at least one character from the list ['|', '&', '^', '~'] in at least one provided equations.values().
2. Be sure that all logical opearation (AND (&), OR (|), XOR (^), NOT (~), NAND (~&), NOR (~|), XNOR (~^)) and arithmetic (+, *) are supported.
3. Use example_1, example_2 and example_3 to test your code before generate results for given data.

'minimal_conflicts' MUST BE IN JSON FORMAT!
RESPONSE MUST BE IN JSON FORMAT. DO NOT GENERATE CODE AND RETURN IT IN RESPONSE.

<example_1>
<input>
equations = {
    'M1': 'a * c = x01', 
    'M2': 'b * d = x02', 
    'M3': 'c * e = x03', 
    'A1': 'x01 + x02 = f', 
    'A2': 'x02 + x03 = g'}
mso = [['M1', 'M2', 'A1'], ['M2', 'M3', 'A2'], ['M1', 'M3', 'A1', 'A2']]
data = {'a': 2, 'b': 2, 'c': 3, 'd': 3, 'e': 2, 'f': 10, 'g': 12}
</input>
For ['M1', 'M2', 'A1'] system of equations is:
[a * c = x01,
b * d = x02,
x01 + x02 = f
]
For ['M2', 'M3', 'A2'] system of equations is:
[b * d = x02,
c * e = x03,
x02 + x03 = g
]
For ['M1', 'M3', 'A1', 'A2'] system of equations is:
[a * c = x01,
c * e = x03,
x01 + x02 = f,
x02 + x03 = g
]
For ['M1', 'M2', 'A1'] and ['M1', 'M3', 'A1', 'A2'] systems of eqautions are inconsistent so output should be:
<output>
{
'minimal_conflicts': [['M1', 'M2', 'A1'], ['M1', 'M3', 'A1', 'A2']]
}
</output>
</example_1>
<example_2>
<input>
mso = [['A1', 'O1', 'O2'], ['A2', 'O2', 'O3'], ['A1', 'A2', 'O1', 'O3']]
equations = {
    'O1': 'a | c = x01',
    'O2': 'b | d = x02',
    'O3': 'c | e = x03',
    'A1': 'x01 & x02 = f',
    'A2': 'x02 & x03 = g'}
data = {'a': True, 'b': True, 'c': True, 'd': True, 'e': True, 'f': False, 'g': False}
</input>
For ['A1', 'O1', 'O2'] system of equations is:
[x01 & x02 = f,
a | c = x01,
b | d = x02
]
For ['A2', 'O2', 'O3'] system of equations is:
[x02 & x03 = g,
b | d = x02,
c | e = x03
]
For ['A1', 'A2', 'O1', 'O3'] system of equations is:
[x01 & x02 = f,
x02 & x03 = g,
a | c = x01,
c | e = x03
]
For ['A1', 'O1', 'O2'] and ['A2', 'O2', 'O3'] systems of eqautions are inconsistent so output should be:
<output>
{
'minimal_conflicts': [['A1', 'O1', 'O2'], ['A2', 'O2', 'O3']]
}
</example_2>
<example_3>
<input>
mso = [['NA1', 'NO2', 'NX1']]
equations = {
    'NA1': 'a ~& b = x01', 
    'NO2': 'c ~| d = x02', 
    'NX1': 'x01 ~^ x02 = e'}
data = {'a': True, 'b': True, 'c': True, 'd': True, 'e': False}
</input>
For ['NA1', 'NO2', 'NX1'] system of equations is:
[a ~& b = x01,
c ~| d = x02,
x01 ~^ x02 = e
]
For ['NA1', 'NO2', 'NX1'] system of eqautions are inconsistent so output should be:
<output>
{
'minimal_conflicts': [['NA1', 'NO2', 'NX1']]
}
</example_3>
"""
GPT_INSTRUCTION_PART_3 = """
Imagine you are an engineer specialized in fault diagnosis.
Use the 'minimal_conflicts' which contains lists of equation symbols
The final result should be in JSON format and should consist of 'minimal_diagnoses'.
Use the algorithm below to generate minimal diagnoses.
Translate the algorithm steps into Python functions:
- Convert the high-level pseudocode into Python functions, maintaining the same logic and operations.
- Implement the main loop of the algorithm, handling conflicts and candidates (diagnoses) generation.
- Implement the logic to update the candidates (diagnoses) collection and ensure duplicates and non-minimal elements are removed.
In <output></output> you have example format of output for this part.
RESPONSE MUST BE IN JSON FORMAT. DO NOT GENERATE CODE AND RETURN IT IN RESPONSE.
<algorithm>
Algorithm 1: Conflicts guide candidates generation.
Inputs: MinimalConflicts
CandidatesCollection←{∅}
for each Conflict ∈ MinimalConflicts do
CurrentCandidates←CandidatesCollection
for each Candidate ∈ CurrentCandidates do
if Candidate ∩ Conflict = ∅ then
CandidatesCollection←UpdateCandidates(Candidate, CandidatesCollection, Conflict)
Return CandidatesCollection

Algorithm 2: UpdateCandidates.
Inputs: Candidate, CandidatesCollection, Conflict
CandidatesCollection←CandidatesCollection - Candidate
for each Component ∈ Conflict do
NewCandidate←Candidate ∪ { Component }
CandidatesCollection←CandidatesCollection ∪ NewCandidate
Remove duplicates and non-minimal elements from CandidatesCollection
Return CandidatesCollection

</algorithm>
<input>
{
"minimal_conflicts": [['M1', 'M2', 'A1'], ['M1', 'M3', 'A1', 'A2']]
}
</input>
<output>
{
"minimal_diagnoses": [['A1'], ['M1'], ['A2', 'M2'], ['M2', 'M3']]
}
</output>
"""
GPT_INSTRUCTION = """
Imagine you are an engineer specialized in fault diagnosis. Your task is divided into three parts.
You have to complete the first part before the second part, and the second part before the third part.
Use the result from the first part to get the result for the second part and use the second part results to get
the final result. The final result should be in JSON format and should consist of 'mso' from the first part and
'minimal_conflicts' from the second part and 'minimal_diagnoses' from the third part. You can use the code 
interpreter but in the answer, I want only JSON with two keys 'minimal_conflicts' and 'minimal_diagnoses'.
<part1>
Use 'equations,' a dictionary where key is a symbol, value is a equation, to build
the system. Analyze all possible combinations of symbols and choose those which meet the following conditions:
1. The set of equations must have exactly one structural redundancy. Structural redundancy is defined as the number 
of equations minus the number of unknown variables that are present in these equations. 
2. All proper subsets of this set must not be PSO. This means that the structural redundancy of each subset must be 0 or less.
Unknowns are the variables that appear in the equations but are not keys in the 'data' dictionary.
Check the number of unknowns at the end as they may be repeated
The result is the list of lists with symbols of equations that satisfy the above conditions.
[USE CODE INTERPRETER]
<example>
<input>
equations = [
('M1', 'a * c = x01'),
('M2', 'b * d = x02'),
('M3', 'c * e = x03'),
('A2', 'x01 + x02 = f'),
('A1', 'x02 + x03 = g')
]
data = {'a': 2, 'b': 2, 'c': 3, 'd': 3, 'e': 2, 'f': 12, 'g': 12}
</input>
<output>
{
"mso" = [['M1', 'M2', 'A1'], ['M2', 'M3', 'A2'], ['M1', 'M3', 'A1', 'A2']]
}
</output>
</example>
</part1>
<part2>
Create a class with the following components:
Initialize the class with three parameters:
    A list of lists containing equation names (call it 'mso')
    A dictionary where keys are equation names and values are equation strings (call it 'equations')
    A dictionary of observed data (call it 'data')
    Store these as instance variables with the same names.
    Create an additional instance variable called 'parsed_equations' by calling a method that defines and parses the equations.
Implement a private method that defines and parses equations:
    Initialize an empty dictionary called 'parsed'
    Iterate through each item in the 'equations' dictionary
    For each equation:
        Split the equation string by '=' into 'left_side' and 'right_side'
        Strip whitespace from both sides
        Create a 'symbols_dict' using dictionary comprehensions:
            For the left side: {var: symbols(var) for var in left_side.split() if var.isidentifier()}
            Update it with right side: {var: symbols(var) for var in right_side.split() if var.isidentifier()}
        Replace observed data in the symbols_dict:
            For each item in 'data', if the key is in symbols_dict, replace the value
        Build the equation based on the operator:
            Check for '&', '|', '^', '~^', '~', '~&', '~|' in the left_side
                For your information what means these symbols: NAND ('~&'), NOR ('~|'), XNOR ('~^'), XOR ('^'), NOT ('~'), AND ('&'), OR ('|')
            For each case, create a parsed_expression using SymPy functions (Not, And, Or, Xor)
            Use list comprehensions to get the relevant symbols from symbols_dict
            Convert to equality using Eq(parsed_expression, symbols_dict[right_side])
        For arithmetic:
            If '+' in left_side: split by '+', use sum() with a generator expression
            If '' in left_side: split by '', use a loop to multiply terms
            Otherwise, assume it's a single term
            Create an Eq object with left_expr and right_expr
        Add the parsed expression to the 'parsed' dictionary
    Return the 'parsed' dictionary
Implement a private method to check if a set of equations is contradictory:
    Check if equations are logical or arithmetic:
        Use: any(char in equation for equation in self.equations.values() for char in ['~', '&', '|', '^'])
    For logical equations:
        Get all variables: {var for eq in equations.values() for var in eq.free_symbols}
        Generate all True/False combinations: product([True, False], repeat=len(variables))
        For each combination:
            Create an assignment dictionary using dict(zip(variables, combination))
            Check if all equations are satisfied using all() and a generator expression
            If any combination satisfies all equations, return False
        If no satisfying combination is found, return True
    For arithmetic equations:
        Convert equations.values() to a list
        Use SymPy's solve function: solve(eqs, dict=True)
        Return True if the solution list is empty, False otherwise
Implement a public method to get minimal conflicts:
    Initialize an empty list called 'minimal_conflicts'
    Iterate through each group in self.mso
    For each group:
        Create a dictionary 'group_equations' using a dictionary comprehension:
        {name: self.parsed_equations[name] for name in group if name in self.parsed_equations}
        Check if the group is contradictory using the method from step 3
        If contradictory, append the sorted group to 'minimal_conflicts'
Return 'minimal_conflicts' sorted using: sorted(minimal_conflicts, key=lambda x: (len(x), x))

HINTS:
1. To check whether the equation is logical, check the presence of at least one character from the list ['|', '&', '^', '~'] in at least one provided equations.values().
2. Be sure that all logical opearation (AND (&), OR (|), XOR (^), NOT (~), NAND (~&), NOR (~|), XNOR (~^)) and arithmetic (+, *) are supported.
3. Use example_1, example_2 and example_3 to test your code before generate results for given data.

<example_1>
<input>
equations = {
    'M1': 'a * c = x01', 
    'M2': 'b * d = x02', 
    'M3': 'c * e = x03', 
    'A1': 'x01 + x02 = f', 
    'A2': 'x02 + x03 = g'}
mso = [['M1', 'M2', 'A1'], ['M2', 'M3', 'A2'], ['M1', 'M3', 'A1', 'A2']]
data = {'a': 2, 'b': 2, 'c': 3, 'd': 3, 'e': 2, 'f': 10, 'g': 12}
</input>
For ['M1', 'M2', 'A1'] system of equations is:
[a * c = x01,
b * d = x02,
x01 + x02 = f
]
For ['M2', 'M3', 'A2'] system of equations is:
[b * d = x02,
c * e = x03,
x02 + x03 = g
]
For ['M1', 'M3', 'A1', 'A2'] system of equations is:
[a * c = x01,
c * e = x03,
x01 + x02 = f,
x02 + x03 = g
]
For ['M1', 'M2', 'A1'] and ['M1', 'M3', 'A1', 'A2'] systems of eqautions are inconsistent so output should be:
<output>
{
'minimal_conflicts': [['M1', 'M2', 'A1'], ['M1', 'M3', 'A1', 'A2']]
}
</output>
</example_1>
<example_2>
<input>
mso = [['A1', 'O1', 'O2'], ['A2', 'O2', 'O3'], ['A1', 'A2', 'O1', 'O3']]
equations = {
    'O1': 'a | c = x01',
    'O2': 'b | d = x02',
    'O3': 'c | e = x03',
    'A1': 'x01 & x02 = f',
    'A2': 'x02 & x03 = g'}
data = {'a': True, 'b': True, 'c': True, 'd': True, 'e': True, 'f': False, 'g': False}
</input>
For ['A1', 'O1', 'O2'] system of equations is:
[x01 & x02 = f,
a | c = x01,
b | d = x02
]
For ['A2', 'O2', 'O3'] system of equations is:
[x02 & x03 = g,
b | d = x02,
c | e = x03
]
For ['A1', 'A2', 'O1', 'O3'] system of equations is:
[x01 & x02 = f,
x02 & x03 = g,
a | c = x01,
c | e = x03
]
For ['A1', 'O1', 'O2'] and ['A2', 'O2', 'O3'] systems of eqautions are inconsistent so output should be:
<output>
{
'minimal_conflicts': [['A1', 'O1', 'O2'], ['A2', 'O2', 'O3']]
}
</example_2>
<example_3>
<input>
mso = [['NA1', 'NO2', 'NX1']]
equations = {
    'NA1': 'a ~& b = x01', 
    'NO2': 'c ~| d = x02', 
    'NX1': 'x01 ~^ x02 = e'}
data = {'a': True, 'b': True, 'c': True, 'd': True, 'e': False}
</input>
For ['NA1', 'NO2', 'NX1'] system of equations is:
[a ~& b = x01,
c ~| d = x02,
x01 ~^ x02 = e
]
For ['NA1', 'NO2', 'NX1'] system of eqautions are inconsistent so output should be:
<output>
{
'minimal_conflicts': [['NA1', 'NO2', 'NX1']]
}
</example_3>
</part2>
<part3>
Use 'minimal_conflicts' from the second part. Use the algorithm below to generate minimal diagnoses.
Translate the algorithm steps into Python functions:
- Convert the high-level pseudocode into Python functions, maintaining the same logic and operations.
- Implement the main loop of the algorithm, handling conflicts and candidates (diagnoses) generation.
- Implement the logic to update the candidates (diagnoses) collection and ensure duplicates and non-minimal elements are removed.
In <output></output> you have example format of output for this part.
[USE CODE INTERPRETER]
<algorithm>
Algorithm 1: Conflicts guide candidates generation.
Inputs: MinimalConflicts
CandidatesCollection←{∅}
for each Conflict ∈ MinimalConflicts do
CurrentCandidates←CandidatesCollection
for each Candidate ∈ CurrentCandidates do
if Candidate ∩ Conflict = ∅ then
CandidatesCollection←UpdateCandidates(Candidate, CandidatesCollection, Conflict)
Return CandidatesCollection

Algorithm 2: UpdateCandidates.
Inputs: Candidate, CandidatesCollection, Conflict
CandidatesCollection←CandidatesCollection - Candidate
for each Component ∈ Conflict do
NewCandidate←Candidate ∪ { Component }
CandidatesCollection←CandidatesCollection ∪ NewCandidate
Remove duplicates and non-minimal elements from CandidatesCollection
Return CandidatesCollection

</algorithm>
<output>
{
"minimal_diagnoses": [['A1'], ['M1'], ['A2', 'M2'], ['M2', 'M3']]
}
</output>
</part3>
"""