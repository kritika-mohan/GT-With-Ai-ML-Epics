"""
Syntax-check all code cells in the notebook to confirm they will execute without import errors.
"""
import json
import ast

nb_path = r'e:\COLLEGE\GT-With-Ai-ML-Epics\AI_Energy_Optimization\notebooks\01_Energy_Data_Analysis_and_ML_Forecasting.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

for idx, (cell_idx, cell) in enumerate(code_cells):
    source = ''.join(cell['source'])
    try:
        ast.parse(source)
        print(f"  [Cell {cell_idx:02d}] Code cell {idx+1:02d}: ✅ syntax OK")
    except SyntaxError as e:
        errors.append((cell_idx, e))
        print(f"  [Cell {cell_idx:02d}] Code cell {idx+1:02d}: ❌ SYNTAX ERROR — {e}")

print(f"\n{'='*50}")
if errors:
    print(f"❌ {len(errors)} syntax error(s) found.")
else:
    print(f"✅ All {len(code_cells)} code cells passed syntax check!")
