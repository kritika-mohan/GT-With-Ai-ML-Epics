import json

nb_path = r'e:\COLLEGE\GT-With-Ai-ML-Epics\AI_Energy_Optimization\notebooks\01_Energy_Data_Analysis_and_ML_Forecasting.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
n_md = sum(1 for c in cells if c['cell_type'] == 'markdown')
n_code = sum(1 for c in cells if c['cell_type'] == 'code')

print(f'Valid JSON: True')
print(f'NBFormat: {nb["nbformat"]}')
print(f'Total cells: {len(cells)}')
print(f'Markdown cells: {n_md}')
print(f'Code cells: {n_code}')
print('\nCell summary:')
for i, c in enumerate(cells):
    ctype = c['cell_type']
    source = ''.join(c['source'])[:80].replace('\n', ' ')
    print(f'  [{i:02d}] {ctype:8} | {source}')
