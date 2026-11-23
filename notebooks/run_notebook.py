"""
Execute the notebook using nbclient.
Runs all cells sequentially, embedding outputs directly into the notebook.
"""
import nbformat
from nbclient import NotebookClient
from pathlib import Path
import sys, traceback

NB_PATH = Path(r"E:\COLLEGE\GT-With-Ai-ML-Epics\notebooks\01_Energy_Data_Analysis_and_ML_Forecasting.ipynb")

print(f"Loading notebook: {NB_PATH}")

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == 'code']
print(f"Code cells to execute: {len(code_cells)}")
print("Executing notebook...\n")

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name='python3',
    resources={'metadata': {'path': str(NB_PATH.parent)}},
    allow_errors=False,
)

try:
    client.execute()
    print("\n✅ All notebook cells executed successfully without errors!")
    
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"✅ Notebook updated with all outputs: {NB_PATH}")

except Exception as e:
    print(f"\n❌ Execution halted: {type(e).__name__}: {e}")
    traceback.print_exc()
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    sys.exit(1)
