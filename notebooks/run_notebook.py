"""
Execute the notebook using nbclient (the programmatic way).
This runs all cells in order and saves the executed notebook.
"""
import nbformat
from nbclient import NotebookClient
from pathlib import Path
import sys, traceback

NB_PATH = Path(r"e:\COLLEGE\GT-With-Ai-ML-Epics\AI_Energy_Optimization\notebooks\01_Energy_Data_Analysis_and_ML_Forecasting.ipynb")
OUT_PATH = Path(r"e:\COLLEGE\GT-With-Ai-ML-Epics\AI_Energy_Optimization\notebooks\01_Energy_Data_Analysis_and_ML_Forecasting_EXECUTED.ipynb")

print(f"Loading notebook: {NB_PATH.name}")
print(f"Output path    : {OUT_PATH.name}")

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Cells to execute: {len([c for c in nb.cells if c.cell_type == 'code'])}")
print("\nStarting execution (this will take several minutes)...\n")

client = NotebookClient(
    nb,
    timeout=600,            # 10 minutes max per cell
    kernel_name='python3',
    resources={'metadata': {'path': str(NB_PATH.parent)}},
    allow_errors=False,     # Stop on first error
)

try:
    client.execute()
    print("\n✅ All cells executed successfully!")
    
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"✅ Executed notebook saved: {OUT_PATH}")

except Exception as e:
    print(f"\n❌ Execution stopped with error:")
    print(f"   {type(e).__name__}: {e}")
    traceback.print_exc()
    
    # Still save the partially executed notebook
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"\nPartially executed notebook saved: {OUT_PATH}")
    sys.exit(1)
