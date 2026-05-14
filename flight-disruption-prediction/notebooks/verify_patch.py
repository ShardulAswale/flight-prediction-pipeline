import json
from pathlib import Path

nb_path = Path(r"c:\Code\flight-disruption-prediction\notebooks\10_threshold_hyperparameter_experiments.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells[-3:]):
    idx = len(cells) - 3 + i
    src = c.get("source", [])
    first_line = src[0][:70].strip() if src else "(no source)"
    print(f"  Cell [{idx}] type={c['cell_type']}, id={c['id']}")
    print(f"    -> {first_line}")
