"""Standalone launcher for the prediction page only."""
from pathlib import Path
import runpy


PAGE_PATH = Path(__file__).resolve().parent / "pages" / "6_Predictions_Explorer.py"
runpy.run_path(str(PAGE_PATH), run_name="__main__")
