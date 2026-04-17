import json
from pathlib import Path

nbs = [
    'notebooks/02_trajectory_reconstruction.ipynb',
    'notebooks/03_feature_extraction.ipynb',
    'notebooks/04_dataset_integration.ipynb',
    'notebooks/05_visualisation.ipynb'
]

for nb_file in nbs:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for cell in data.get('cells', []):
            if 'source' in cell and len(cell['source']) == 1:
                # Get the single source string and replace literal backslash-n with actual newline
                text = cell['source'][0].replace('\\\\n', '\\n')
                # For proper Jupyter formatting, source should be a list of strings, each ending with a newline
                lines = [line + '\\n' for line in text.split('\\n')]
                # Remove the trailing newline from the very last line
                if lines:
                    lines[-1] = lines[-1].rstrip('\\n')
                cell['source'] = lines
                
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Fixed formatting in {nb_file}")
    except Exception as e:
        print(f"Failed to fix {nb_file}: {e}")
