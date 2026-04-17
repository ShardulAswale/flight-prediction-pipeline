import json
from pathlib import Path

file = 'notebooks/02_trajectory_reconstruction.ipynb'
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Locate the cell with reconstruct_trajectories
for cell in data['cells']:
    if cell.get('cell_type') == 'code':
        source = "".join(cell['source'])
        if "def reconstruct_trajectories" in source:
            new_func = """def reconstruct_trajectories(df, gap_minutes=15):
    \"\"\"Group raw pings into continuous flight segments.\"\"\"
    if 'time' in df.columns:
        df = df.sort_values(by=['icao24', 'time'])
        
        if pd.api.types.is_numeric_dtype(df['time']):
            time_diff = df.groupby('icao24')['time'].diff().fillna(0)
            gap_threshold = gap_minutes * 60
        else:
            time_diff = df.groupby('icao24')['time'].diff().dt.total_seconds().fillna(0)
            gap_threshold = gap_minutes * 60
            
        new_traj = (time_diff > gap_threshold).astype(int)
        traj_num = new_traj.groupby(df['icao24']).cumsum()
        df['trajectory_id'] = df['icao24'].astype(str) + '_' + traj_num.astype(str)
        df = df.drop(columns=['time_diff', 'new_traj'])
    else:
        if 'trajectory_id' not in df.columns:
            df['trajectory_id'] = df['icao24'].astype(str) + '_0'
    return df

if raw_adsb_path.exists():
    df_traj = reconstruct_trajectories(df_raw)
    print(f'Reconstructed {len(df_traj):,} ping rows into {df_traj["trajectory_id"].nunique()} trajectories.')
    
    # Save for the next step
    out_path = Path('..') / paths['processed_data_dir'] / paths['trajectories_file']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_traj.to_parquet(out_path, index=False)
    print(f'Saved trajectories to {out_path}')"""
            lines = [line + '\n' for line in new_func.split('\n')]
            lines[-1] = lines[-1].rstrip('\n')
            cell['source'] = lines

# Add visualization cells
import_vis_code = """import matplotlib.pyplot as plt
import seaborn as sns
from src.visualisation import TrajectoryVisualiser"""

vis_markdown = """## 3. Visualise Reconstructed Trajectory
Let's pick a random trajectory and plot its geographical map and altitude profile."""

vis_code = """import random

if raw_adsb_path.exists() and 'trajectory_id' in df_traj.columns:
    traj_ids = list(df_traj['trajectory_id'].unique())
    if traj_ids:
        # Pick a trajectory that has enough points to be interesting
        counts = df_traj['trajectory_id'].value_counts()
        valid_trajs = counts[counts > 50].index.tolist()
        random_traj = random.choice(valid_trajs) if valid_trajs else random.choice(traj_ids)
        
        print(f'Displaying trajectory: {random_traj}')
        
        # 1. Interactive Map
        m = TrajectoryVisualiser.generate_interactive_map(df_traj, trajectory_ids=[random_traj])
        display(m)
        
        # 2. Altitude Graph
        plt.figure(figsize=(10, 4))
        df_single = df_traj[df_traj['trajectory_id'] == random_traj].sort_values('time')
        
        # Use 'geoaltitude' if available, otherwise 'baroaltitude'
        alt_col = 'geoaltitude' if 'geoaltitude' in df_single.columns and not df_single['geoaltitude'].isna().all() else 'baroaltitude'
        
        sns.lineplot(data=df_single, x='time', y=alt_col, marker='o', markersize=4)
        plt.title(f'Altitude Profile for {random_traj}')
        plt.xlabel('Time (Unix)')
        plt.ylabel(f'Altitude ({alt_col})')
        plt.grid(True)
        plt.show()
    else:
        print('No valid trajectories to display.')"""

def new_markdown(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines: lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def new_code(text):
    lines = [line + '\n' for line in text.split('\n')]
    if lines: lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

has_vis = any("Visualise Reconstructed Trajectory" in str(c.get('source', [])) for c in data['cells'])
if not has_vis:
    data['cells'].insert(2, new_code(import_vis_code))
    data['cells'].append(new_markdown(vis_markdown))
    data['cells'].append(new_code(vis_code))

with open(file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
print(f"Updated {file}")
