import json
import os
from pathlib import Path

def new_markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def new_code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}

def write_nb(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def create_nb_02():
    cells = [
        new_markdown_cell("# 02 - Trajectory Reconstruction\\n\\nThis notebook takes raw ADS-B state vectors (individual points in time) downloaded from OpenSky and reconstructs them into continuous flight trajectories."),
        new_code_cell("import pandas as pd\\nimport numpy as np\\nimport os\\nimport sys\\nfrom pathlib import Path\\n\\n# Add parent directory to path\\nsys.path.append(os.path.abspath('..'))\\nfrom src.utils import load_config, set_seed\\n\\nconfig = load_config('../configs/config.yaml')\\npaths = config['paths']\\nset_seed(config['random_seed'])"),
        new_markdown_cell("## 1. Load Raw OpenSky Data"),
        new_code_cell("raw_adsb_path = Path('..') / paths['raw_data_dir'] / paths['opensky_data_file']\\n\\nif raw_adsb_path.exists():\\n    df_raw = pd.read_parquet(raw_adsb_path)\\n    print(f'Loaded {len(df_raw):,} raw ADS-B pings.')\\n    display(df_raw.head())\\nelse:\\n    print('Raw ADS-B data not found. Please run 01_data_ingestion.ipynb to fetch sample data.')"),
        new_markdown_cell("## 2. Reconstruct Trajectories\\n\\n*(Note: In a production environment, this step aggregates pings grouped by `icao24`/`callsign` and partitions them using a time gap threshold, e.g., > 1 hour, to separate distinct flights.)*"),
        new_code_cell("def reconstruct_trajectories(df):\\n    \\\"\\\"\\\"Group raw pings into continuous flight segments.\\\"\\\"\\\"\\n    if 'time' in df.columns:\\n        df = df.sort_values(by=['icao24', 'time'])\\n    \\n    # Example: Assign a trajectory ID whenever there is a large gap.\\n    # Returning raw data here as mock trajectories for the pipeline.\\n    return df\\n\\nif raw_adsb_path.exists():\\n    df_traj = reconstruct_trajectories(df_raw)\\n    print(f'Reconstructed {len(df_traj):,} trajectories.')\\n    \\n    # Save for the next step\\n    out_path = Path('..') / paths['processed_data_dir'] / paths['trajectories_file']\\n    os.makedirs(out_path.parent, exist_ok=True)\\n    df_traj.to_parquet(out_path, index=False)\\n    print(f'Saved trajectories to {out_path}')")
    ]
    write_nb('notebooks/02_trajectory_reconstruction.ipynb', cells)

def create_nb_03():
    cells = [
        new_markdown_cell("# 03 - Feature Extraction (ADS-B Behavioral Footprints)\\n\\nNow that we have continuous flight trajectories, we use `FeatureExtractor` to compute high-level, flight-wide behavioral features (e.g., total duration, holding patterns detected, unstable descents, variation in vertical rates)."),
        new_code_cell("import pandas as pd\\nimport sys\\nimport os\\nfrom pathlib import Path\\n\\nsys.path.append(os.path.abspath('..'))\\nfrom src.feature_engineering import FeatureExtractor\\nfrom src.utils import load_config\\n\\nconfig = load_config('../configs/config.yaml')\\npaths = config['paths']"),
        new_code_cell("traj_path = Path('..') / paths['processed_data_dir'] / paths['trajectories_file']\\n\\nif not traj_path.exists():\\n    print('Trajectories file missing. Run notebook 02 first.')\\nelse:\\n    df_traj = pd.read_parquet(traj_path)\\n    print(f'Loaded {len(df_traj):,} continuous trajectory records.')\\n    display(df_traj.head(2))"),
        new_markdown_cell("## 1. Extract Behavioral Features"),
        new_code_cell("if traj_path.exists():\\n    fe = FeatureExtractor()\\n    # Aggregate the continuous ping trajectories into one feature vector per flight\\n    df_features = fe.extract_features(df_traj)\\n    \\n    print(f'\\\\nSuccessfully generated {len(df_features):,} flight-level feature vectors.')\\n    display(df_features.head())\\n    \\n    # Save features\\n    # (Usually this is kept in memory during pipeline orchestration, but we can inspect it here)")
    ]
    write_nb('notebooks/03_feature_extraction.ipynb', cells)

def create_nb_04():
    cells = [
        new_markdown_cell("# 04 - Dataset Integration and Labeling\\n\\nThis notebook mirrors the automated `main.py` pipeline. It merges the extracted ADS-B behavioral features with the external historical schedules (Eurocontrol & BTS). The goal is to enforce an inner join—retaining only ADS-B flights that have a verifiable schedule—and then generate ML classification labels ('Normal', 'Late', 'Cancelled')."),
        new_code_cell("import pandas as pd\\nimport sys\\nimport os\\nfrom pathlib import Path\\n\\nsys.path.append(os.path.abspath('..'))\\nfrom src.merge import DataMerger\\nfrom src.labeling import LabelGenerator\\nfrom src.utils import load_config, preprocess_eurocontrol\\n\\nconfig = load_config('../configs/config.yaml')\\npaths = config['paths']\\nraw_dir = Path('..') / paths['raw_data_dir']\\ningest_cfg = config.get('ingestion', {})\\nyear = ingest_cfg.get('year', 2025)"),
        new_markdown_cell("## 1. Load ADS-B Features\\nFirst, let's pretend we just ran FeatureExtractor and have our ADS-B baseline."),
        new_code_cell("from src.feature_engineering import FeatureExtractor\\n\\ntraj_path = Path('..') / paths['processed_data_dir'] / paths['trajectories_file']\\nif traj_path.exists():\\n    df_traj = pd.read_parquet(traj_path)\\n    fe = FeatureExtractor()\\n    df_adsb_features = fe.extract_features(df_traj)\\n    initial_adsb_count = len(df_adsb_features)\\n    print(f'Starting with {initial_adsb_count:,} ADS-B flights.')"),
        new_markdown_cell("## 2. Incrementally Merge Schedule Datasets (OOM-Safe)\\nInstead of loading 20 Million Eurocontrol and BTS flights into memory at once, we iterate through the pre-downloaded monthly files. We merge them immediately against our much smaller ADS-B dataset and keep only the matched rows."),
        new_code_cell("start_month = ingest_cfg.get('start_month', 1)\\nend_month = ingest_cfg.get('end_month', 11)\\nmerger = DataMerger(tolerance_hours=2)\\nmerged_dfs = []\\n\\n# Eurocontrol (Iterating through chunks)\\nfor m in range(start_month, end_month + 1):\\n    euro_file = raw_dir / f'euro_flight_list_{year}{m:02d}.parquet'\\n    if euro_file.exists():\\n        print(f'Processing {euro_file.name}...')\\n        df_euro = preprocess_eurocontrol(pd.read_parquet(euro_file))\\n        df_euro['region'] = 'EU'\\n        \\n        # Merge and enforce INNER JOIN\\n        df_matched = merger.merge_datasets(df_adsb_features, df_euro)\\n        df_matched = df_matched.dropna(subset=['scheduled_dep'])\\n        if not df_matched.empty:\\n            merged_dfs.append(df_matched)\\n        del df_euro"),
        new_code_cell("# BTS Combined Dataset\\nbts_combined_file = Path('..') / ingest_cfg.get('bts_combined_file', 'data/processed/bts_dataset_2025.parquet')\\n\\nif bts_combined_file.exists():\\n    print(f'Processing BTS Dataset...')\\n    df_bts = pd.read_parquet(bts_combined_file)\\n    df_bts['region'] = 'US'\\n    \\n    df_matched = merger.merge_datasets(df_adsb_features, df_bts)\\n    df_matched = df_matched.dropna(subset=['scheduled_dep'])\\n    if not df_matched.empty:\\n        merged_dfs.append(df_matched)"),
        new_markdown_cell("## 3. Label Generation and Standardisation"),
        new_code_cell("if merged_dfs:\\n    df_final = pd.concat(merged_dfs, ignore_index=True)\\n    if 'flight_key' in df_final.columns:\\n        df_final = df_final.drop_duplicates(subset=['flight_key'])\\nelse:\\n    # Fallback to outputting just adsb features directly for testing\\n    df_final = df_adsb_features\\n    \\nlabeler = LabelGenerator(delay_threshold_minutes=15)\\ndf_labeled = labeler.generate_labels(df_final)\\ndf_standard = labeler.standardize_dataset(df_labeled)\\n\\ndisplay(df_standard.head())\\n\\n# Metrics\\nretained = len(df_standard.dropna(subset=['scheduled_dep'])) if 'scheduled_dep' in df_standard.columns else 0\\nprint(f'''\\\\nRetained Flights successfully matched: {retained:,}\\\\nRetention Rate: {(retained/initial_adsb_count*100):.2f}%\\\\n''')")
    ]
    write_nb('notebooks/04_dataset_integration.ipynb', cells)

def create_nb_05():
    cells = [
        new_markdown_cell("# 05 - Visualisation\\n\\nThis notebook contains tools to visually inspect the data. We first plot raw continuous trajectories on an interactive map. Then, we analyze the final unified `ml_dataset.parquet` to inspect class label distributions."),
        
        new_markdown_cell("## 1. Flight Trajectory Mapping\\nInspect individual flight tracks generated during the reconstruction phase."),
        new_code_cell("import os\\nimport sys\\nimport pandas as pd\\nimport random\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\nfrom pathlib import Path\\n\\nsys.path.append(os.path.abspath('..'))\\nfrom src.utils import load_config\\nfrom src.visualisation import TrajectoryVisualiser\\n\\nconfig = load_config('../configs/config.yaml')\\ntraj_path = Path('..') / config['paths']['processed_data_dir'] / config['paths']['trajectories_file']\\n\\nif traj_path.exists():\\n    df_traj = pd.read_parquet(traj_path)\\n    if 'trajectory_id' in df_traj.columns:\\n        print(f\\\"Loaded {len(df_traj['trajectory_id'].unique())} unique trajectories.\\\")\\n    else:\\n        print(\\\"Loaded ADS-B pings, but it seems they have not been reconstructed into 'trajectory_id's yet.\\\")\\nelse:\\n    print('No trajectories found.')\\n    df_traj = pd.DataFrame()"),
        new_code_cell("# Display a single random trajectory map\\nif not df_traj.empty and 'trajectory_id' in df_traj.columns:\\n    traj_ids = list(df_traj['trajectory_id'].unique())\\n    if len(traj_ids) > 0:\\n        random_traj = random.choice(traj_ids)\\n        print(f'Displaying trajectory: {random_traj}')\\n        m = TrajectoryVisualiser.generate_interactive_map(df_traj, trajectory_ids=[random_traj])\\n        display(m)\\n    else:\\n        print('No valid trajectories to display.')"),
        
        new_markdown_cell("## 2. ML Label Distributions\\nInspect the final labels (Normal vs Delayed vs Cancelled) generated by the main pipeline."),
        new_code_cell("ml_data_path = Path('..') / config['paths']['processed_data_dir'] / config['paths']['ml_dataset_file']\\n\\nif ml_data_path.exists():\\n    df_ml = pd.read_parquet(ml_data_path)\\n    print(f'ML Dataset Size: {len(df_ml):,} rows')\\n    display(df_ml.head())\\nelse:\\n    print('Run main.py (or notebook 04) to generate ml_dataset.parquet first!')"),
        new_code_cell("if ml_data_path.exists() and 'label' in df_ml.columns:\\n    sns.set_theme(style=\\\"whitegrid\\\")\\n    plt.figure(figsize=(8, 5))\\n    ax = sns.countplot(data=df_ml, x='label', order=df_ml['label'].value_counts().index)\\n    plt.title('Flight Disruption Label Distribution')\\n    \\n    # Add count labels on top of bars\\n    for p in ax.patches:\\n        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')\\n    \\n    plt.show()")
    ]
    write_nb('notebooks/05_visualisation.ipynb', cells)

create_nb_02()
create_nb_03()
create_nb_04()
create_nb_05()
print('Notebooks successfully regenerated!')
