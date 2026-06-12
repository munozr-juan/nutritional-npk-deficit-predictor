# ml_pipeline/src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def get_processed_dataset(file_path):
    df = pd.read_csv(file_path, header=None)
    
    # Standardize column headers to strict English naming schema
    df.columns = [
        'timestamp', 'group', 'plant_num', 'leaf_age', 'leaf_num',
        'deficit_type', 'deficit_severity', 'EC', 'N', 'P', 'K',
        'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm'
    ]
    
    # Target old leaves context and sort chronologically per crop entity
    df = df[df['leaf_age'] == 'Vieja'].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['group', 'plant_num', 'timestamp'])
    df['unique_plant_id'] = df['group'].astype(str) + '_' + df['plant_num'].astype(str)
    
    # Compute operational time delta in hours from timeline baseline
    df['evolution_hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
    
    # Replace invalid probe logs (-1) and execute KNN Imputation
    sensor_cols = ['EC', 'N', 'P', 'K', 'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm']
    df[sensor_cols] = df[sensor_cols].replace(-1, np.nan)
    imputer = KNNImputer(n_neighbors=5)
    df[sensor_cols] = imputer.fit_transform(df[sensor_cols])
    
    # Mathematical Feature Engineering: Spectral Indices and NPK Ratios
    df['TGI'] = -0.5 * (150 * (df['R_650nm'] - df['G_550nm']) - 100 * (df['R_650nm'] - df['B_500nm']))
    df['VARI'] = (df['G_550nm'] - df['R_650nm']) / (df['G_550nm'] + df['R_650nm'] - df['B_500nm'] + 1e-8)
    df['GLI'] = (2 * df['G_550nm'] - df['R_650nm'] - df['B_500nm']) / (2 * df['G_550nm'] + df['R_650nm'] + df['B_500nm'] + 1e-8)
    df['NDYI'] = (df['G_550nm'] - df['B_500nm']) / (df['G_550nm'] + df['B_500nm'] + 1e-8)
    df['Stress_Index'] = df['R_650nm'] / (df['B_500nm'] + 1e-8)
    df['NP_ratio'] = np.log1p(df['N'] / (df['P'] + 1e-8))
    df['NK_ratio'] = np.log1p(df['N'] / (df['K'] + 1e-8))
    
    return df