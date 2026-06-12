import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import binomtest
import warnings
warnings.filterwarnings('ignore')

print("Starting early validation statistical modeling...")
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'raw', 'dataset_lechugas.csv'))

# 1. Load raw dataset and apply standardized English structure
df = pd.read_csv(csv_path, header=None)
df.columns = [
    'timestamp', 'group', 'plant_num', 'leaf_age', 'leaf_num',
    'deficit_type', 'deficit_severity', 'EC', 'N', 'P', 'K',
    'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm'
]

# 2. Filter biological structures and sort
df = df[df['leaf_age'] == 'Vieja'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['group', 'plant_num', 'timestamp'])

# Unique plant tracking identifier
df['unique_plant_id'] = df['group'].astype(str) + '_' + df['plant_num'].astype(str)

# 3. Calculate early detection windows (Pre-symptom tracking)
symptom_df = df[df['deficit_severity'] > 0]
first_symptom = symptom_df.groupby('unique_plant_id')['timestamp'].min().reset_index()
first_symptom.rename(columns={'timestamp': 'first_symptom_date'}, inplace=True)

df = df.merge(first_symptom, on='unique_plant_id', how='left')
df['days_until_symptom'] = (df['first_symptom_date'] - df['timestamp']).dt.total_seconds() / (24 * 3600.0)
df['days_before_label'] = np.ceil(df['days_until_symptom'])

# 4. Timeline operational tracking and CRITICAL LATENCY FILTER (> 72h)
df['evolution_hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
df_train = df[df['evolution_hours'] > 72.0].copy().reset_index(drop=True)

# 5. KNN Imputation executed within the post-acclimatization scope
sensor_cols = ['EC', 'N', 'P', 'K', 'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm']
df_train[sensor_cols] = df_train[sensor_cols].replace(-1, np.nan)
imputer = KNNImputer(n_neighbors=5)
df_train[sensor_cols] = imputer.fit_transform(df_train[sensor_cols])

# 6. Feature Engineering alignment
df_train['TGI'] = -0.5 * (150 * (df_train['R_650nm'] - df_train['G_550nm']) - 100 * (df_train['R_650nm'] - df_train['B_500nm']))
df_train['VARI'] = (df_train['G_550nm'] - df_train['R_650nm']) / (df_train['G_550nm'] + df_train['R_650nm'] - df_train['B_500nm'] + 1e-8)
df_train['GLI'] = (2 * df_train['G_550nm'] - df_train['R_650nm'] - df_train['B_500nm']) / (2 * df_train['G_550nm'] + df_train['R_650nm'] + df_train['B_500nm'] + 1e-8)
df_train['NDYI'] = (df_train['G_550nm'] - df_train['B_500nm']) / (df_train['G_550nm'] + df_train['B_500nm'] + 1e-8) 
df_train['NP_ratio'] = np.log1p(df_train['N'] / (df_train['P'] + 1e-8))
df_train['NK_ratio'] = np.log1p(df_train['N'] / (df_train['K'] + 1e-8))

exclude_cols = ['timestamp', 'group', 'plant_num', 'unique_plant_id', 'leaf_age', 'leaf_num', 
                'deficit_type', 'deficit_severity', 'first_symptom_date', 'days_until_symptom', 'days_before_label']
features = [col for col in df_train.columns if col not in exclude_cols]

X = df_train[features]
y = df_train['group']
groups = df_train['unique_plant_id'].values 

# 7. Cross-Validation and Mapping
gkf = GroupKFold(n_splits=5)
all_preds = np.zeros(len(df_train), dtype=object)

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)
    all_preds[test_idx] = model.predict(X_test_scaled)

df_train['AI_Prediction'] = all_preds

print("\n" + "="*80 + "\nEARLY ACCURACY RELIABILITY AND SIGNIFICANCE ANALYSIS (PRE-SYMPTOM WINDOW)\n" + "="*80)

def wilson_confidence_interval(successes, total, z=1.96):
    if total == 0: return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    dev = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (center - dev) / denom, (center + dev) / denom

df_early = df_train[df_train['days_before_label'].isin([1.0, 2.0, 3.0, 4.0, 5.0])]
stat_results = []
chance_prob = 0.25 

for day in [5.0, 4.0, 3.0, 2.0, 1.0]:
    day_data = df_early[df_early['days_before_label'] == day]
    total_n = len(day_data)
    if total_n > 0:
        successes = sum(day_data['group'] == day_data['AI_Prediction'])
        acc = successes / total_n
        lower_bound, upper_bound = wilson_confidence_interval(successes, total_n)
        p_val = binomtest(successes, total_n, chance_prob, alternative='greater').pvalue
        significance = "Highly Significant" if p_val < 0.01 else ("Significant" if p_val < 0.05 else "Not Significant")
        
        stat_results.append({
            'Days Prior': f"-{int(day)}", 'Samples': total_n, 'Successes': f"{successes}/{total_n}",
            'Accuracy': f"{acc*100:.1f}%", '95% CI Range': f"[{lower_bound*100:.1f}% - {upper_bound*100:.1f}%]",
            'p-value': f"{p_val:.4f}", 'Reliability': significance
        })

stat_df = pd.DataFrame(stat_results)
print(stat_df.to_string(index=False))