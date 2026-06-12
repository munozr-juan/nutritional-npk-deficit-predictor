import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def evaluate_scenario(df_base, scenario_name):
    df = df_base.copy()
    
    # 1. KNN Imputation executed within the specific scenario scope to prevent data leakage
    sensor_cols = ['EC', 'N', 'P', 'K', 'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm']
    df[sensor_cols] = df[sensor_cols].replace(-1, np.nan)
    imputer = KNNImputer(n_neighbors=5)
    df[sensor_cols] = imputer.fit_transform(df[sensor_cols])
    
    # 2. Feature Engineering applied on the isolated imputed data
    df['TGI'] = -0.5 * (150 * (df['R_650nm'] - df['G_550nm']) - 100 * (df['R_650nm'] - df['B_500nm']))
    df['VARI'] = (df['G_550nm'] - df['R_650nm']) / (df['G_550nm'] + df['R_650nm'] - df['B_500nm'] + 1e-8)
    df['GLI'] = (2 * df['G_550nm'] - df['R_650nm'] - df['B_500nm']) / (2 * df['G_550nm'] + df['R_650nm'] + df['B_500nm'] + 1e-8)
    df['NDYI'] = (df['G_550nm'] - df['B_500nm']) / (df['G_550nm'] + df['B_500nm'] + 1e-8)
    df['Stress_Index'] = df['R_650nm'] / (df['B_500nm'] + 1e-8)
    df['NP_ratio'] = np.log1p(df['N'] / (df['P'] + 1e-8))
    df['NK_ratio'] = np.log1p(df['N'] / (df['K'] + 1e-8))
    
    # Generate unique plant tracking identifiers
    df['unique_plant_id'] = df['group'].astype(str) + '_' + df['plant_num'].astype(str)
    
    # Define features alignment (evolution_hours remains active for training)
    exclude_cols = ['timestamp', 'group', 'plant_num', 'unique_plant_id', 'leaf_age', 'leaf_num', 'deficit_type', 'deficit_severity']
    features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features]
    y = df['group']
    groups = df['unique_plant_id'].values

    models = {
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Bagging': BaggingClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = []
    gkf = GroupKFold(n_splits=5)

    for name, model in models.items():
        fold_acc, fold_f1 = [], []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_f1.append(f1_score(y_test, y_pred, average='macro'))
            
        results.append({
            'Model': name, 'Accuracy': np.mean(fold_acc), 'Std_Acc': np.std(fold_acc),
            'F1-Score': np.mean(fold_f1), 'Std_F1': np.std(fold_f1)
        })
        
    res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print(f"\n{'-'*70}\n{scenario_name} ({len(X)} records)\n{'-'*70}")
    print(res_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'raw', 'dataset_lechugas.csv'))
    
    # Load raw dataset and apply standardized English structure
    df_master = pd.read_csv(csv_path, header=None)
    df_master.columns = [
        'timestamp', 'group', 'plant_num', 'leaf_age', 'leaf_num',
        'deficit_type', 'deficit_severity', 'EC', 'N', 'P', 'K',
        'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm'
    ]
    
    # Clean and filter target biological structures (Old Leaves Only)
    df_master = df_master[df_master['leaf_age'] == 'Vieja'].copy()
    df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])
    df_master = df_master.sort_values(['group', 'plant_num', 'timestamp'])
    
    # Track operational time timeline delta hours
    df_master['evolution_hours'] = (df_master['timestamp'] - df_master['timestamp'].min()).dt.total_seconds() / 3600.0

    # Table 6 Framework Execution
    evaluate_scenario(df_master, "Table 6: Performance Evaluation on Full Dataset")

    # Table 7 Framework Execution (> 72h latency filter matching identical isolated pipeline)
    df_post = df_master[df_master['evolution_hours'] > 72.0].copy()
    evaluate_scenario(df_post, "Table 7: Performance Evaluation on Post-Acclimatization Data")