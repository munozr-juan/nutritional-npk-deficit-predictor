import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("Starting training pipeline...")
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'raw', 'dataset_lechugas.csv'))

# 1. Load raw dataset and assign English baseline columns
df = pd.read_csv(csv_path, header=None)
df.columns = [
    'timestamp', 'group', 'plant_num', 'leaf_age', 'leaf_num',
    'deficit_type', 'deficit_severity', 'EC', 'N', 'P', 'K',
    'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm'
]

# 2. Replicate the precise sorting and timeline filtering from baseline script
df = df[df['leaf_age'] == 'Vieja'].copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')  # Global chronological sort matching 'modelo_lechugas.py'

df['evolution_hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600.0
df = df[df['evolution_hours'] > 72.0].copy()  # Latency filter executed BEFORE imputation

# 3. KNN Imputation bounded strictly within the 445 isolated records
sondas_y_sensores = ['EC', 'N', 'P', 'K', 'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm']
df[sondas_y_sensores] = df[sondas_y_sensores].replace(-1, np.nan)
imputer = KNNImputer(n_neighbors=5)
df[sondas_y_sensores] = imputer.fit_transform(df[sondas_y_sensores])

# 4. Feature Engineering mapping exact equations and tracking IDs
df['TGI'] = -0.5 * (150 * (df['R_650nm'] - df['G_550nm']) - 100 * (df['R_650nm'] - df['B_500nm']))
df['VARI'] = (df['G_550nm'] - df['R_650nm']) / (df['G_550nm'] + df['R_650nm'] - df['B_500nm'] + 1e-8)
df['GLI'] = (2 * df['G_550nm'] - df['R_650nm'] - df['B_500nm']) / (2 * df['G_550nm'] + df['R_650nm'] + df['B_500nm'] + 1e-8)
df['NDYI'] = (df['G_550nm'] - df['B_500nm']) / (df['G_550nm'] + df['B_500nm'] + 1e-8) 

df['NP_ratio'] = np.log1p(df['N'] / (df['P'] + 1e-8))
df['NK_ratio'] = np.log1p(df['N'] / (df['K'] + 1e-8))

df['unique_plant_id'] = df['group'].astype(str) + '_' + df['plant_num'].astype(str)

# 5. Strictly align feature columns matching the exact index generation sequence
exclude_cols = ['timestamp', 'group', 'plant_num', 'unique_plant_id', 'leaf_age', 'leaf_num', 'deficit_type', 'deficit_severity']
features = [col for col in df.columns if col not in exclude_cols]

X = df[features]
y = df['group']
groups = df['unique_plant_id'].values 

print(f"Total records to evaluate (post-latency filter): {len(X)}")

# K Fold Group Validation Framework (N=5)
gkf = GroupKFold(n_splits=5)
fold_accs, fold_f1s = [], []
y_true, y_pred = [], []
importances = []

print("\n" + "="*50)
print("STARTING CROSS-VALIDATION (5-FOLDS) WITH GB")
print("="*50)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    fold_accs.append(accuracy_score(y_test, pred))
    fold_f1s.append(f1_score(y_test, pred, average='macro'))
    
    y_true.extend(y_test)
    y_pred.extend(pred)
    importances.append(model.feature_importances_)
    print(f"Fold {fold} | Accuracy: {fold_accs[-1]:.4f} | F1-Macro: {fold_f1s[-1]:.4f}")

print("\n=== Consolidated Global Evaluation Summary ===")
print(f"Mean Accuracy: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
print(f"Mean F1-Macro: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Document Visualizations Exporting Block
docs_dir = os.path.join('..', 'docs')
os.makedirs(docs_dir, exist_ok=True)
sns.set_theme(style="whitegrid")

# Confusion Matrix
plt.figure(figsize=(8, 6))
labels = np.unique(y_true)
cm = confusion_matrix(y_true, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Global Confusion Matrix (GroupKFold)', fontsize=14, pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(docs_dir, 'thesis_confusion_matrix.png'), dpi=300)
plt.close()

# Feature Importance
plt.figure(figsize=(9, 6)) 
mean_importances = np.mean(importances, axis=0)
imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': mean_importances})
imp_df = imp_df[imp_df['Feature'] != 'evolution_hours'].sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
plt.title('Agronomic Feature Importance Evaluation', fontsize=14, pad=15)
plt.xlabel('Mean Relative Importance', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(docs_dir, 'thesis_feature_importance.png'), dpi=300)
plt.close()
print(f"Charts successfully saved to: {docs_dir}")