import sys
import os
# Add parent directory to system path to enable modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.preprocessing import get_processed_dataset

print("Generating descriptive statistics summary...")

# Dynamically calculate paths relative to this script location
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'raw', 'dataset_lechugas.csv'))

df = get_processed_dataset(csv_path)

numeric_cols = ['EC', 'N', 'P', 'K', 'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm']
summary_list = []

for col in numeric_cols:
    for group in df['group'].unique():
        stats = df[df['group'] == group][col].describe()
        summary_list.append({
            'Variable': col, 'Group': group,
            'Mean': round(stats['mean'], 2), 'Std_Dev': round(stats['std'], 2),
            'Q1_25%': round(stats['25%'], 2), 'Median_50%': round(stats['50%'], 2),
            'Q3_75%': round(stats['75%'], 2)
        })

summary_df = pd.DataFrame(summary_list).sort_values(by=['Variable', 'Group'])
print("\n--- Statistical Summary (Old Leaves - KNN Imputed) ---")
print(summary_df.to_string(index=False))

# Dynamic output routing to absolute project root structure
output_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'processed', 'descriptive_summary.csv'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
summary_df.to_csv(output_path, index=False)
print(f"\nExecution complete. Output saved to: {output_path}")