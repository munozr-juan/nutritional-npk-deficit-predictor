import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from ml_pipeline.src.preprocessing import get_processed_dataset

st.set_page_config(page_title="IoT Crop Diagnosis", layout="wide", page_icon="🌱")

@st.cache_data(ttl=3600) 
def process_data_and_train_model():
    df = get_processed_dataset('data/raw/dataset_lechugas.csv')
    df = df[df['evolution_hours'] > 72.0].copy()

    sensor_cols = ['EC', 'N', 'P', 'K', 'V_450nm', 'B_500nm', 'G_550nm', 'Y_570nm', 'O_600nm', 'R_650nm']
    exclude_cols = ['timestamp', 'group', 'plant_num', 'unique_plant_id', 'leaf_age', 'leaf_num', 'deficit_type', 'deficit_severity', 'Stress_Index']
    features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features]
    y = df['group']
    groups = df['unique_plant_id'].values 

    gkf = GroupKFold(n_splits=5)
    fold_accs, fold_f1s, fold_metrics = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train)
        X_test_scaled = scaler_fold.transform(X_test)

        model_fold = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        model_fold.fit(X_train_scaled, y_train)

        pred = model_fold.predict(X_test_scaled)
        fold_accs.append(accuracy_score(y_test, pred))
        fold_f1s.append(f1_score(y_test, pred, average='macro'))
        fold_metrics.append({"Fold": fold, "Accuracy": fold_accs[-1], "F1-Macro": fold_f1s[-1]})

    model_summary = {
        "mean_acc": np.mean(fold_accs), "std_acc": np.std(fold_accs),
        "mean_f1": np.mean(fold_f1s), "std_f1": np.std(fold_f1s),
        "total_records": len(X), "total_plants": len(np.unique(groups)),
        "folds_df": pd.DataFrame(fold_metrics)
    }

    latest_idx = df.groupby('unique_plant_id').tail(1).index
    historical_idx = df.index.difference(latest_idx)
    
    X_hist = df.loc[historical_idx, features]
    y_hist = df.loc[historical_idx, 'group']
    
    scaler_final = StandardScaler()
    X_hist_scaled = scaler_final.fit_transform(X_hist)
    
    model_final = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model_final.fit(X_hist_scaled, y_hist)

    smoothed_df = df.copy()
    smoothing_cols = sensor_cols + ['TGI', 'VARI', 'GLI', 'NDYI', 'NP_ratio', 'NK_ratio']
    smoothed_df[smoothing_cols] = smoothed_df.groupby('unique_plant_id')[smoothing_cols].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    latest_df = smoothed_df.loc[latest_idx].copy()
    X_latest_scaled = scaler_final.transform(latest_df[features])
    latest_df['AI_Diagnosis'] = model_final.predict(X_latest_scaled)

    def get_agronomic_recommendation(prediction):
        if prediction in ['Control', 'Sano']: return '✅ Optimal condition. Maintain standard nutrient feed.'
        elif prediction == '-N': return '🧪 ALERT: Apply Nitrogen-rich foliar spray / Check solution EC.'
        elif prediction == '-P': return '🧪 ALERT: Balance pH to unlock Phosphorus / Add phosphate supplement.'
        elif prediction == '-K': return '🧪 ALERT: Increase Potassium concentrations in reservoir solution.'
        return 'Requires immediate manual inspection.'

    latest_df['Suggested_Action'] = latest_df['AI_Diagnosis'].apply(get_agronomic_recommendation)
    return latest_df, model_summary

# UI Structure
title_col, action_col = st.columns([4, 1])
with title_col:
    st.title("🌱 Real-Time Crop Diagnostic Dashboard")
with action_col:
    st.write("") 
    if st.button("🔄 Sync Telemetry Streams"):
        st.cache_data.clear() 
        st.rerun()

st.markdown("Predictive architecture for machine learning classification of micro-nutritional deficiencies.")
tab1, tab2 = st.tabs(["🎯 Live Monitoring & Action Plans", "⚙️ Algorithmic Health Engine"])

with st.spinner("Processing telemetry metrics and loading models..."):
    results_df, metrics = process_data_and_train_model()

with tab1:
    st.subheader("Greenhouse Health Matrix")
    status_counts = results_df['AI_Diagnosis'].value_counts()
    cols = st.columns(len(status_counts))
    for i, (status, count) in enumerate(status_counts.items()):
        if status in ["Control", "Sano"]:
            cols[i].metric(label=f"Status: {status}", value=f"{count}", delta="Healthy Plants")
        else:
            cols[i].metric(label=f"Status: {status}", value=f"{count}", delta="Anomalies Found", delta_color="inverse")
    
    st.markdown("---")
    at_risk_plants = results_df[~results_df['AI_Diagnosis'].isin(['Control', 'Sano'])]
    if not at_risk_plants.empty:
        st.error(f"⚠️ **CRITICAL ACTION REQUIRED:** System identified anomalies in {len(at_risk_plants)} plants.")
        with st.expander("Priority Operational Intervention List", expanded=True):
            alert_df = at_risk_plants[['unique_plant_id', 'AI_Diagnosis', 'Suggested_Action', 'EC', 'VARI']].copy()
            alert_df.columns = ['Plant ID / Sector', 'Deficit Flag', 'Agronomic Action Plan', 'Electrical Conductivity (EC)', 'Health Score (VARI)']
            st.dataframe(alert_df.style.format({'Electrical Conductivity (EC)': '{:.2f}', 'Health Score (VARI)': '{:.3f}'}), use_container_width=True, hide_index=True)
    else:
        st.success("✅ Spectral signatures fall entirely within optimal baselines.")

    st.markdown("---")
    st.subheader("📋 Complete Plant Inventory Ledger")
    display_df = results_df[['unique_plant_id', 'AI_Diagnosis', 'Suggested_Action', 'EC', 'N', 'P', 'K', 'VARI']].copy()
    display_df.columns = ['Plant ID', 'AI Diagnosis', 'Suggested Action Plan', 'EC Average', 'N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)', 'VARI Score']
    
    st.dataframe(display_df.style.apply(lambda row: ['background-color: #ffe6e6; color: #a30000; font-weight: bold'] * len(row) if row['AI Diagnosis'] not in ['Control', 'Sano'] else [''] * len(row), axis=1).format({
        'EC Average': '{:.2f}', 'N (mg/kg)': '{:.1f}', 'P (mg/kg)': '{:.1f}', 'K (mg/kg)': '{:.1f}', 'VARI Score': '{:.4f}'
    }), use_container_width=True, hide_index=True)

with tab2:
    st.info("Technical deployment metrics for validating pure statistical model capabilities using raw telemetry data.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Processing Logs", metrics['total_records'])
    col_b.metric("Tracked Unique Plants", metrics['total_plants'])
    col_c.metric("Mean Pipeline Accuracy", f"{metrics['mean_acc']*100:.1f}%")
    st.markdown("### GroupKFold Cross-Validation Metrics (5-Folds)")
    st.markdown(f"**Cross-Validation Accuracy Mean:** {metrics['mean_acc']:.4f} (± {metrics['std_acc']:.4f})")
    st.markdown(f"**Macro F1-Score Mean:** {metrics['mean_f1']:.4f} (± {metrics['std_f1']:.4f})")