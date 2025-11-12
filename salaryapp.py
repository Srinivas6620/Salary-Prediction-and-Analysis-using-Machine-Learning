import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

# ------------------------------
# STREAMLIT APP
# ------------------------------

st.set_page_config(page_title="Advanced Outlier Detection Dashboard", layout="wide")

st.title("📊 Salary Dataset — Advanced Outlier Detection & Handling")

# Upload dataset
uploaded_file = st.file_uploader(r"C:\Users\teste\Downloads\Salary_Dataset_with_Extra_Features.csv.zip", type=["csv", "zip"])

if uploaded_file is not None:
    # Load dataset
    try:
        df = pd.read_csv(uploaded_file, compression="zip")
    except Exception:
        df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Dataset Loaded — Shape: {df.shape}")
    st.write(df.head())

    # Select numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found in dataset!")
        st.stop()

    # ------------------------------
    # Step 1 — Preprocessing
    # ------------------------------
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_num = imputer.fit_transform(df[num_cols])
    num_df_imputed = pd.DataFrame(X_num, columns=num_cols, index=df.index)

    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_qt = qt.fit_transform(X_num)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_qt)

    # ------------------------------
    # Step 2 — Autoencoder-like anomaly score
    # (simplified: PCA reconstruction error)
    # ------------------------------
    pca_model = PCA(n_components=min(5, X_scaled.shape[1]))
    X_reduced = pca_model.fit_transform(X_scaled)
    X_recon = pca_model.inverse_transform(X_reduced)
    errors = np.mean((X_scaled - X_recon)**2, axis=1)

    med = np.median(errors)
    mad = np.median(np.abs(errors - med)) + 1e-12
    ae_threshold = med + 6 * mad
    ae_anomaly = errors > ae_threshold

    # ------------------------------
    # Step 3 — DBSCAN anomalies
    # ------------------------------
    db = DBSCAN(eps=0.5, min_samples=10)
    db_labels = db.fit_predict(X_scaled)
    db_anomaly = (db_labels == -1)

    # ------------------------------
    # Step 4 — Combine anomalies
    # ------------------------------
    combined_high_conf = ae_anomaly & db_anomaly
    anomaly_mask = combined_high_conf
    anomaly_pct = (anomaly_mask.sum() / len(df)) * 100

    # ------------------------------
    # Step 5 — Predictive imputation
    # ------------------------------
    df_clean = df.copy()
    for col in num_cols:
        out_idx = df.index[anomaly_mask]
        train_idx = df.index[~anomaly_mask]
        if len(out_idx) == 0: continue
        Xtrain = num_df_imputed.loc[train_idx].drop(columns=[col])
        ytrain = num_df_imputed.loc[train_idx, col]
        Xpred  = num_df_imputed.loc[out_idx].drop(columns=[col])

        if Xtrain.shape[0] < 30:
            df_clean.loc[out_idx, col] = np.median(ytrain)
        else:
            gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
            gbr.fit(Xtrain, ytrain)
            preds = gbr.predict(Xpred)
            df_clean.loc[out_idx, col] = preds

    # ------------------------------
    # Step 6 — Real-Time Anomaly Alert
    # ------------------------------
    st.subheader("⚠️ Real-Time Anomaly Alert System")

    # user can set threshold
    threshold = st.slider("Set anomaly alert threshold (%)", min_value=1, max_value=20, value=5)

    if anomaly_pct > threshold:
        st.error(f"🚨 ALERT: {anomaly_pct:.2f}% anomalies detected! (Threshold = {threshold}%)")
    else:
        st.success(f"✅ Normal: Only {anomaly_pct:.2f}% anomalies detected (Threshold = {threshold}%)")

    # ------------------------------
    # STREAMLIT BUTTONS
    # ------------------------------

    st.subheader("🔘 Interactive Analysis Buttons")

    # 1. Show Raw Dataset
    if st.button("📂 Show Raw Dataset"):
        st.write("This shows the original dataset before any outlier handling.")
        st.dataframe(df.head(20))

    # 2. Autoencoder Reconstruction Error
    if st.button("🤖 Show Autoencoder Error Plot"):
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(errors, bins=50, kde=True, ax=ax, color="steelblue")
        ax.axvline(ae_threshold, color="red", linestyle="--", label="Threshold")
        ax.set_title("Autoencoder Reconstruction Error")
        ax.legend()
        st.pyplot(fig)

    # 3. DBSCAN Clustering
    if st.button("📌 Show DBSCAN Clustering (2D PCA)"):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=db_labels, cmap="Set1", s=20, alpha=0.6)
        ax.set_title("DBSCAN Clustering — PCA Projection")
        st.pyplot(fig)

    # 4. Before vs After Cleaning
    if st.button("📊 Show Before vs After Boxplot (Salary)"):
        if "Salary" in num_cols:
            fig, ax = plt.subplots(figsize=(12,6))
            sns.boxplot(data=[df["Salary"], df_clean["Salary"]], ax=ax)
            ax.set_xticklabels(["Before", "After"])
            ax.set_title("Salary Distribution Before vs After Cleaning")
            st.pyplot(fig)
        else:
            st.warning("No 'Salary' column found!")

    # 5. Outlier Summary
    if st.button("📋 Show Outlier Summary"):
        st.write("Outlier Detection Summary:")
        st.write(f"Autoencoder anomalies: {ae_anomaly.sum()}")
        st.write(f"DBSCAN anomalies: {db_anomaly.sum()}")
        st.write(f"High-confidence anomalies: {combined_high_conf.sum()}")

    # 6. Download Cleaned Dataset
    if st.button("💾 Download Cleaned Dataset"):
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Click to Download", data=csv, file_name="cleaned_salary_dataset.csv", mime="text/csv")
