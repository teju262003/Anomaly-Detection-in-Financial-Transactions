import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Advanced Anomaly Detection on Financial Transactions")

uploaded_file = st.file_uploader("Upload your file here")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data")
    st.write(df.head())

    # --- Feature Engineering ---
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

    # Extract datetime features
    df['Hour'] = df['Transaction Date'].dt.hour
    df['DayOfWeek'] = df['Transaction Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Encode Transaction Type
    le = LabelEncoder()
    df['TransactionTypeEncoded'] = le.fit_transform(df['Transaction Type'])

    # Final features
    features = ['Amount', 'Balance', 'Hour', 'DayOfWeek', 'IsWeekend', 'TransactionTypeEncoded']
    X = df[features].copy()
    X = X.fillna(X.mean())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write("Standardized Feature Matrix")
    st.write(X_scaled)

    # Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X_scaled)

    # Predictions
    df['AnomalyScore'] = iso_forest.decision_function(X_scaled)
    df['IsAnomaly'] = iso_forest.predict(X_scaled)

    # Results
    st.write("Detected Fraudulent Transactions")
    st.write(df[df['IsAnomaly'] == -1])

    st.write("Number of Fraudulent Transactions")
    st.write((df['IsAnomaly'] == -1).sum())


