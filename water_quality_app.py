# ============================================================ 
# 🧠 AI-Based Water Quality Prediction - Styled Browser Version
# ============================================================ 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ============================================================ 
# STEP 0: Generate Synthetic Dataset
# ============================================================ 
np.random.seed(42)
data = []

for _ in range(300):
    pH = np.round(np.random.uniform(5.0, 9.0), 2)
    bod = np.round(np.random.uniform(1, 10), 2)
    cod = np.round(np.random.uniform(10, 60), 2)
    tds = np.round(np.random.uniform(200, 1200), 2)

    if 6.5 <= pH <= 8.5 and bod <= 3 and cod <= 20 and tds <= 500:
        quality = "Safe"
    elif (bod <= 6 and cod <= 40 and tds <= 1000) and not (6.5 <= pH <= 8.5 and bod <= 3 and cod <= 20 and tds <= 500):
        quality = "Needs Treatment"
    else:
        quality = "Unfit"

    data.append([pH, bod, cod, tds, quality])

df = pd.DataFrame(data, columns=["pH", "BOD", "COD", "TDS", "Quality"])

# ============================================================ 
# STEP 1: Train Model
# ============================================================ 
X = df[["pH", "BOD", "COD", "TDS"]]
y = df["Quality"]

from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(model, "water_quality_model.pkl")

# ============================================================ 
# STEP 2: Streamlit Styled App
# ============================================================ 
st.set_page_config(page_title="💧 AI Water Quality Prediction", layout="wide")
st.markdown("<h1 style='text-align:center; color:#003366;'>💧 AI-Based Water Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:#007BFF;'>Developed by Vinit - B.Tech (CSE), MIET</h5>", unsafe_allow_html=True)
st.write("---")

# Input layout in columns
col1, col2 = st.columns(2)
with col1:
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.01)
    cod = st.number_input("COD (mg/L)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
with col2:
    bod = st.number_input("BOD (mg/L)", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
    tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=400.0, step=1.0)

st.write("---")
if st.button("🔍 Predict Water Quality"):
    features = pd.DataFrame([[pH, bod, cod, tds]], columns=["pH", "BOD", "COD", "TDS"])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features)[0]

    if prediction == "Safe":
        st.success("✅ Water is Safe for Reuse")
    elif prediction == "Needs Treatment":
        st.warning("⚠️ Water Needs Treatment")
    else:
        st.error("❌ Water is Unfit")

# Optional checkboxes
st.write("---")
col3, col4 = st.columns(2)
with col3:
    if st.checkbox("Show Model Accuracy"):
        st.info(f"Model Accuracy: {acc*100:.2f}%")
with col4:
    if st.checkbox("Show Class Distribution"):
        st.bar_chart(df["Quality"].value_counts())

st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>© Developed by Vinit | B.Tech (CSE), MIET</p>", unsafe_allow_html=True)


