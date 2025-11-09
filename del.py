import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Delivery Delay Prediction", layout="centered")
st.title("🚚 Delivery Delay Prediction (k-NN)")

# ===== Model loading =====
# Option A: keep knn_model.sav in the same folder as this script
DEFAULT_MODEL_PATH = Path(__file__).with_name("knn_model.sav")

# Option B: let user pick a file if default not found
model_file = None
if DEFAULT_MODEL_PATH.exists():
    model_file = DEFAULT_MODEL_PATH
else:
    st.warning("`knn_model.sav` not found next to this script. Upload it below or set an absolute path.")
    uploaded = st.file_uploader("Upload knn_model.sav", type=["sav","pkl","joblib"])
    if uploaded is not None:
        model_file = uploaded

loaded_model = None
if model_file is not None:
    try:
        # joblib handles sklearn models saved with joblib.dump
        loaded_model = joblib.load(model_file)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")

# ===== Feature schema =====
COLUMNS = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

st.write("### Enter inputs")
col1, col2 = st.columns(2)

with col1:
    Delivery_Distance = st.number_input("Delivery Distance (km)", min_value=0.0, step=0.1)
    Traffic_Congestion = st.number_input("Traffic Congestion (1–5)", min_value=1, max_value=5, step=1)
    Weather_Condition = st.number_input("Weather Condition (1–5)", min_value=1, max_value=5, step=1)
    Delivery_Slot = st.number_input("Delivery Slot (1-based index)", min_value=1, step=1)
    Driver_Experience = st.number_input("Driver Experience (years)", min_value=0.0, step=0.1)
    Num_Stops = st.number_input("Number of Stops", min_value=0, step=1)

with col2:
    Vehicle_Age = st.number_input("Vehicle Age (years)", min_value=0.0, step=0.1)
    Road_Condition_Score = st.number_input("Road Condition Score (1–5)", min_value=1, max_value=5, step=1)
    Package_Weight = st.number_input("Package Weight (kg)", min_value=0.0, step=0.1)
    Fuel_Efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=0.0, step=0.1)
    Warehouse_Processing_Time = st.number_input("Warehouse Processing Time (min)", min_value=0.0, step=0.5)

# Assemble row in the exact required order
row = [
    Delivery_Distance, Traffic_Congestion, Weather_Condition,
    Delivery_Slot, Driver_Experience, Num_Stops, Vehicle_Age,
    Road_Condition_Score, Package_Weight, Fuel_Efficiency,
    Warehouse_Processing_Time
]
input_df = pd.DataFrame([row], columns=COLUMNS)

# Enforce numeric types
for c in COLUMNS:
    input_df[c] = pd.to_numeric(input_df[c], errors="coerce")
if input_df.isna().any().any():
    st.info("Fill all inputs with valid numbers.")

st.write("#### Preview of input to model")
st.dataframe(input_df)

# ===== Predict =====
if st.button("Predict Delivery Delay"):
    if loaded_model is None:
        st.error("Load the model first.")
    elif input_df.isna().any().any():
        st.error("Please provide valid numbers for all inputs.")
    else:
        try:
            pred = loaded_model.predict(input_df)
            label = int(pred[0])
            msg = "0 (No significant delay expected)" if label == 0 else "1 (Delay expected)"
            st.success(f"**Predicted Delivery Delay:** {msg}")

            # If classifier has predict_proba, show confidence
            if hasattr(loaded_model, "predict_proba"):
                proba = loaded_model.predict_proba(input_df)[0]
                st.write("Class probabilities:")
                st.write({f"class_{i}": float(p) for i, p in enumerate(proba)})
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Tip: If your model was saved with `joblib.dump`, prefer loading with `joblib.load` (as above).")
