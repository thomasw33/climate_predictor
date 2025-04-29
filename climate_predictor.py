import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 📂 Load models and encoders
avg_temp_model = joblib.load("avg_temp_model_with_crop.pkl")
precip_model = joblib.load("precip_model_with_crop.pkl")
state_encoder = joblib.load("state_encoder.pkl")
commodity_encoder = joblib.load("commodity_encoder.pkl")

# 📂 Load your dataset for dropdowns
df = pd.read_csv("avg_temp_combined.csv")  # or precipitation_combined.csv, both have similar structure

# 🖥️ Streamlit UI
st.title("🌎 Climate Predictor App")
st.subheader("Predict Avg Temp and Precipitation by Year, State, and Crop")

# 📅 Year input
year = st.number_input("Select a Year:", min_value=1990, max_value=2025, value=2020)

# 📍 State input
states = sorted(df["state"].dropna().unique())
state = st.selectbox("Select a State:", states)

# 🌾 Crop input
crops = sorted(df["commodity"].dropna().unique())
crop = st.selectbox("Select a Crop:", crops)

# 📥 Button
if st.button("Predict"):
    # Encode the state and crop
    try:
        state_encoded = state_encoder.transform([state])[0]
        crop_encoded = commodity_encoder.transform([crop])[0]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Dummy value (you can leave at 0.0 if you want)
    value = 0.0

    # Build input DataFrame
    input_features = pd.DataFrame([{
        "year": year,
        "state_encoded": state_encoded,
        "commodity_encoded": crop_encoded,
        "value": value
    }])

    # Make predictions
    avg_temp_pred = avg_temp_model.predict(input_features)[0]
    precip_pred = precip_model.predict(input_features)[0]

    # 🎯 Display results
    st.success(f"Predicted Average Temperature for {crop} in {state} in {year}: **{avg_temp_pred:.2f} °F**")
    st.success(f"Predicted Precipitation for {crop} in {state} in {year}: **{precip_pred:.2f} Inches**")
