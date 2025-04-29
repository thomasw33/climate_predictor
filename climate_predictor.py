import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 📂 Load models and encoders (avg temp & precipitation)
avg_temp_model = joblib.load("avg_temp_model_with_crop.pkl")
precip_model = joblib.load("precip_model_with_crop.pkl")
state_encoder_main = joblib.load("state_encoder.pkl")
commodity_encoder_main = joblib.load("commodity_encoder.pkl")

# 📂 Load models and encoders (min/max temp)
min_temp_model = joblib.load("min_temp_model.pkl")
max_temp_model = joblib.load("max_temp_model.pkl")
state_encoder_minmax = joblib.load("minmax_state_encoder.pkl")
commodity_encoder_minmax = joblib.load("minmax_commodity_encoder.pkl")

# 📄 Load dataset for dropdowns
df = pd.read_csv("avg_temp_combined.csv")

# 🖥️ UI Title
st.title("🌎 Climate Predictor App")
st.subheader("Predict Avg/Min/Max Temperature and Precipitation by Year, State, and Crop")

# 📅 Year input
year = st.number_input("Select a Year:", min_value=1990, max_value=2025, value=2020)

# 📍 State input
states = sorted(df["state"].dropna().unique())
state = st.selectbox("Select a State:", states)

# 🌾 Crop input
crops = sorted(df["commodity"].dropna().unique())
crop = st.selectbox("Select a Crop:", crops)

# 📥 Predict Button
if st.button("Predict"):
    try:
        # 🔐 Encode using both sets of encoders
        state_encoded_main = state_encoder_main.transform([state])[0]
        crop_encoded_main = commodity_encoder_main.transform([crop])[0]

        state_encoded_minmax = state_encoder_minmax.transform([state])[0]
        crop_encoded_minmax = commodity_encoder_minmax.transform([crop])[0]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Dummy value for input
    value = 0.0

    # Input DataFrame for main models
    input_main = pd.DataFrame([{
        "year": year,
        "state_encoded": state_encoded_main,
        "commodity_encoded": crop_encoded_main,
        "value": value
    }])

    # Input DataFrame for min/max models
    input_minmax = pd.DataFrame([{
        "year": year,
        "state_encoded": state_encoded_minmax,
        "commodity_encoded": crop_encoded_minmax
    }])

    # 🧠 Make predictions
    avg_temp_pred = avg_temp_model.predict(input_main)[0]
    precip_pred = precip_model.predict(input_main)[0]
    min_temp_pred = min_temp_model.predict(input_minmax)[0]
    max_temp_pred = max_temp_model.predict(input_minmax)[0]

    # 📊 Show predictions
    st.success(f"🌡️ Predicted **Average Temperature**: **{avg_temp_pred:.2f} °F**")
    st.success(f"🌧️ Predicted **Precipitation**: **{precip_pred:.2f} Inches**")
    st.success(f"❄️ Predicted **Minimum Temperature**: **{min_temp_pred:.2f} °F**")
    st.success(f"🔥 Predicted **Maximum Temperature**: **{max_temp_pred:.2f} °F**")
