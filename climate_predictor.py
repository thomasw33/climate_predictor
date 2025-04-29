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

# 📄 Load dataset for dropdowns and actual lookup
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

    # 📦 Prepare input for models
    input_main = pd.DataFrame([{
        "year": year,
        "state_encoded": state_encoded_main,
        "commodity_encoded": crop_encoded_main,
        "value": value
    }])

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

    # 📄 Try to find actual values
    actual_row = df[(df["year"] == year) & (df["state"] == state) & (df["commodity"] == crop)]

    if not actual_row.empty:
        actual_avg_temp = actual_row["avg_temp"].values[0]
        actual_precip = actual_row["precipitation"].values[0]
        actual_min_temp = actual_row["min_temp"].values[0]
        actual_max_temp = actual_row["max_temp"].values[0]
    else:
        actual_avg_temp = actual_precip = actual_min_temp = actual_max_temp = None

    # 📊 Display comparison
    st.subheader("🌟 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Predicted Values")
        st.write(f"**Average Temperature**: {avg_temp_pred:.2f} °F")
        st.write(f"**Precipitation**: {precip_pred:.2f} Inches")
        st.write(f"**Minimum Temperature**: {min_temp_pred:.2f} °F")
        st.write(f"**Maximum Temperature**: {max_temp_pred:.2f} °F")

    with col2:
        st.markdown("### 🟠 Actual Values")
        if actual_row.empty:
            st.write("❌ No actual data available for this selection.")
        else:
            st.write(f"**Average Temperature**: {actual_avg_temp:.2f} °F")
            st.write(f"**Precipitation**: {actual_precip:.2f} Inches")
            st.write(f"**Minimum Temperature**: {actual_min_temp:.2f} °F")
            st.write(f"**Maximum Temperature**: {actual_max_temp:.2f} °F")
