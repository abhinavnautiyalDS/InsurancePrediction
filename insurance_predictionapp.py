import os

os.system("pip install -r requirements.txt")

import streamlit as st

import onnxruntime as rt
import numpy as np
import gdown  # For downloading files from Google Drive
import os
import zipfile  # Extract ZIP files


# Streamlit Page Configuration
st.set_page_config(page_title="Insurance Policy Predictor", page_icon="💰", layout="wide")

# Google Drive File (ONNX Model)
file_id = "1r86nZ1L5K8-aoLfA0uTm_m1C5auFS64s"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_filename = "RandomForestModel.onnx"
# Download the Joblib model if not already present
#https://drive.google.com/file/d/1r86nZ1L5K8-aoLfA0uTm_m1C5auFS64s/view?usp=sharing
if not os.path.exists(model_filename):
    st.info("Downloading model ZIP file... 📥")
    gdown.download(download_url, model_filename, quiet=False)

    # Extract ZIP File
   

    st.success("Model downloaded successfully! 🚀")
# Load ONNX Model
@st.cache_resource
def load_model():
    return rt.InferenceSession(model_filename)

# Try to load the model
session = load_model()
# Function to make predictions
if session:
    st.success("Created by Abhinav Nautiyal 🚀")
else:
    st.error("Failed to load the model. Please check the file path.")

# Min-Max Scaling Constants
MIN_REGION, MAX_REGION = 0.0, 52.0  # Example values
MIN_CHANNEL, MAX_CHANNEL = 1.0, 165.0
MIN_PREMIUM, MAX_PREMIUM = 2630.0, 540000.0

def min_max_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Function to make predictions
def predict_response(features):
    if session is None:
        return "Model could not be loaded!"

    try:
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Convert features to numpy array
        input_data = np.array([features], dtype=np.float32)

        # Debug: Print feature shape
        st.write(f"🔍 Feature Count: {input_data.shape[1]}")

        # Run inference
        prediction = session.run([output_name], {input_name: input_data})[0]

        return int(prediction[0])
    except Exception as e:
        return f"⚠️ Prediction failed: {e}"

# UI Components
st.markdown("<h1 style='text-align: center;'>🚀 Vehicle Insurance Policy Predictor 💰</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FEB47B;'>Find out if a customer will take the policy!</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Customer Information")
    age_60_plus = st.checkbox("Age 60+")
    age_40_60 = st.checkbox("Age Between 40 and 60")
    age_20_40 = st.checkbox("Age Between 20 and 40")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    driving_license = st.checkbox("Has Driving License")
    region_code = min_max_scale(st.number_input("Region Code", min_value=0.0, value=28.0, step=1.0), MIN_REGION, MAX_REGION)
    previously_insured = st.radio("Previously Insured", ["Yes", "No"], horizontal=True)

with col2:
    st.markdown("### 🚗 Vehicle & Policy Details")
    vehicle_age = st.radio("Vehicle Age", ["<1 year", "1-2 years", ">2 years"], horizontal=True)
    annual_premium = min_max_scale(st.number_input("Annual Premium", min_value=0.0, value=40000.0), MIN_PREMIUM, MAX_PREMIUM)
    policy_sales_channel = min_max_scale(st.number_input("Policy Sales Channel", min_value=0.0, value=26.0, step=1.0), MIN_CHANNEL, MAX_CHANNEL)
    customer_type = st.radio("Customer Type", ["Long-term", "Mid-term", "Short-term"], horizontal=True)

# Predict Button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("🔥 Predict Now 🔥"):
    user_input = [
        int(age_60_plus), int(age_40_60), int(age_20_40),
        1 if gender == "Male" else 0, 1 if gender == "Female" else 0, int(driving_license),
        float(region_code), 1 if vehicle_age == "1-2 years" else 0, 1 if vehicle_age == "<1 year" else 0, 1 if vehicle_age == ">2 years" else 0,
        float(annual_premium), 1 if previously_insured == "Yes" else 0, 1 if previously_insured == "No" else 0,
        float(policy_sales_channel), 1 if customer_type == "Long-term" else 0, 1 if customer_type == "Mid-term" else 0
    ]
    
    if len(user_input) != 16:
        st.error(f"⚠️ Model expects 16 features, but got {len(user_input)}. Please check inputs.")
    else:
        prediction = predict_response(user_input)
        if prediction is not None:
            result = "✅ Likely to Take the Policy!" if int(prediction) == 1 else "❌ Not Interested in the Policy"
            st.success(result)
            if int(prediction) == 1:
                st.balloons()
st.markdown("</div>", unsafe_allow_html=True)
