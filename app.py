# web link for the app
# https://cadpredictor-novaheart.streamlit.app/

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import zipfile


# Page config
st.set_page_config(
    page_title="❤️ Coronary Artery Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Custom header with gradient background and emoji
st.markdown("""
    <div style="background: linear-gradient(90deg, #ffe6ea 0%, #eaf6ff 100%); 
                border-radius: 10px; padding: 12px 0; margin-bottom:25px;">
      <h1 style="color: #d6293e; text-align:center; margin:0;">❤️ Coronary Artery Disease <br> Risk Predictor</h1>
      <p style="color: #444; text-align:center; margin:0;">Enter your health details to predict risk of CAD (Coronary Artery Disease).</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for logo and info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2656/2656390.png", width=120)
st.sidebar.markdown(
    # its Arjun Singh and team's project expected to not to be used directly or indirectly by anyone else
    "<h3 style='color:#d6293e'>Project Info</h3><p>This tool uses a ML model (Random Forest) trained on real clinical data.</p>",
    unsafe_allow_html=True
)

# Load trained model
with open('xgb_cad_model.pkl', 'rb') as f:
    model = pickle.load(f)

#with zipfile.ZipFile('rf_cad_model.zip', 'r') as zip_ref:        # uncomment to load random-forest trained model
#    with zip_ref.open('rf_cad_model.pkl') as f:
#        model = pickle.load(f)


# Main Input Section
st.subheader("Patient Risk Factors")

# Use columns for compact layout
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age (years)', min_value=18, max_value=100, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    height = st.number_input('Height (cm)', min_value=120, max_value=220, value=170)
    weight = st.number_input('Weight (kg)', min_value=35, max_value=200, value=70)
    ap_hi = st.number_input('Systolic BP', min_value=80, max_value=250, value=120)
    ap_lo = st.number_input('Diastolic BP', min_value=40, max_value=150, value=80)
with col2:
    cholesterol = st.selectbox('Cholesterol', ['Normal', 'Above normal', 'Well above normal'])
    gluc = st.selectbox('Glucose', ['Normal', 'Above normal', 'Well above normal'])
    smoke = st.selectbox('Smoker', ['No', 'Yes'])
    alco = st.selectbox('Alcohol intake', ['No', 'Yes'])
    active = st.selectbox('Physically Active', ['No', 'Yes'])

# Prepare input for prediction
gender_num = 1 if gender == 'Male' else 0
chol_num = {'Normal': 1, 'Above normal': 2, 'Well above normal': 3}[cholesterol]
gluc_num = {'Normal': 1, 'Above normal': 2, 'Well above normal': 3}[gluc]
smoke_num = 1 if smoke == 'Yes' else 0
alco_num = 1 if alco == 'Yes' else 0
active_num = 1 if active == 'Yes' else 0

input_dict = {
    "age": [age],
    "gender": [gender_num],
    "height": [height],
    "weight": [weight],
    "ap_hi": [ap_hi],
    "ap_lo": [ap_lo],
    "cholesterol": [chol_num],
    "gluc": [gluc_num],
    "smoke": [smoke_num],
    "alco": [alco_num],
    "active": [active_num]
}
features_df = pd.DataFrame(input_dict)


# Prediction Section
st.markdown("---")
st.subheader('Risk Prediction')

if st.button('Predict Risk', help="Click to get your CAD risk!"):
    proba = model.predict_proba(features_df)[0,1]
    pred = model.predict(features_df)[0]
    pred_text = 'CAD Present' if pred == 1 else 'No CAD Risk'

    # Display probability as a progress bar
    st.markdown(f"<b>Risk Probability:</b> {100*proba:.1f}%", unsafe_allow_html=True)
    st.progress(proba)

    # Show result with emoji and color
    if pred == 1:
        st.error("🚨 **CAD Present**: You are at risk. Please consult your cardiologist!")
    else:
        st.success("✅ **No CAD Risk**: Keep maintaining a healthy lifestyle!")

    # Additional visuals
    st.image("https://cdn-icons-png.flaticon.com/512/625/625047.png", width=70)
    st.caption("Output based on the model is for educational purposes. Please consult medical professionals for diagnosis.")

# Footer
st.markdown("---")
st.markdown("<center><sub>Powered by Random Forest ML & Streamlit | Designed for CAD risk prediction</sub></center>", unsafe_allow_html=True)
