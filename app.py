import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========== Page Configuration ==========
st.set_page_config(
    page_title="UCS Prediction App",
    page_icon="🧱",
    layout="wide"
)

# ========== Load Model ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load('ucs_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("❌ Model file not found!")
        return None, None

model, feature_names = load_model()

# ========== Custom CSS for Geotechnical Theme ==========
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #f5f0e8;
        }
        
        /* All text color fix */
        .stApp, .stMarkdown, .stText, .stNumberInput label, .stSelectbox label, .stTextInput label {
            color: #2d1f14 !important;
        }
        
        /* Title styling */
        .main-title {
            color: #3d2b1f !important;
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            padding: 10px 0;
            border-bottom: 4px solid #8b6b4c;
            margin-bottom: 10px;
        }
        
        /* Subtitle styling */
        .sub-title {
            color: #5c3d2e !important;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        /* Input section header */
        .section-header {
            background-color: #d4c5b2;
            padding: 12px 20px;
            border-radius: 8px;
            color: #2d1f14 !important;
            font-weight: bold;
            font-size: 20px;
            border-left: 6px solid #8b6b4c;
            margin-bottom: 20px;
        }
        
        /* Prediction box */
        .prediction-box {
            background: linear-gradient(145deg, #d4c5b2, #c4b5a2);
            border: 3px solid #8b6b4c;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            margin: 20px 0;
        }
        
        .prediction-value {
            color: #2d1f14 !important;
            font-size: 48px;
            font-weight: bold;
        }
        
        .prediction-label {
            color: #3d2b1f !important;
            font-size: 20px;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #8b6b4c;
            color: white !important;
            font-weight: bold;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 40px;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #6b4c3b;
            transform: scale(1.02);
        }
        
        /* Input fields */
        .stNumberInput > div > div > input {
            background-color: #faf7f2;
            border: 2px solid #c4b5a2;
            border-radius: 6px;
            color: #2d1f14 !important;
        }
        
        .stNumberInput label {
            color: #2d1f14 !important;
            font-weight: 500;
        }
        
        /* Expander header */
        .streamlit-expanderHeader {
            background-color: #d4c5b2;
            border-radius: 8px;
            color: #2d1f14 !important;
            font-weight: 600;
        }
        
        /* Expander content */
        .streamlit-expanderContent {
            background-color: #faf7f2;
            border-radius: 0 0 8px 8px;
        }
        
        /* Metric box */
        [data-testid="metric-container"] {
            background-color: #e8ddd0;
            border-radius: 12px;
            padding: 15px;
            border: 2px solid #8b6b4c;
        }
        
        /* Dataframe text */
        .dataframe {
            color: #2d1f14 !important;
        }
        
        /* Warning and info boxes */
        .stAlert {
            color: #2d1f14 !important;
        }
        
        /* Caption text */
        .stCaption {
            color: #4d3a2a !important;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Title Section ==========
st.markdown('<div class="main-title">🏗️ UCS Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Geotechnical Engineering - Unconfined Compressive Strength Prediction</div>', unsafe_allow_html=True)

# ========== About Section ==========
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This application uses a **Gradient Boosting** machine learning model to predict the 
    **Unconfined Compressive Strength (UCS)** of soil based on 20 input parameters.
    
    **Model Performance:**
    - Trained on real geotechnical data
    - Predicts UCS in **kPa** (kilopascals)
    
    **How to use:**
    1. Enter all 20 soil parameters below
    2. Click the **"Predict UCS"** button
    3. View the predicted strength result
    """)

# ========== Input Section ==========
st.markdown('<div class="section-header">📊 Enter 20 Soil Parameters</div>', unsafe_allow_html=True)

if feature_names:
    # Split input fields into two columns for better layout
    col1, col2 = st.columns(2)
    
    input_values = []
    mid_point = len(feature_names) // 2
    
    # Dynamically create input fields for all features
    for i, feature in enumerate(feature_names):
        if i < mid_point:
            with col1:
                value = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{i}",
                    help=f"Enter value for {feature}"
                )
                input_values.append(value)
        else:
            with col2:
                value = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{i}",
                    help=f"Enter value for {feature}"
                )
                input_values.append(value)
    
    # ========== Prediction Button ==========
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict UCS", type="primary", use_container_width=True)
    
    # ========== Prediction Logic ==========
    if predict_button:
        if model is not None:
            try:
                # Convert inputs to DataFrame
                input_df = pd.DataFrame([input_values], columns=feature_names)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # ========== Display Results ==========
                st.markdown("---")
                st.markdown('<div class="section-header">✅ Prediction Result</div>', unsafe_allow_html=True)
                
                # Prediction Box with styled output
                st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">Unconfined Compressive Strength (UCS)</div>
                        <div class="prediction-value">{prediction:.2f} kPa</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Visual progress bar for UCS value
                st.progress(min(prediction / 1000, 1.0))
                st.caption(f"📊 UCS Value: {prediction:.2f} kPa")
                
                # ========== Show input data ==========
                with st.expander("📋 View Input Data"):
                    st.dataframe(input_df, use_container_width=True)
                    
                    # Download button for input data
                    csv = input_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Input Data (CSV)",
                        data=csv,
                        file_name="input_data.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please make sure all values are entered correctly.")
        else:
            st.error("❌ Model not loaded. Please check the model files.")
else:
    st.warning("⚠️ Feature names not found. Please check the 'feature_names.pkl' file.")

# ========== Footer ==========
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #4d3a2a; padding: 10px; font-size: 14px;">
        📌 Developed for Geotechnical Engineering Research | Gradient Boosting Model
    </div>
""", unsafe_allow_html=True)
