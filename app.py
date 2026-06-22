import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
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
        
        /* Title styling */
        .main-title {
            color: #5c3d2e;
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            padding: 10px 0;
            border-bottom: 4px solid #8b6b4c;
            margin-bottom: 10px;
        }
        
        /* Subtitle styling */
        .sub-title {
            color: #6b4c3b;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        /* Input section header */
        .section-header {
            background-color: #d4c5b2;
            padding: 12px 20px;
            border-radius: 8px;
            color: #3d2b1f;
            font-weight: bold;
            font-size: 20px;
            border-left: 6px solid #8b6b4c;
            margin-bottom: 20px;
        }
        
        /* Prediction box */
        .prediction-box {
            background: linear-gradient(145deg, #e8ddd0, #d4c5b2);
            border: 3px solid #8b6b4c;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            margin: 20px 0;
        }
        
        .prediction-value {
            color: #3d2b1f;
            font-size: 48px;
            font-weight: bold;
        }
        
        .prediction-label {
            color: #5c3d2e;
            font-size: 20px;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #8b6b4c;
            color: white;
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
        }
        
        /* Expander header */
        .streamlit-expanderHeader {
            background-color: #d4c5b2;
            border-radius: 8px;
            color: #3d2b1f;
            font-weight: 600;
        }
        
        /* Metric box */
        [data-testid="metric-container"] {
            background-color: #e8ddd0;
            border-radius: 12px;
            padding: 15px;
            border: 2px solid #8b6b4c;
        }
        
        /* Image container styling */
        .image-container {
            display: flex;
            justify-content: center;
            margin: 10px 0 20px 0;
        }
        
        .image-container img {
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            max-width: 100%;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Header with Image from GitHub ==========

# Display image from GitHub repository
# URL format: https://raw.githubusercontent.com/USERNAME/REPOSITORY/BRANCH/FILENAME
st.markdown("""
    <div class="image-container">
        <img src="https://raw.githubusercontent.com/Baldr1919/UCS/main/soil.jpeg" 
             alt="Soil Profile" 
             width="800">
    </div>
""", unsafe_allow_html=True)

# ========== Title ==========
st.markdown('<div class="main-title">🏗️ UCS Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Geotechnical Engineering - Unconfined Compressive Strength Prediction</div>', unsafe_allow_html=True)

# ========== Info Section ==========
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This application uses a **Random Forest** machine learning model to predict the 
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
    # Split into two columns
    col1, col2 = st.columns(2)
    
    input_values = []
    mid_point = len(feature_names) // 2
    
    # Create input fields dynamically
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
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict UCS", type="primary", use_container_width=True)
    
    # ========== Prediction ==========
    if predict_button:
        if model is not None:
            try:
                # Create DataFrame from inputs
                input_df = pd.DataFrame([input_values], columns=feature_names)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # ========== Display Result ==========
                st.markdown("---")
                st.markdown('<div class="section-header">✅ Prediction Result</div>', unsafe_allow_html=True)
                
                # Prediction Box
                st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">Unconfined Compressive Strength (UCS)</div>
                        <div class="prediction-value">{prediction:.2f} kPa</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress bar visualization
                st.progress(min(prediction / 1000, 1.0))
                st.caption(f"📊 UCS Value: {prediction:.2f} kPa")
                
                # Show input data
                with st.expander("📋 View Input Data"):
                    st.dataframe(input_df, use_container_width=True)
                    
                    # Download input data
                    csv = input_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Input Data (CSV)",
                        data=csv,
                        file_name="input_data.csv",
                        mime="text/csv"
                    )
                
                # Feature importance
                with st.expander("🌟 Feature Importance"):
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.bar_chart(importance_df.set_index('Feature'))
                
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
    <div style="text-align: center; color: #6b4c3b; padding: 10px; font-size: 14px;">
        📌 Developed for Geotechnical Engineering Research | Random Forest Model
    </div>
""", unsafe_allow_html=True)
