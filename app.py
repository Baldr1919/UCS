import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="UCS Soil Prediction",
    page_icon="🧱",
    layout="wide"
)

# ========== Header with image ==========
# Add header image with PNG format
try:
    st.image('header_image.png', use_container_width=True)
except:
    # If image not found, show title
    st.title("🧱 Unconfined Compressive Strength (UCS) Prediction")

st.markdown("---")

# ========== Load model ==========
@st.cache_resource
def load_models():
    try:
        model = joblib.load('ucs_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found! Error: {str(e)}")
        st.info("Please make sure the following files are in the same directory:\n- ucs_model.pkl\n- feature_names.pkl")
        return None, None

model, feature_names = load_models()

# ========== Description ==========
with st.expander("ℹ️ About this app"):
    st.write("""
    This application uses a **Random Forest** model trained on real data to predict
    the Unconfined Compressive Strength (UCS) of soil based on 20 input parameters.
    
    **Model Accuracy:**  
    Please enter all parameters carefully to get the best results.
    """)

# ========== Inputs ==========
st.header("📊 Enter Soil Parameters")

if feature_names:
    # Split into two columns for better display
    col1, col2 = st.columns(2)
    
    input_values = []
    mid_point = len(feature_names) // 2
    
    # Create dynamic inputs
    for i, feature in enumerate(feature_names):
        if i < mid_point:
            with col1:
                value = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{i}"
                )
                input_values.append(value)
        else:
            with col2:
                value = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{i}"
                )
                input_values.append(value)
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("🔮 Predict UCS", type="primary", use_container_width=True)
    
    # ========== Prediction ==========
    if predict_button:
        if model is not None:
            try:
                # Create dataframe
                input_df = pd.DataFrame([input_values], columns=feature_names)
                
                # Predict
                prediction = model.predict(input_df)[0]
                
                # Display result
                st.markdown("---")
                st.header("✅ Prediction Result")
                
                # Display UCS value
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="Unconfined Compressive Strength (UCS)",
                        value=f"{prediction:.2f} kPa",
                        delta="Predicted by model"
                    )
                    
                    # Graphical display
                    st.progress(min(prediction / 1000, 1.0))
                    st.caption(f"UCS Value: {prediction:.2f} kPa")
                
                # Display input data
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
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please make sure all values are entered correctly.")
        else:
            st.error("❌ Model not loaded. Please build the model first.")

# ========== Footer ==========
st.markdown("---")
st.caption("📌 This app is built for scientific and research purposes. | Random Forest Model")
