import streamlit as st
import pandas as pd
import numpy as np
import joblib

# تنظیمات صفحه
st.set_page_config(
    page_title="پیش‌بینی UCS خاک",
    page_icon="🧱",
    layout="wide"
)

# ========== بارگذاری مدل ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load('ucs_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("❌ فایل مدل پیدا نشد!")
        return None, None

model, feature_names = load_model()

# ========== عنوان ==========
st.title("🧱 پیش‌بینی مقاومت فشاری تک محوره (UCS) خاک")
st.markdown("---")

# ========== توضیحات ==========
with st.expander("ℹ️ درباره این اپ"):
    st.write("""
    این اپلیکیشن با استفاده از مدل **Random Forest** مقاومت فشاری تک محوره (UCS) خاک را 
    بر اساس ۲۰ پارامتر ورودی پیش‌بینی می‌کند.
    """)

# ========== ورودی‌ها ==========
st.header("📊 وارد کردن پارامترهای خاک")

if feature_names:
    # تقسیم به دو ستون
    col1, col2 = st.columns(2)
    
    input_values = []
    mid_point = len(feature_names) // 2
    
    # ساخت ورودی‌ها
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
    
    # دکمه پیش‌بینی
    st.markdown("---")
    predict_button = st.button("🔮 پیش‌بینی UCS", type="primary", use_container_width=True)
    
    # ========== پیش‌بینی ==========
    if predict_button:
        if model is not None:
            try:
                # ساخت دیتافریم
                input_df = pd.DataFrame([input_values], columns=feature_names)
                
                # پیش‌بینی
                prediction = model.predict(input_df)[0]
                
                # نمایش نتیجه
                st.markdown("---")
                st.header("✅ نتیجه پیش‌بینی")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="مقاومت فشاری تک محوره (UCS)",
                        value=f"{prediction:.2f} kPa"
                    )
                    
                    # نمایش گرافیکی
                    st.progress(min(prediction / 1000, 1.0))
                    st.caption(f"مقدار UCS: {prediction:.2f} kPa")
                
                # نمایش داده‌های ورودی
                with st.expander("📋 مشاهده داده‌های ورودی"):
                    st.dataframe(input_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ خطا: {str(e)}")
        else:
            st.error("❌ مدل بارگذاری نشد!")

# ========== فوتر ==========
st.markdown("---")
st.caption("📌 این اپ برای اهداف علمی و تحقیقاتی ساخته شده است.")