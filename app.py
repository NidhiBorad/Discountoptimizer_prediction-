# app.py
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('models/tuned_model.pkl')

# Title
st.title("🛍️ Discount Optimization Predictor")

st.markdown("Enter product details below to predict whether it will convert!")

# Inputs
base_price = st.number_input("Base Price", min_value=10.0, max_value=10000.0, step=10.0)
discount = st.slider("Discount Offered (%)", min_value=0, max_value=100, value=20)
category_enc = st.number_input("Category (encoded)", min_value=0, max_value=10, step=1)
brand_enc = st.number_input("Brand (encoded)", min_value=0, max_value=50, step=1)
popularity_enc = st.number_input("Popularity (encoded)", min_value=0, max_value=5, step=1)

# Predict button
if st.button("Predict Conversion"):
    features = np.array([[base_price, discount, category_enc, brand_enc, popularity_enc]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("✅ This product is likely to be **converted** (purchased)!")
    else:
        st.warning("❌ This product is **not likely** to convert.")

# Optional: Add a simple profit calculator
st.markdown("---")
st.subheader("💰 Profit Calculator")
if st.checkbox("Show Estimated Profit"):
    profit = base_price * (1 - discount / 100)
    st.write(f"Estimated Profit per Conversion: ₹{profit:.2f}")


