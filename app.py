import streamlit as st
from tensorflow.keras.models import load_model

import numpy as np

# Load model
model = load_model('my_model.h5')

st.title("Simple TensorFlow Model Demo")
st.write("Enter two numbers to perform XOR prediction:")

num1 = st.number_input("Input 1", min_value=0, max_value=1, value=0)
num2 = st.number_input("Input 2", min_value=0, max_value=1, value=0)

if st.button("Predict"):
    input_data = np.array([[num1, num2]])
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction:.3f} (rounded: {int(prediction>=0.5)})")