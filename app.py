import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# ✅ Load the trained ANN model
MODEL_PATH = "best_ann_model.h5"
model = load_model(MODEL_PATH)

# ✅ Load the scaler if available
if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None

# 🏷️ App title
st.title("🌸 ANN Classifier Web App")
st.write("Enter the input values below to predict the species/class.")

# 🧩 Feature names — update this with your dataset column names
feature_names = [
    "elevation", "soil_type", "sepal_length", "sepal_width", "petal_length",
    "petal_width", "sepal_area", "petal_area", "sepal_aspect_ratio",
    "petal_aspect_ratio", "sepal_to_petal_length_ratio",
    "sepal_to_petal_width_ratio", "sepal_petal_length_diff",
    "sepal_petal_width_diff", "petal_curvature_mm",
    "petal_texture_trichomes_per_mm2", "leaf_area_cm2",
    "sepal_area_sqrt", "petal_area_sqrt", "area_ratios"
]

# 📥 Collect input from user
input_data = []
cols = st.columns(2)  # two-column layout for cleaner UI

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        val = st.number_input(f"{feature}", value=0.0, format="%.4f")
        input_data.append(val)

# 🧮 Convert to array and reshape
input_array = np.array(input_data).reshape(1, -1)

# 🧠 Scale the input if scaler exists
if scaler:
    input_array = scaler.transform(input_array)

# 🔮 Predict button
if st.button("Predict"):
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    st.success(f"🌼 Predicted Class: {predicted_class}")
