import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ------------------------------
# 🔹 Load Model and Scaler
# ------------------------------
MODEL_PATH = "best_ann_model.h5"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))

# ------------------------------
# 🔹 Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="🌸 Iris Extended ANN Classifier", layout="wide")
st.title("🌼 Iris Flower Classification using ANN (Extended Dataset)")
st.write("Provide the following feature values to predict the Iris species:")

# ------------------------------
# 🔹 Input Columns
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    elevation = st.number_input("Elevation", 0.0, 500.0, 150.0)
    soil_type = st.selectbox("Soil Type", ["sandy", "clay", "loamy"])
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)
    sepal_area = st.number_input("Sepal Area", 0.0, 100.0, 17.5)

with col2:
    petal_area = st.number_input("Petal Area", 0.0, 100.0, 0.5)
    sepal_aspect_ratio = st.number_input("Sepal Aspect Ratio", 0.0, 10.0, 1.5)
    petal_aspect_ratio = st.number_input("Petal Aspect Ratio", 0.0, 10.0, 4.5)
    sepal_to_petal_length_ratio = st.number_input("Sepal/Petal Length Ratio", 0.0, 10.0, 3.0)
    sepal_to_petal_width_ratio = st.number_input("Sepal/Petal Width Ratio", 0.0, 30.0, 10.0)
    sepal_petal_length_diff = st.number_input("Sepal-Petal Length Diff", -5.0, 10.0, 3.0)
    sepal_petal_width_diff = st.number_input("Sepal-Petal Width Diff", -5.0, 10.0, 3.0)

with col3:
    petal_curvature_mm = st.number_input("Petal Curvature (mm)", 0.0, 10.0, 5.0)
    petal_texture = st.number_input("Petal Texture (trichomes/mm²)", 0.0, 100.0, 20.0)
    leaf_area = st.number_input("Leaf Area (cm²)", 0.0, 100.0, 50.0)
    sepal_area_sqrt = st.number_input("Sepal Area √", 0.0, 10.0, 4.0)
    petal_area_sqrt = st.number_input("Petal Area √", 0.0, 10.0, 0.7)
    area_ratios = st.number_input("Area Ratios", 0.0, 100.0, 40.0)

# ------------------------------
# 🔹 Encode Soil Type
# ------------------------------
soil_mapping = {"sandy": 0, "clay": 1, "loamy": 2}
soil_type_encoded = soil_mapping[soil_type]

# ------------------------------
# 🔹 Prepare Input Array
# ------------------------------
input_data = np.array([[
    elevation, soil_type_encoded, sepal_length, sepal_width, petal_length, petal_width,
    sepal_area, petal_area, sepal_aspect_ratio, petal_aspect_ratio,
    sepal_to_petal_length_ratio, sepal_to_petal_width_ratio, sepal_petal_length_diff,
    sepal_petal_width_diff, petal_curvature_mm, petal_texture,
    leaf_area, sepal_area_sqrt, petal_area_sqrt, area_ratios
]])

# ------------------------------
# 🔹 Scale Input
# ------------------------------
scaled_input = scaler.transform(input_data)

# ------------------------------
# 🔹 Predict Button
# ------------------------------
if st.button("🔍 Predict Species"):
    prediction = model.predict(scaled_input)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_labels = {0: "Setosa 🌸", 1: "Versicolor 🌿", 2: "Virginica 🌺"}
    st.success(f"**Predicted Species:** {class_labels[predicted_class]}")

    st.write("🔢 Model Output Probabilities:")
    st.write({label: f"{prob:.4f}" for label, prob in zip(class_labels.values(), prediction[0])})

# ------------------------------
# 🔹 Footer
# ------------------------------
st.markdown("---")
st.caption("Developed by **Manosree Nagarajan** | Artificial Neural Network Project | BIT 💡")
