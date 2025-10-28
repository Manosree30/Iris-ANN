# 🧠 Iris Extended Classification using Artificial Neural Networks (ANN)

## 📘 Project Overview
This project uses an **Artificial Neural Network (ANN)** to classify flowers based on an **extended version of the Iris dataset**.  
The dataset includes both the original Iris features and additional engineered attributes such as petal/sepal ratios, area, curvature, and texture — making it a richer dataset for deep learning.

---

## 🧩 Objectives
- Perform **data preprocessing** (handle missing values, encode categories, treat outliers).
- Visualize feature distributions and skewness.
- Build and train an **Artificial Neural Network** for multi-class classification.
- Optimize hyperparameters using **Keras Tuner**.
- Deploy a user-friendly **Streamlit web app** for live predictions.

---

## 📂 Project Structure
Iris-ANN/
│
├── iris_extended.csv # Dataset file
├── app.py # Streamlit application
├── best_ann_model.h5 # Trained ANN model
├── scaler.pkl # Saved StandardScaler
├── your_notebook.ipynb # Full preprocessing & training notebook
├── ann_tuning/ # Hyperparameter tuning logs
└── README.md # Project documentation


---

## ⚙️ Technologies Used
- **Python 3.10+**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **Pandas / NumPy**
- **Keras Tuner**
- **Streamlit** (for deployment)

---

## 🔬 Workflow Summary

### 1. **Data Preprocessing**
- Loaded the dataset using `pandas`.
- Encoded categorical columns (`species`, `soil_type`) using `LabelEncoder`.
- Applied log transformation to reduce skewness in selected columns.
- Detected and capped outliers using IQR method.
- Standardized numeric features using `StandardScaler`.

### 2. **Exploratory Data Analysis**
- Visualized feature distributions using histograms and KDE plots.
- Inspected outliers through boxplots before and after capping.

### 3. **Model Building**
- Designed an ANN with multiple hidden layers using `Sequential` API.
- Activation Functions: **ReLU**, **Softmax**.
- Loss Function: **Sparse Categorical Crossentropy**.
- Optimizer: **Adam**.

### 4. **Hyperparameter Tuning**
- Used **Keras Tuner (RandomSearch)** to tune:
  - Number of hidden layers  
  - Units per layer  
  - Dropout rate  
  - Activation function  
  - Learning rate
- Achieved accuracy close to **100%** on the test set.

## Streamlit Deployment
 Built an interactive web app to input feature values.

Scales user input and predicts flower species using the trained ANN model.
app("https://manosree30-iris-ann-app-vakpb1.streamlit.app/")

## Results

Successfully trained and optimized a deep learning model for multi-class flower classification.

Achieved ~100% test accuracy after hyperparameter tuning.

Created an end-to-end workflow — from preprocessing to web deployment.