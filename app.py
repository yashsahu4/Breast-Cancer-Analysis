import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load and prepare the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Create a pipeline with feature scaling and the ANN model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        max_iter=500,  # Increased iterations
        verbose=True  # Optional: enables verbose output for monitoring
    ))
])

# Train the model
X = df.drop('target', axis=1)
y = df['target']
pipeline.fit(X, y)

# Streamlit app
st.title("Breast Cancer Prediction")

# Add descriptive text
st.write("""
    **Instructions:**
    - Use the sliders to input feature values.
    - The model will predict whether the sample is benign or malignant based on the input values.
    - The prediction probability shows the confidence of the model for each class.
""")

# Create user input fields with sliders for each feature
user_input = {}
for feature in data.feature_names:
    min_value = float(df[feature].min())
    max_value = float(df[feature].max())
    mean_value = float(df[feature].mean())
    user_input[feature] = st.slider(
        feature,
        min_value,
        max_value,
        mean_value
    )

# Convert user input to DataFrame with consistent feature names
input_df = pd.DataFrame([user_input], columns=data.feature_names)

# Make prediction
prediction = pipeline.predict(input_df)
prediction_proba = pipeline.predict_proba(input_df)

# Display results
st.write(f"**Prediction:** {'Malignant' if prediction[0] == 1 else 'Benign'}")
st.write(f"**Prediction Probability:**")
st.write(f"  - Malignant: {prediction_proba[0][1]:.2f}")
st.write(f"  - Benign: {prediction_proba[0][0]:.2f}")

