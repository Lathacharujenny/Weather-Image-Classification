import streamlit as st
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model
import json
from PIL import Image
import numpy as np
import pickle

# Paths to preloaded model and labels
MODEL_PATH = "artifacts/model_training/models/Resnet152V2.pkl"  
LABELS_PATH = "artifacts/labels/labels.json"  
IMG_SIZE = 256


# Load model and labels when the app starts
@st.cache_resource
def load_resources():
    with open(MODEL_PATH, 'rb') as file:
      model = pickle.load(file)
    #model = load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as file:
        labels = json.load(file)
    return model, labels

# Streamlit App
def main():
    st.title("Image Classification App")
    st.write("Upload an image to classify the Weather")

    # Load model and labels
    model, labels = load_resources()

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Classify the image when the "Classify Image" button is clicked
        if st.button("Classify Image"):
            try:
                # Preprocess the image in memory
                img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE)))  # Resize to model input size
                img_array_expanded = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_array_expanded)

                # Predict using the model
                predictions = model.predict(img_preprocessed)
                predicted_index = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)
                predicted_label = labels[str(predicted_index)]

                # Display the prediction and confidence
                st.success(f"Prediction: **{predicted_label}**")
                st.info(f"Confidence: **{confidence:.2f}**")

            except Exception as e:
                st.error(f"Error occurred during prediction: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
