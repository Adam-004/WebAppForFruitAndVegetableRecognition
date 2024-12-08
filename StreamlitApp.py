import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Constants
MODEL_PATH = 'model/model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_LABELS = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'peas', 'pear', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Fruit and Vegetable Classification")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    index = np.argmax(score)
    confidence = f"Confidence: {(predictions[0][index] * 100):.2f}%"
    predicted_label = CLASS_LABELS[index]
    
    # Display prediction and confidence
    st.write(f"Prediction: {predicted_label}")
    st.write(confidence)
