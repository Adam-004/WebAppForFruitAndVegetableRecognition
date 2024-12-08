#Imports
from flask import Flask, request, render_template_string, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Constants
MODEL_PATH = 'model/model.keras'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_LABELS = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'peas', 'pear', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Flask Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model (with error handling for missing model file)
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback: Create a dummy model if the actual model is not available
        return Sequential([  
            Dense(128, input_shape=(224, 224, 3), activation='relu'),
            Dense(len(CLASS_LABELS), activation='softmax')
        ])

# Load the model at the start
model = load_trained_model()

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(file_path, target_size=(224, 224)):
    image = load_img(file_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for upload and prediction
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        # Check if file is included in the request
        file = request.files.get("file")
        if not file or file.filename == "":
            return "<p>No file was selected.</p>"

        # Validate the file type
        if file and allowed_file(file.filename):
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image
            image = preprocess_image(file_path)
            predictions = model.predict(image)
            score = tf.nn.softmax(predictions[0])
            index = np.argmax(score)
            confidence = f"Confidence: {(predictions[0][index] * 100):.2f}%"
            predicted_label = CLASS_LABELS[index]
            result = f"Result: {predicted_label}"

            # Render the result page with the prediction and image preview
            return render_template_string(RESULT_PAGE_HTML, 
                                          image_path=f"/uploads/{file.filename}", 
                                          result=result, 
                                          confidence = confidence,
                                          file_name=file.filename)

    # Render the upload form for GET requests
    return render_template_string(UPLOAD_PAGE_HTML)

# HTML Template for Upload Form
UPLOAD_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classification</title>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Open Sans', sans-serif; background-color: #003366; color: #fff; margin: 0; padding: 0; }
        .upload-container { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); max-width: 400px; margin: 50px auto; text-align: center; background-color: #001f3f; }
        h1 { color: #FF8500; font-weight: 600; }
        input[type="file"] { margin-bottom: 20px; padding: 15px; background-color: #003366; border: 2px solid none; border-radius: 5px; font-size: 1.1em; }
        input[type="submit"] { padding: 20px 40px; background-color: #FF8500; color: #fff; border: none; border-radius: 5px; cursor: pointer; font-size: 1.2em; }
        input[type="submit"]:hover { background-color: #FF6220; }
        .image-preview { margin-bottom: 20px; border-radius: 5px; max-width: 100%; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); }
        .image-name { color: #333333; font-size: 1.1em; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="upload-container">
        <h1>Upload an Image for Prediction</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept=".jpg,.jpeg,.png" required onchange="previewImage(event)">
            <img id="imagePreview" class="image-preview" style="display: none;" />
            <p id="imageName" class="image-name" style="display: none;"></p>
            <input type="submit" value="Predict">
        </form>
    </div>
    <script>
        function previewImage(event) {
            const reader = new FileReader();
            reader.onload = function() {
                const img = document.getElementById('imagePreview');
                img.style.display = 'block';
                img.src = reader.result;
            }
            reader.readAsDataURL(event.target.files[0]);
        }
    </script>
</body>
</html>
"""

# HTML Template for Prediction Result Page
RESULT_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Open Sans', sans-serif; background-color: #003366; color: #fff; margin: 0; padding: 0; }
        .container { max-width: 600px; margin: 50px auto; background: #fff; padding: 30px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); border-radius: 8px; text-align: center; background-color: #001f3f; color: #FFDC00; }
        h1 { color: #FF8500; font-weight: 600; }
        .result { font-size: 1.2em; margin-top: 20px; color: #FF8500; }
        .confidence { font-size: 0.9em; color: #FF6200; }
        a { display: inline-block; margin-top: 20px; padding: 12px 24px; background-color: #FF8500; color: #fff; text-decoration: none; border-radius: 5px; font-weight: 500; }
        a:hover { background-color: #FF6200; }
        .image-preview { margin-bottom: 20px; border-radius: 5px; max-width: 100%; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); }
        .image-name { color: #333333; font-size: 1.1em; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Result</h1>
        <img src="{{ image_path }}" alt="Uploaded Image" class="image-preview">
        <p class="result">{{ result }}</p>
        <p class="confidence">{{ confidence }}</p>
        <a href="/">Upload Another Image</a>
    </div>
</body>
</html>
"""

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
