# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load features and filenames
with open("features.pkl", "rb") as f:
    filenames, features = pickle.load(f)

# Folder for uploaded query images
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Pretrained VGG16 model for feature extraction
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features of an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x)
    return feat.flatten()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Save uploaded file
    file = request.files['query_img']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract features of uploaded image
    query_features = extract_features(file_path)

    # Compute similarity with dataset features
    sims = cosine_similarity([query_features], features)[0]

    # Top 3 similar images (exclude uploaded query image)
    top_indices = np.argsort(sims)[::-1]
    top_results = []
    for i in top_indices:
        if filenames[i] != file.filename:  # exclude uploaded image if same filename exists
            top_results.append(filenames[i])
        if len(top_results) == 5:  # get top 3
            break

    # Prepare full path for result images
    results = [os.path.join("static/dataset", img) for img in top_results]

    return render_template('result.html', query=os.path.join("static/uploads", file.filename), results=results)

if __name__ == '__main__':
    app.run(debug=True)
