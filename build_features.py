# build_features.py
import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

DATASET_PATH = "static/dataset"
os.makedirs(DATASET_PATH, exist_ok=True)  # dataset folder तयार करतो

# Pretrained VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

filenames = []
features_list = []

for img_name in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, img_name)
    try:
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x).flatten()
        filenames.append(img_name)
        features_list.append(features)
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

features_array = np.array(features_list)

with open("features.pkl", "wb") as f:
    pickle.dump((filenames, features_array), f)

print("✅ Features saved successfully!")
