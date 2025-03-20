import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

def load_model(model_path='lung_cancer_model.h5'):
    return tf.keras.models.load_model(model_path)

def load_class_indices(json_path='class_indices.json'):
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # {0: 'adenocarcinoma', ...}

def predict_image(img_path, model, class_indices):
    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    
    return {
        'class': class_indices[predicted_index],
        'confidence': float(np.max(predictions) * 100)
    }