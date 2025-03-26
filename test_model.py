import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import json

# Configuration
TEST_DIR = "./dataset/test"
MODEL_PATH = "./lung_cancer_model.keras"
BATCH_SIZE = 32 
CLASS_INDICES_PATH = "./class_indices.json"
SAMPLE_PER_CLASS = 2  # Number of sample images to show per class

def load_test_data():
    """Load test dataset with ImageDataGenerator"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

def evaluate_model(model, test_dataset):
    """Evaluate model accuracy on test set"""
    loss, accuracy = model.evaluate(test_dataset)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

def display_predictions(model, class_indices):
    """Display sample predictions with images"""
    reverse_class_indices = {v: k for k, v in class_indices.items()}
    
    plt.figure(figsize=(15, 10))
    plot_index = 1
    
    # Iterate through each class folder
    for class_name in os.listdir(TEST_DIR):
        class_path = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Get sample images
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_images = images[:SAMPLE_PER_CLASS]
        
        for img_name in sample_images:
            img_path = os.path.join(class_path, img_name)
            
            # Preprocess image
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            pred = model.predict(img_array)
            pred_index = np.argmax(pred)
            confidence = np.max(pred) * 100
            predicted_class = reverse_class_indices[pred_index]
            
            # Plot results
            plt.subplot(len(class_indices), SAMPLE_PER_CLASS, plot_index)
            plt.imshow(img)
            plt.title(f"True: {class_name}\nPred: {predicted_class}\nConf: {confidence:.1f}%")
            plt.axis('off')
            plot_index += 1

    plt.tight_layout()
    plt.show()

def main():
    # Load model and class indices
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Load test data
    test_dataset = load_test_data()
    
    # Evaluate model
    evaluate_model(model, test_dataset)
    
    # Display sample predictions
    display_predictions(model, class_indices)

if __name__ == "__main__":
    main()