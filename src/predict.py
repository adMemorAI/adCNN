import torch
from PIL import Image
import argparse
from models.resad import ResAD # Updated to match your trained model
from config import device, transform
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict Dementia from an image using a trained CNN model.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    return parser.parse_args()

def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        exit(1)
    
    model = ResAD().to(device)  # Updated to match your trained model class
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading the model state_dict: {e}")
        exit(1)
    
    model.eval()
    return model

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        # Ensure the image is in grayscale mode since your model expects 1-channel input
        image = image.convert('L')
    except Exception as e:
        print(f"Error opening image: {e}")
        exit(1)
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model, device):
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        class_probabilities = {
            'Non-Dementia': 1 - probability,
            'Dementia': probability
        }
        predicted_class = 'Dementia' if probability >= 0.5 else 'Non-Dementia'
        confidence = class_probabilities[predicted_class] * 100

    return predicted_class, class_probabilities, confidence

def main():
    args = parse_arguments()
    image_path = args.image_path

    # Verify the image path exists
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        exit(1)
    
    # Load the model
    model_path = "/home/diejor/projects/aimf24/adCNN/src/models/ResAD_DementiaDetection_20241012_175906/ResAD_best_epoch1_valLoss0.2724_f162.69.pth"  # Update with the correct model path
    model = load_model(model_path, device)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Make prediction
    predicted_class, class_probabilities, confidence = predict(image, model, device)
    
    # Display results
    print(f"\n=== Prediction Results ===")
    print(f"Image Path       : {image_path}")
    print(f"Predicted Class  : {predicted_class}")
    print(f"Confidence       : {confidence:.2f}%\n")
    print(f"Class Probabilities:")
    for cls, prob in class_probabilities.items():
        print(f"  - {cls}: {prob*100:.2f}%")
    print("==========================\n")
    
    # Optional: Visualize the image with prediction
    try:
        import matplotlib.pyplot as plt
        image_display = Image.open(image_path).convert('L')  # Load as grayscale for display
        plt.imshow(image_display, cmap='gray')
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()
    except ImportError:
        pass  # If matplotlib is not installed, skip visualization

if __name__ == '__main__':
    main()