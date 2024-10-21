# predict.py

import torch
from PIL import Image
import argparse
from models.model_factory import get_model, extract_model_name  # Correct import
from configs.config import Config
from utils.logger import Logger
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict Dementia from an image using a trained CNN model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    return parser.parse_args()

def preprocess_image(image_path, transform):
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Error opening image: {e}") from e

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model, device, threshold=0.5):
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        class_probabilities = {
            'Non-Dementia': 1 - probability,
            'Dementia': probability
        }
        predicted_class = 'Dementia' if probability >= threshold else 'Non-Dementia'
        confidence = class_probabilities[predicted_class] * 100

    return predicted_class, class_probabilities, confidence

def main():
    args = parse_arguments()
    model_path = args.model_path
    image_path = args.image_path

    # Initialize Logger
    logger = Logger("predictor")
    
    # Initialize Config
    config = Config()

    # Load the model
    try:
        model = get_model(model_path, config.device)
        model_name = extract_model_name(model_path)
        logger.info(f"Model '{model_name}' loaded successfully from {model_path}.")
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Preprocess the image using model-specific transform
    try:
        transform = config.transform
        image = preprocess_image(image_path, transform)
        logger.info(f"Image '{image_path}' loaded and preprocessed successfully.")
        print(f"Preprocessed image tensor shape: {image.shape}")  # Debug statement
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Make prediction
    try:
        predicted_class, class_probabilities, confidence = predict(image, model, config.device)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        sys.exit(1)

    # Display results
    print(f"\n=== Prediction Results ===")
    print(f"Image Path        : {image_path}")
    print(f"Predicted Class   : {predicted_class}")
    print(f"Confidence        : {confidence:.2f}%\n")
    print(f"Class Probabilities:")
    for cls, prob in class_probabilities.items():
        print(f"  - {cls}: {prob*100:.2f}%")
    print("==========================\n")

    # Optional: Visualize the image with prediction
    try:
        import matplotlib.pyplot as plt
        image_display = Image.open(image_path)
        # Determine number of channels from model_params
        model_params = config.model_params.get(model_name, {})
        input_channels = model_params.get('input_channels', 3)
        if input_channels == 1:
            image_display = image_display.convert('L')
            cmap = 'gray'
        else:
            image_display = image_display.convert('RGB')
            cmap = None
        plt.imshow(image_display, cmap=cmap)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()
    except ImportError:
        logger.info("matplotlib is not installed. Skipping image visualization.")
    except Exception as e:
        logger.error(f"Error during image visualization: {e}")

if __name__ == '__main__':
    main()
