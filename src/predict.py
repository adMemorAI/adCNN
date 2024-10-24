# predict.py

import torch
from torchvision import transforms
from PIL import Image
import argparse
import subprocess
import sys
import logging
import wandb
import os

from config import Config
from models.model_factory import get_default_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

logger = logging.getLogger(__name__)

def get_git_commit():
    """Retrieve the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        logger.debug(f"Git commit hash: {commit_hash}")
        return commit_hash
    except Exception as e:
        logger.warning(f"Could not retrieve git commit hash: {e}")
        return "unknown"

def load_image(image_path, transform):
    """Load and preprocess the image."""
    try:
        logger.info(f"Loading image from {image_path}")
        image = Image.open(image_path).convert('L')  # Convert to grayscale if needed
        image = transform(image).unsqueeze(0)  # Add batch dimension
        logger.debug(f"Preprocessed image tensor shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None  # Return None to handle errors gracefully

def load_model(model_path, device):
    """Load the trained model."""
    try:
        logger.info(f"Loading model from {model_path}")
        model = get_default_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

def make_prediction(model, image, device):
    """Perform inference and make a prediction."""
    try:
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities >= 0.5).float()
        logger.debug(f"Raw Outputs: {outputs}")
        logger.debug(f"Probabilities: {probabilities}")
        logger.debug(f"Predictions: {preds}")
        return preds.item(), probabilities.item()
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None, None  # Return None values to handle errors gracefully

def main():
    parser = argparse.ArgumentParser(description='ADCNN Prediction Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--image_dir', type=str, required=False, help='Path to the directory containing images')
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="ResAD",
        entity="adCNN",
        config={
            "model_path": args.model_path,
            "image_dir": args.image_dir,
            "git_commit": get_git_commit(),
        },
        name=f"Prediction-{wandb.util.generate_id()}",
        tags=["prediction", "ResAD"],
    )

    # Initialize configuration
    config = Config()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define image transformations (ensure consistency with training)
    transform = config.transform

    # Determine the image directory
    if args.image_dir:
        image_dir = args.image_dir
    else:
        # Use default images directory in project_dir
        image_dir = os.path.join(config.project_root, 'images')

    # Ensure the image directory exists
    if not os.path.isdir(image_dir):
        logger.error(f"Image directory does not exist: {image_dir}")
        sys.exit(1)

    # Load the model once
    model = load_model(args.model_path, device)

    # Prepare the wandb Table
    columns = ["id", "image", "predicted_label", "probability"]
    prediction_table = wandb.Table(columns=columns)

    # Define class labels
    classes = ["Non-Dementia", "Dementia"]

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    if not image_files:
        logger.warning(f"No images found in directory: {image_dir}")
        sys.exit(0)

    # Process each image
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        # Load and preprocess the image
        image = load_image(img_path, transform)
        if image is None:
            continue  # Skip if image failed to load

        # Make prediction
        prediction, probability = make_prediction(model, image, device)
        if prediction is None:
            continue  # Skip if prediction failed

        predicted_class = classes[int(round(prediction))]

        # Log prediction with confidence
        logger.info(f"Image: {img_file} - Prediction: {predicted_class} with probability {probability:.4f}")

        # Add data to the wandb Table
        prediction_table.add_data(
            img_file,                       # Image ID
            wandb.Image(img_path),          # Image
            predicted_class,                # Predicted label
            probability                     # Confidence probability
        )

    # Log the Table to wandb
    wandb.log({"prediction_table": prediction_table})
    wandb.finish()

if __name__ == "__main__":
    main()
