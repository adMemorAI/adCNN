import torch
from PIL import Image
from model import SimpleCNN
from config import device, transform
import gradio as gr
from huggingface_hub import hf_hub_download

# -----------------------------
# 1. Load the Model
# -----------------------------

# Download the model file from the Hugging Face Model Hub
model_dir = hf_hub_download(repo_id="diejor/adCNN-simple", filename="adCNN.pth")

model = SimpleCNN()
model.load_state_dict(torch.load(model_dir, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Define the Prediction Function
# -----------------------------

def predict_dementia(image):
    try:
        # Apply preprocessing
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Move the input and model to device (CPU or GPU)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, dim=0)

        # Define class labels
        class_labels = ['No Dementia', 'Dementia']

        return f"{class_labels[predicted_class]} (Confidence: {confidence.item() * 100:.2f}%)"

    except Exception as e:
        return f"Error in processing image: {e}"

# -----------------------------
# 3. Create the Gradio Interface
# -----------------------------

iface = gr.Interface(
    fn=predict_dementia,  # The prediction function
    inputs=gr.inputs.Image(type="pil", label="Upload MRI Slice (PNG)"),  # Input component
    outputs=gr.outputs.Textbox(label="Prediction"),  # Output component
    title="Dementia Detection from MRI Slices",
    description="Upload an MRI slice as a PNG image, and the model will predict whether it shows signs of dementia.",
    examples=[
        # Add paths to example images if available
    ]
)

# -----------------------------
# 4. Launch the Interface
# -----------------------------

if __name__ == "__main__":
    iface.launch()
