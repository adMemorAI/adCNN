import torch
from PIL import Image
import sys
from model import SimpleCNN
from config import device, transform

if len(sys.argv) < 2:
    print('Usage: python predict.py <image_path>')
    sys.exit(1)

model = SimpleCNN()
model.load_state_dict(torch.load('models/adCNN.pth'), map_location=device)
model.to(device)
model.eval()

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).float()
        class_index = int(prediction.item())
        classes = ['Non-Dementia', 'Dementia']
        return classes[class_index]

result = predict_image(image_path)
print(f'The provided brain is classified as: {result}')
