import torchvision.models as models
import torch
import warnings

# Flask and server request
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image

warnings.simplefilter(action ='ignore')

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode



app = Flask(__name__)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/', methods=['GET'])
def  home():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(file.stream)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Convert the prediction index to a class label
    label = predicted.item()
    return jsonify({'prediction': label})

if __name__ == "__main__":
    app.run(debug=True)

