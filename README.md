This repository contains a simple web service for image classification using a pre-trained ResNet50 model. The service is built using Flask.
Image Transformation
The input image is pre-processed using the following transformations:

Resize to 256x256 pixels.
Crop the center to 224x224 pixels.
Convert to a PyTorch tensor.
Normalize the image with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].
The service utilizes a pre-trained ResNet50 model obtained from the torchvision library. The model is set to evaluation mode before serving predictions.
License:
This project is licensed under the MIT License.
