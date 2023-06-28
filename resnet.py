import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the ResNet model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_resnet(image_path):
    # Load and preprocess the input image
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class probabilities or labels
    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    # Create a result image (optional)
    # You can use libraries like OpenCV or PIL to create visualizations or overlays
    result_image = image.copy()  # Replace with your desired processing or visualization

    # Save the result image
    result_image_path = 'path_to_result_image.jpg'  # Replace with the actual result image path
    result_image.save(result_image_path)

    # Return the result image path
    return result_image_path
