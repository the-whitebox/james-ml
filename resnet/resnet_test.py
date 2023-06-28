import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='', static_folder='../static')


@app.route('/detect-resnet', methods=['POST'])
# def detect_resnet():
#     image_path = request.form.get('image_path')

#     # Load the pre-trained ResNet50 model
#     model = ResNet50(weights='imagenet')

#     # Load and preprocess the input image
#     img = image.load_img(image_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     # Perform the prediction using the pre-trained model
#     preds = model.predict(x)

#     # Decode the predictions
#     decoded_preds = decode_predictions(preds, top=3)[0]

#     # Create a PIL image object
#     image_pil = Image.open(image_path)
#     draw = ImageDraw.Draw(image_pil)
#     font = ImageFont.truetype("arial.ttf", 16)

#     # Print the results on the image
#     y = 10
#     for class_id, class_name, prob in decoded_preds:
#         result_text = f"{class_name} ({prob:.2%})"
#         draw.text((10, y), result_text, font=font, fill=(255, 255, 255))
#         y += 20

#     # Display the result image
#     plt.imshow(image_pil)
#     plt.axis('off')
#     plt.show()

#     # Save the result image
#     os.makedirs('resnet-results', exist_ok=True)
#     result_image_path = os.path.join(os.getcwd(), 'resnet-results', 'image.jpg')
#     image_pil.save(result_image_path)

#     # Return the result image path
#     return result_image_path
def detect_resnet():
    model = ResNet50(weights='imagenet')
    img_path = request.form.get('image_path')
    # img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    decoded_preds = decode_predictions(preds, top=3)[0]
    print('Predicted:', decode_predictions(preds, top=3)[0])
    
    image_pil = Image.open(img_path)
    draw = ImageDraw.Draw(image_pil)

    # Set the font and font size
    font = ImageFont.truetype("arial.ttf", 72)

    # Get the predicted label and probability for the most probable result
    label = decoded_preds[0][1]
    probability = decoded_preds[0][2]

    # Draw the label and probability on the image
    text = f"{label} ({probability:.2%})"
    text_position = (10, 10)
    text_color = (255, 0, 0)
    rectangle_color = (255, 0, 0)
    rectangle_thickness = 2
    text_width, text_height = draw.textsize(text, font=font)
    rectangle_coords = [
        text_position[0],
        text_position[1],
        text_position[0] + text_width + 4,
        text_position[1] + text_height + 4,
    ]
    draw.rectangle(rectangle_coords, outline=rectangle_color, width=rectangle_thickness)

# Draw the label and probability on the image
    draw.text(text_position, text, font=font, fill=text_color)

    # Save the modified image
    result_image_path = 'result_image.jpg'
    image_pil.save(result_image_path)

    # Display the modified image
    # image_pil.show()
    return text

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5003)
