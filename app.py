from flask import Flask, render_template, request
import subprocess
import os
import stat
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import sys
import requests
app = Flask(__name__, static_url_path='', static_folder='static')

def delete_files_in_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Check if the directory is empty
    if len(os.listdir(directory)) == 0:
        print(f"Directory '{directory}' is already empty.")
        return

    # Delete the files and directories in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"All files in directory '{directory}' have been deleted.")



def run_keras_resnet(image_path):
    # Import the necessary dependencies
    venv_path = os.path.join(os.getcwd(), 'resnetvenv')
    if not os.path.exists(venv_path):
        subprocess.run(['python', '-m', 'venv', venv_path])

    # Activate the virtual environment
    activate_script = os.path.join(venv_path, 'Scripts', 'activate')
    activate_cmd = f'"{activate_script}" &&'

    install_keras_cmd = 'pip install keras'
    subprocess.run(install_keras_cmd, shell=True)

    install_tf_cmd = 'pip install tensorflow'
    subprocess.run(install_tf_cmd, shell=True)


    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    import numpy as np

    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Perform the prediction using the pre-trained model
    preds = model.predict(x)

    # Decode the predictions
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Create a PIL image object
    image_pil = Image.open(image_path)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("arial.ttf", 16)

    # Print the results on the image
    y = 10
    for class_id, class_name, prob in decoded_preds:
        result_text = f"{class_name} ({prob:.2%})"
        draw.text((10, y), result_text, font=font, fill=(255, 255, 255))
        y += 20

    # Save the result image
    result_image_path = os.path.join(os.getcwd(), 'resnet-results', 'result_image.jpeg')
    os.makedirs(result_image_path, exist_ok=True)
    image_pil.save(result_image_path)

    # Return the result image path
    return result_image_path

@app.route('/')
def home():
    # image_path = '.\\static\\background.jpg'
    return render_template('index.html')


@app.route('/launch-label-studio')
def launch_label_studio():
    
    try:
        detect_resnet = requests.post("http://localhost:5023/launch_label_studio")
        return "Label Studio is Launching!"
    except Exception as e:
        raise ValueError(e)


@app.route('/yolov5')
def yolov5():
    # Delete the uploaded image in the static directory
    directory_path = os.path.join(os.getcwd(), 'static', 'yolov5')
    delete_files_in_directory(directory_path)
    return render_template('yolov5.html')


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':

        venv_path = os.path.join(os.getcwd(), 'yolovenv')
        if not os.path.exists(venv_path):
            subprocess.run(['python', '-m', 'venv', venv_path])

        # Activate the virtual environment
        activate_script = os.path.join(venv_path, 'Scripts', 'activate')
        activate_cmd = f'"{activate_script}" &&'

        os.makedirs(os.path.join(os.getcwd(), 'static/yolov5/original'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'static/yolov5/processed'), exist_ok=True)

        # Save the uploaded image
        image_file = request.files['image']
        image_path = os.path.join(os.path.join(os.getcwd(), 'static\\yolov5\\original'), 'image.jpeg')
        # os.makedirs('uploaded', exist_ok=True)
        image_file.save(image_path)

        # Change directory to yolov5
        yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
        os.chdir(yolov5_dir)

        # Install requirements
        subprocess.run([f'{activate_cmd} pip install -r requirements.txt'], shell=True)


        # Run YOLOv5 detection
        yolo_cmd = f'{activate_cmd} python detect.py --weights yolov5s.pt --source "{image_path}"'
        subprocess.run(yolo_cmd, shell=True)

        # Find the latest folder in runs/detect
        detect_dir = os.path.join(yolov5_dir, 'runs', 'detect')
        
        newest = max([f for f in os.listdir(detect_dir)], key=lambda x: os.stat(os.path.join(detect_dir, x)).st_mtime)

        result_image_path = os.path.join(os.getcwd(),'runs\detect', newest, 'image.jpeg')

        # Move the result image to a separate folder
        os.chdir('..')
        result_image_dir = os.path.join(os.getcwd(), 'static', 'yolov5', 'processed')
        os.makedirs(result_image_dir, exist_ok=True)

        shutil.copy(result_image_path, result_image_dir)

        # Deactivate the virtual environment
        subprocess.run([f'{activate_cmd} deactivate'], shell=True)

        # Change back to the original directory
        os.chdir(os.getcwd())
        # print("----------------------------------",result_image_path)
        # Display the result image
        return render_template('yolov5.html', image_path=result_image_path)

# from resnet import resnet_test
@app.route('/resnet')
def resnet():
    directory_path = os.path.join(os.getcwd(), 'static', 'resnet')
    delete_files_in_directory(directory_path)
    return render_template('resnet.html')

@app.route('/detect-resnet', methods=['POST'])
def detect_resnet():
    if request.method == 'POST':

        os.makedirs(os.path.join(os.getcwd(), 'static/resnet/original'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'static/resnet/processed'), exist_ok=True)

        # Save the uploaded image
        image_file = request.files['image']
        image_path = os.path.join(os.path.join(os.getcwd(), 'static\\resnet\\original'), 'image.jpg')
        image_file.save(image_path)

        print("----------------(3)--------------Running Inference-----------------------------")
        try:
            detect_resnet = requests.post("http://localhost:5003/detect-resnet", data={'image_path': image_path})
            result_text = detect_resnet.text
            print(result_text)
        except Exception as e:
            raise ValueError(e)

        return render_template('resnet.html', result_text=result_text)

@app.route('/padim')
def padim():
    directory_path = os.path.join(os.getcwd(), 'static' , 'padim')
    delete_files_in_directory(directory_path)
    return render_template('padim.html')

@app.route('/detect-padim', methods=['POST'])
def detect_padim():
    if request.method == 'POST':

        os.makedirs(os.path.join(os.getcwd(), 'static/padim/original'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'static/padim/processed'), exist_ok=True)

        # Save the uploaded image
        image_file = request.files['image']
        image_path = os.path.join(os.path.join(os.getcwd(), 'static\\padim\\original'), 'image.png')
        
        image_file.save(image_path)

        print("----------------(3)--------------Running Inference-----------------------------")
        try:
            detect_padim = requests.post("http://localhost:5012/detect-padim", data={'image_path': image_path})

        except Exception as e:
            raise ValueError(e)

        print("----------------(4)--------------Done-----------------------------")

        result_image_path = os.path.join(os.getcwd(),'padim', 'resultImg','original', 'image.png')

        print('Result image', result_image_path)
        # Move the result image to a separate folder
        
        result_image_dir = os.path.join(os.getcwd(), 'static', 'padim','processed')
        os.makedirs(result_image_dir, exist_ok=True)    

        shutil.copy(result_image_path, result_image_dir)

        # Display the result image
        return render_template('padim.html', image_path=result_image_path)

@app.route('/resnet_training')
def launch_train_resnet():
    return render_template('trainresnet.html')

@app.route('/yolov5_training')
def launch_train_yolov5():
    return render_template('trainyolo.html')

@app.route('/padim_training')
def launch_train_padim():
    return render_template('trainpadim.html')

@app.route('/train_yolo', methods=['POST'])
def train_yolo():
    if request.method == 'POST':
        try:
            train_yolo = requests.post("http://localhost:5005/train_yolo")
        except Exception as e:
            raise ValueError(e)
        return render_template('trainyolo.html')

@app.route('/train_padim', methods=['POST'])
def train_padim():
    if request.method == 'POST':
        try:
            train_yolo = requests.post("http://localhost:5009/train_padim")
        except Exception as e:
            raise ValueError(e)
        return render_template('trainpadim.html')

# 
@app.route('/test_trained_padim', methods=['POST'])
def test_trained_padim():
    if request.method == 'POST':
        try:
            train_yolo = requests.post("http://localhost:5009/test_trained_padim1")
        except Exception as e:
            raise ValueError(e)
        return render_template('trainpadim.html')

if __name__ == '__main__':
    app.run(debug=True)
