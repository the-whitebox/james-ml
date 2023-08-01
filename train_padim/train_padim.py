from flask import Flask, render_template, request
import subprocess
import requests

app = Flask(__name__)

@app.route('/train_padim', methods=['POST'])
def train_padim():
    
    run_training_padim = 'python train.py --config padim_new.yaml'
    subprocess.run(run_training_padim, shell=True)

    return {"success": True}

@app.route('/test_trained_padim1',methods=['POST'])
def test_trained_padim1():
    test_trained_model = 'python gradio_inference.py \
            --weights ./results/padim/mydataset/run/weights/onnx/model.onnx \
            --metadata ./results/padim/mydataset/run/weights/onnx/metadata.json \
            --share True'
    subprocess.run(test_trained_model, shell=True)
    return {"success": True}

    # image_path = "Bad.jpg"  
    # onnx_model_path = "./results/padim/mydataset/run/weights/onnx/model.onnx"  
    # metadata_path = "./results/padim/mydataset/run/weights/onnx/metadata.json"  

    # data = {
    #     'image_path': image_path,
    #     'onnx_model_path': onnx_model_path,
    #     'metadata_path': metadata_path
    # }

    # response = requests.post("http://localhost:5010/detect-padim", data=data)

    # if response.status_code == 200:
    #     # The response content will contain the processed image data
    #     with open('processed_image.jpg', 'wb') as f:
    #         f.write(response.content)
    #     print("Image processed and saved as processed_image.jpg")
    # else:
    #     print("Error occurred during inference.")

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=5009)