from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/train_yolo', methods=['POST'])
def train_yolo():
    
    run_training_yolo = 'python train.py --img 640 --epochs 3 --data yolo_config.yaml --weights yolov5s.pt'
    subprocess.run(run_training_yolo, shell=True)

    return {"success": True}

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0')