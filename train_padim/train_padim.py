from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/train_padim', methods=['POST'])
def train_padim():
    
    run_training_padim = 'python train.py --config padim_new.yaml'
    subprocess.run(run_training_padim, shell=True)

    return {"success": True}

if __name__ == "__main__":

    app.run(debug=True, host='localhost', port=5009)