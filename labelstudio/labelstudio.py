from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/launch_label_studio', methods=['POST'])
def launch_label_studio():
    install_label_studio_cmd = 'pip install -U label-studio'
    subprocess.run(install_label_studio_cmd, shell=True)

    label_studio_cmd = 'label-studio'
    subprocess.Popen(label_studio_cmd, shell=True)

    return {"success": True}

if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0")