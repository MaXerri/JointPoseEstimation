import io 
from PIL import Image
from flask import Flask, render_template, request, send_file, redirect, url_for
from network import run_model
from io import BytesIO
import numpy as np
import torch
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No File Passed"

    file = request.files['image']

    if file:
        img = io.BytesIO(file.read())

        # model is set in evaluation mode earlier 
        with torch.no_grad():
            f, ax = run_model(img) # output image generated in network.py

        # save the plot to a BytesI0 object (in memory buffer)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        #send the image as as a response:
        #return send_file(buf, mimetype='image/png')
        return render_template('index.html', img_data=encoded_img_data)
    
    

if __name__ == '__main__':
    app.run(debug=True) # need to change IP eventually 

