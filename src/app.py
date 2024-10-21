import io 
from PIL import Image
from flask import Flask, render_template, request, send_file
from network import run_model
from io import BytesIO
import matplotlib.pyplot as plt

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
        img = Image.open(io.BytesIO(file.read()))

        fig, ax = run_model(img)

        # save the plot to a BytesI) object (in memory buffer)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        #send the image as as a response:
        return send_file(buf, mimetype='image/png')
    

if __name__ == '__main__':
    app.run(debug=True) # need to change IP eventually 

