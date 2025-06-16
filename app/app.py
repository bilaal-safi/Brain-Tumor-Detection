from flask import Flask, render_template, request
import os
from model import image_pre, predict

app = Flask(__name__)

# or you can write the whole path till static folder in the directory: UPLOAD_FOLDER = 'your whole path/Brain-tumor-classification/app/static''
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'There is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        file1.save(path)
        data = image_pre(path)  # Returns PyTorch tensor
        s = predict(data)       # Returns 0 or 1
        if s == 1:
            result = 'Healthy'
        else:
            result = 'Brain Cancer'
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
