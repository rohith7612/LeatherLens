from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


model = load_model('cnn.h5')


class_mapping = {
    0: 'Folding marks',
    1: 'Grain off',
    2: 'Growth marks',
    3: 'loose grains',
    4: 'non defective',
    5: 'pinhole'
}

IMG_SIZE = 128


def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    label = class_mapping[class_idx]
    return label


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            
            label = predict_image(filepath)
            return render_template('index.html', label=label, image=file.filename)
    
    return render_template('index.html', label=None, image=None)

if __name__ == '__main__':
    app.run(debug=True)
