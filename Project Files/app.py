from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('healthy_vs_rotten.keras')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def output():
    if request.method == 'POST':
 
        if 'pc_image' not in request.files:
            return redirect(request.url)
        
        f = request.files['pc_image']
        
        if f.filename == '':
            return redirect(request.url)
        
      
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
            
        img_path = "static/uploads/" + f.filename
        f.save(img_path)
        
        img = load_img(img_path, target_size=(224,224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize if your model expects this
        
 
        pred = np.argmax(model.predict(img_array), axis=1)
        
        classes = [
            'Apple__Healthy (0)', 'Apple__Rotten (1)', 
            'Banana__Healthy (2)', 'Banana__Rotten (3)',
            'Bellpepper__Healthy (4)', 'Bellpepper__Rotten (5)',
            'Carrot__Healthy (6)', 'Carrot__Rotten (7)',
            'Cucumber__Healthy (8)', 'Cucumber__Rotten (9)',
            'Grape__Healthy (10)', 'Grape__Rotten (11)',
            'Jujube__Healthy (14)', 'Jujube__Rotten (15)',
            'Mango__Healthy (16)', 'Mango__Rotten (17)',
            'Orange__Healthy (18)', 'Orange_Rotten (19)',
            'Pomegranate__Healthy (20)', 'Pomegranate__Rotten (21)',
            'Potato__Healthy (22)', 'Potato__Rotten (23)',
            'Strawberry__Healthy (24)', 'Strawberry__Rotten (25)',
            'Tomato__Healthy (26)', 'Tomato__Rotten (27)'
        ]
        
        prediction = classes[int(pred)]
        print("Prediction:", prediction)
        
        return render_template("portfolio-details.html", 
                             predict=prediction,
                             image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True, port=2222)