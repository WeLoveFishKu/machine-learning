import os 
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('modelValidateFishImage.h5')

@app.route('/validate', methods=['POST'])
def validate_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    
    image_file = request.files['image']
    img = image.load_img(image_file, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images = np.vstack([img_array])
    
    prediction = model.predict(images, batch_size=32)
    label = 'Fish' if prediction < 0.4 else 'Not a fish'
    
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))