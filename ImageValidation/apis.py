import os 
import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)
model = load_model('modelValidateFishImage.h5')

def download_image(url, output_file):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except:
        pass
    return

@app.route('/validate', methods=['POST'])
def validate_image():
    if 'photo_url' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    url = request.files.get('photo_url')
    if not url:
        return
    download_image(url, 'current_img.jpg')
    img = load_img('current_img.jpg', target_size=(160, 160))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images = np.vstack([img_array])
    prediction = model.predict(images, batch_size=32)
    label = 'Fish' if prediction < 0.4 else 'Not a fish'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))