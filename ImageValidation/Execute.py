import tensorflow as tf
from keras.models import load_model

from ValidateFishImage import validatefishimageFunctions

validatefishimage_functions = validatefishimageFunctions()

# Load the data
train_set, validation_set = validatefishimage_functions.load_data()

# Develop models
learning_rate = 0.001
epochs = 20
model = validatefishimage_functions.create_model()
model = validatefishimage_functions.model_fitting(model, learning_rate, epochs, train_set, validation_set)

# Saves the model in h5 format
model.save('modelValidateFishImage.h5')

# saves the model in tflite format
model = load_model('modelValidateFishImage.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
lite_model = converter.convert()
with open("lite_model.tflite", 'wb') as f:
    f.write(lite_model)