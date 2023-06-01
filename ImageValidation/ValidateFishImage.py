import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

class validatefishimageFunctions:
    def __init__(self):
        self.batch_size = 32
        self.image_size = (160,160)
    
    def load_data(self, directory='Images/'):
        """Load the data into training and validation data from the designated folder path
        Args:
        directory (string) - the folder directory path that contains the data

        Returns:
        train_set - contain images used to train the model
        validation_set - contain images used for validation while training the model
        """
        train_set = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=self.batch_size,
                                             image_size=self.image_size,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
        validation_set = image_dataset_from_directory(directory,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
                                                    image_size=self.image_size,
                                                    validation_split=0.2,
                                                    subset='validation',
                                                    seed=42)
        
        return (train_set, validation_set)
        
    def create_model(self):
        """Generates the computer vision model using pre-trained model MobileNetV2, then uses data augmentation and additional layers for the model
        Returns:
        model (TF Keras Model) - the generated computer vision model
        """
        image_shape = self.image_size + (3,)
        pre_trained_model = MobileNetV2(input_shape=image_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        preprocess_input = mobilenet_v2.preprocess_input
        pre_trained_model.trainable = False
        inputs = tf.keras.Input(shape=image_shape)
        x = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),
                                tf.keras.layers.RandomRotation(0.2)])(inputs)
        x = preprocess_input(inputs)
        x = pre_trained_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)

        return model
    
    def model_fitting(self, model, learning_rate, epochs, train_set, validation_set):
        """Fits the model using the training data
        Args:
        model (TF Keras Model) - the generated computer vision model
        learning_rate (float) - the learning rate used for fitting the model
        epochs (int) - the number of epoch used to train the model
        train_set - contain images used to train the model
        validation_set - contain images used for validation while training the model

        Returns:
        model (TF Keras Model) - the fitted computer vision model
        """
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
        _ = model.fit(train_set, validation_data=validation_set, epochs=epochs)

        return model
    
    def validate_fish_image(self, model, image_path):
        """Create predictions for whether the picture contains fish or not
        Args:
        model (TF Keras Model) - the fitted computer vision model
        image_path (str) - file path to the image

        Returns:
        predictions (str) - the predicted label of an image
        """
        img = load_img(image_path, target_size=self.image_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        prob = model.predict(images, batch_size=self.batch_size)
        predictions = 'Fish' if prob < 0.4 else 'Not a fish'
        
        return predictions