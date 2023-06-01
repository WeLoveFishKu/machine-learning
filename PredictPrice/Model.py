import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ModelFunctions:
    def __init__(self):
        pass

    def plot_series(self, x, y, format="-", start=0, end=None, title=None, xlabel=None, ylabel=None, legend=None ):
        """ Visualizes time series data
        Args:
        x (array of int) - contains values for the x-axis
        y (array of int or tuple of arrays) - contains the values for the y-axis
        format (string) - line style when plotting the graph
        start (int) - first time step to plot
        end (int) - last time step to plot
        title (string) - title of the plot
        xlabel (string) - label for the x-axis
        ylabel (string) - label for the y-axis
        legend (list of strings) - legend for the plot
        """
        plt.figure(figsize=(10, 6))
        if type(y) is tuple:
            for y_curr in y:
                plt.plot(x[start:end], y_curr[start:end], format)
            else:
                plt.plot(x[start:end], y[start:end], format)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend(legend)
        plt.title(title)
        plt.grid(True)
        plt.show()

        return
    
    def windowed_dataset(self, series, window_size, batch_size, shuffle_buffer):
        """Generates dataset windows
        Args:
        series (array of float) - contains the values of the time series
        window_size (int) - the number of time steps to include in the feature
        batch_size (int) - the batch size
        shuffle_buffer(int) - buffer size to use for the shuffle method

        Returns:
        dataset (TF Dataset) - TF Dataset containing time windows
        """
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size).prefetch(1)
        
        return dataset


    def prepare_dataset(self, target_data, feature_data, split_index):
        """Slices the target and feature data to create training and testing data for both the target and feature data
        Args:
        target_data (array of float) - contains the target data used for training
        feature_data (numpy array) - contains the features data
        split_index (int) - the number of data contained in the training data

        Returns:
        test_feature (array of float) - contains the first split_index target data
        train_feature (numpy array) - contains the first split_index target data
        test_target (array of float) - contains the final split_index target data
        test_feature (numpy array) - contains the final split_index target data
        """
        train_target = target_data[:split_index]
        train_feature = feature_data[:split_index]
        test_target = target_data[split_index:]
        test_feature = feature_data[split_index:]

        return (train_target, train_feature, test_target, test_feature)
    
    def create_model(self, window_size):
        """Generates the model using recurrenct neural network
        Args:
        window_size (int) - the number of data contained within a list used for fitting

        Returns:
        model (TF Keras Model) - the generated reccurrent neural network
        """
        model = tf.keras.models.Sequential([
                tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                    input_shape=[window_size]),
                tf.keras.layers.SimpleRNN(40, return_sequences=True),
                tf.keras.layers.SimpleRNN(40),
                tf.keras.layers.Dense(8),
                tf.keras.layers.Dense(1)
                ])
        model.summary()

        return model
    
    def find_best_learningrate(self, model, train_target):
        """Used to find the best learningrate for the generated recurrenct neural network
        Args:
        model (TF Keras Model) - the generated reccurrent neural network
        train_target (array of float) - contains the data used for training the model
        """
        init_weights = model.get_weights()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10**(epoch / 20))
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)
        history = model.fit(train_target, epochs=100, callbacks=[lr_schedule])
        lrs = 1e-8 * (10 ** (np.arange(100) / 20))
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.semilogx(lrs, history.history["loss"])
        plt.tick_params('both', length=10, width=1, which='both')
        plt.axis([1e-8, 1e-3, 0, 100])
        plt.show()
        tf.keras.backend.clear_session()
        model.set_weights(init_weights)

        return 

    def model_fitting(self, model, learning_rate, epochs, train_target):
        """Fits the model using the training data
        Args:
        model (TF Keras Model) - the generated reccurrent neural network
        learning_rate (float) - the learning rate used for fitting the model
        epochs (int) - the number of epoch used to train the model
        train_target (array of float) - contains the data used for training the model

        Returns:
        model (TF Keras Model) - the fitted model
        """
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mae"])
        _ = model.fit(train_target,epochs=epochs)

        return model

    def develop_model(self, train_traget, window_size, batch_size, shuffle_buffer_size, learning_rate, epochs):
        """Create and fit the model using training data
        Args:
        train_traget (array of float) - contains the data used for training the model
        window_size (int) - the number of data contained within a list used for fitting
        batch_size (int) - the batch size
        shuffle_buffer (int) - buffer size to use for the shuffle method
        learning_rate (float) - the learning rate used for fitting the model
        epochs (int) - the number of epoch used to train the model

        Returns:
        model (TF Keras Model) - the fitted model
        """
        train_dataset = self.windowed_dataset(train_traget, window_size, batch_size, shuffle_buffer_size)
        model = self.create_model(window_size)
        model = self.model_fitting(model, learning_rate, epochs, train_dataset)
   
        return model

    def model_forecast(self, model, series, window_size, batch_size, scaler):
        """Uses an input model to generate predictions on data windows
        Args:
        model (TF Keras Model) - model that accepts data windows
        series (array of float) - contains the values of the time series
        window_size (int) - the number of time steps to include in the window
        batch_size (int) - the batch size

        Returns:
        forecast (numpy array) - array containing predictions
        """
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(window_size))
        dataset = dataset.batch(batch_size).prefetch(1)
        forecast = model.predict(dataset)
        forecast = scaler.inverse_transform(forecast)
        
        return forecast