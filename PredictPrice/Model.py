import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class model:
    def __init__(self):
        pass

    def plot_series(self, x, y, format="-", start=0, end=None, 
                    title=None, xlabel=None, ylabel=None, legend=None ):
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
        plt.figure(figsize=(10, 6)) # Setup dimensions of the graph figure
        if type(y) is tuple: # Check if there are more than two series to plot
            for y_curr in y: # Loop over the y elements
                plt.plot(x[start:end], y_curr[start:end], format) # Plot the x and current y values
            else:
                plt.plot(x[start:end], y[start:end], format) # Plot the x and y values
        # Give labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Set the legend
        if legend:
            plt.legend(legend)
        plt.title(title) # Set the title
        plt.grid(True) # Overlay a grid on the graph
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
        dataset = tf.data.Dataset.from_tensor_slices(series) # Generate a TF Dataset from the series values
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) # Window the data but only take those with the specified size
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1)) # Flatten the windows by putting its elements in a single batch
        dataset = dataset.map(lambda window: (window[:-1], window[-1])) # Create tuples with features and labels 
        dataset = dataset.shuffle(shuffle_buffer) # Shuffle the windows
        dataset = dataset.batch(batch_size).prefetch(1) # Create batches of windows
        
        return dataset

    def model_forecast(self, model, series, window_size, batch_size):
        """Uses an input model to generate predictions on data windows
        Args:
        model (TF Keras Model) - model that accepts data windows
        series (array of float) - contains the values of the time series
        window_size (int) - the number of time steps to include in the window
        batch_size (int) - the batch size

        Returns:
        forecast (numpy array) - array containing predictions
        """
        dataset = tf.data.Dataset.from_tensor_slices(series) # Generate a TF Dataset from the series values
        dataset = dataset.window(window_size, shift=1, drop_remainder=True) # Window the data but only take those with the specified size
        dataset = dataset.flat_map(lambda w: w.batch(window_size)) # Flatten the windows by putting its elements in a single batch
        dataset = dataset.batch(batch_size).prefetch(1) # Create batches of windows
        forecast = model.predict(dataset) # Get predictions on the entire dataset
        
        return forecast

    def prepare_dataset(self, target_data, feature_data, split_index):
        """Slices the target and feature data to create training and testing data for both the target and feature data
        Args:
        target_data - contains the target data used for training
        feature_data - contains the features data
        split_index - the number of data contained in the training data

        Returns:
        test_feature - contains the first split_index target data
        train_feature - contains the first split_index target data
        test_target - contains the final split_index target data
        test_feature - contains the final split_index target data
        """
        # Get the train set 
        train_target = target_data[:split_index]
        train_feature = feature_data[:split_index]
        # Get the validation set
        test_target = target_data[split_index:]
        test_feature = feature_data[split_index:]

        return (train_target, train_feature, test_target, test_feature)
    
    def create_model(self, window_size):
        """Generates the model using recurrenct neural network
        Args:
        window_size - the number of data contained within a list used for fitting

        Returns:
        model - the generated reccurrent neural network
        """
        # Build the Model
        model = tf.keras.models.Sequential([
                tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                    input_shape=[window_size]),
                tf.keras.layers.SimpleRNN(40, return_sequences=True),
                tf.keras.layers.SimpleRNN(40),
                tf.keras.layers.Dense(8),
                tf.keras.layers.Dense(1)
                ])
        model.summary() # Print the model summary 

        return model
    
    def find_best_learningrate(self, model, train_target):
        """Used to find the best learningrate for the generated recurrenct neural network
        Args:
        model - the generated reccurrent neural network
        train_target - contains the data used for training the model
        """
        init_weights = model.get_weights() # Get initial weights
        # Set the learning rate scheduler
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10**(epoch / 20))
        optimizer = tf.keras.optimizers.SGD(momentum=0.9) # Initialize the optimizer
        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer) # Set the training parameters
        history = model.fit(train_target, epochs=100, callbacks=[lr_schedule]) # Train the model
        lrs = 1e-8 * (10 ** (np.arange(100) / 20)) # Define the learning rate array
        plt.figure(figsize=(10, 6)) # Set the figure size
        plt.grid(True) # Set the grid
        plt.semilogx(lrs, history.history["loss"]) # Plot the loss in log scale
        plt.tick_params('both', length=10, width=1, which='both') # Increase the tickmarks size
        # Set the plot boundaries
        plt.axis([1e-8, 1e-3, 0, 100])
        plt.show()
        tf.keras.backend.clear_session() # Reset states generated by Keras
        model.set_weights(init_weights) # Reset the weights

        return 

    def model_fitting(self, model, learning_rate, epochs, train_target):
        """Fits the model using the training data
        Args:
        model - the generated reccurrent neural network
        learning_rate - the learning rate used for fitting the model
        epochs - the number of epoch used to train the model
        train_target - contains the data used for training the model

        Returns:
        model - the fitted model
        """
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9) # Set the optimizer 
        # Set the training parameters
        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mae"])
        _ = model.fit(train_target,epochs=epochs) # Train the model

        return model

    def predict_using_training_data(self, model, train_dataset, data_train):
        predictions = model.predict(train_dataset)
        plt.plot(data_train)
        plt.plot(predictions)
        plt.legend(['Data', 'Prediction'])
        plt.show()

        return
    