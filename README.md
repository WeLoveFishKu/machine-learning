# machine-learning

There are two directories in this repository that correspond to the machine learning models we have created.

In the "PredictPrice" folder, there is a program that retrieves data from the Badan Pangan Nasional, cleans the data, builds the machine learning models, and generates fish price forecasts. The model used is a recurrent neural network, which has a final forecast accuracy of 12 percent. This means that, on average, the forecasts are within a range of 12 percent higher or lower than the actual price.

The "ImageValidation" folder contains a program that collects approximately 300 images of fish and non-fish, develops the machine learning models, and saves the model in tflite format. The model is created using the transfer learning method, with MobileNetV2 as the base model. Additional layers are added for data augmentation, pooling, and dropouts. The predictions achieved by this model are remarkably accurate, with an accuracy rate exceeding 90 percent.
