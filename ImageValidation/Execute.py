from ValidateFishImage import validatefishimageFunctions

validatefishimage_functions = validatefishimageFunctions()

# Load the data
train_set, validation_set = validatefishimage_functions.load_data()

# Develop models
learning_rate = 0.001
epochs = 20
model = validatefishimage_functions.create_model()
model = validatefishimage_functions.model_fitting(model, learning_rate, epochs, train_set, validation_set)

model.save('modelValidateFishImage.h5')