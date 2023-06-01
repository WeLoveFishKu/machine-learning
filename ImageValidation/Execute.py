from ValidateFishImage import validatefishimageFunctions

validatefishimage_functions = validatefishimageFunctions()

# Load the data
train_set, validation_set = validatefishimage_functions.load_data()

# Develop models
learning_rate = 0.001
epochs = 10
model = validatefishimage_functions.create_model()
model = validatefishimage_functions.model_fitting(model, learning_rate, epochs, train_set, validation_set)

# Create prediction
ikan1 = '/Users/ariabagus/Desktop/Coding_Duniawi/Code/machine-learning/ImageValidation/TestImages/ikan1.jpeg'
ikan2 = '/Users/ariabagus/Desktop/Coding_Duniawi/Code/machine-learning/ImageValidation/TestImages/ikan2.jpeg'
bukanikan1 = '/Users/ariabagus/Desktop/Coding_Duniawi/Code/machine-learning/ImageValidation/TestImages/bukanikan1.png'

print(validatefishimage_functions.validate_fish_image(model, ikan1))
print(validatefishimage_functions.validate_fish_image(model, ikan2))
print(validatefishimage_functions.validate_fish_image(model, bukanikan1))
