import numpy as np

from AnomalyDetection.DetectionModel.CreateModel import CreateModel

# Creating the Model
model = CreateModel(input_size=24576, hidden_size=1024)

# Loading the model
model.load('path_to_model.pth')

# Defining test data
test_data = np.load('path_to_test_data.npy')

print(test_data.shape)

# Predicting the test data
model.predict(test_data)
