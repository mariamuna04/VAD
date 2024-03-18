"""
This script creates a model, preprocesses the data, and trains the model. The model is then evaluated on the test data.
Use this as an example to train the model on your own data.

NOTE:
    Run the training code with hardware acceleration (GPU) to speed up the training process. Without GPU, the script
    may crash due to memory issues.
"""

# Importing the required libraries
from AnomalyDetection.DetectionModel.Preprocess import preprocess_data, create_dataloader
from AnomalyDetection.DetectionModel.CreateModel import CreateModel, load_params

# Preprocess the data
# train_data and test_data are of type TensorDataset
train_data, test_data = preprocess_data(
    '../dataset/embeddings.npy',
    '../dataset/labels.npy'
)

# Create the PyTorch DataLoader
# train_loader and test_loader are of type DataLoader
train_loader, test_loader = create_dataloader(train_data, test_data, batch_size=8)

"""
CreateModel is a custom class that creates a model with a specified input size and hidden size. You can specify various
parameters such as the number of layers, dropout, bidirectional, etc. To do that, make a dictionary of the parameters
and pass it to the CreateModel class. The default parameters are as follows:
    default_params = {
        'num_layers': 2,
        'dropout': 0.0,
        'bidirectional': False,
        'layer_norm': False,
        'highway_bias': 0.0,
        'has_skip_term': True,
        'rescale': True,
        'nn_rnn_compatible_return': False,
        'proj_input_to_hidden_first': False,
        'amp_recurrence_fp16': False,
        'normalize_after': False,
    }
"""
# Load the default parameters, you can skip this
# default_params = load_params('parameters.json')

# Create the model
model = CreateModel(input_size=24576, hidden_size=1024, model_name='MyCustomModel')

# Train the model
# You can specify the number of epochs, criterion, optimizer, and log_report
model.fit(train_loader, epochs=10)

# Evaluate the model
model.evaluate(test_loader)

# Save the model
model.save('model.pth')
