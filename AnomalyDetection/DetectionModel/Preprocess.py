"""
This file contains the functions to preprocess the data for the model. The preprocess_data function takes the embeddings
and labels and returns the training and testing data. The create_dataloader function takes the training and testing data
and returns the PyTorch DataLoader.
"""

# Importing the required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from AnomalyDetection import device
import torch
from torch.utils.data import TensorDataset, DataLoader


# Function to preprocess the data (npy files)
def preprocess_data(embeddings: np.ndarray or str,
                    labels: np.ndarray or str,
                    test_size: float = 0.2):
    """
    This function preprocesses the data for the model and returns the training and testing data. It can take both
    variable and file path in parameters

    Parameters
    ----------
    embeddings: np.ndarray or str: The embeddings of the videos.
    It should be a 4D array [instances, windows, frames, features].
    labels: np.ndarray or str: The labels of the videos.
    test_size: float, optional: The size of the test data. Default is 0.2.
    """
    # Check if the parameters are file path. If so, load them with np.load()
    if isinstance(embeddings, str):
        embeddings = np.load(embeddings)
    if isinstance(labels, str):
        labels = np.load(labels)

    # Check if the embeddings and labels are of the same length
    if len(embeddings) != len(labels):
        raise ValueError("The length of the embeddings and labels should be the same")

    # check if the embedding is a 4D array
    if len(embeddings.shape) != 4:
        raise ValueError(f"The embeddings should be a 4D array [instances, windows, frames, features]."
                         f" Found {len(embeddings.shape)}D instead.")

    # Change the shape to fit the model (into 2d Array)
    """
    Why Reshape?
    This is because the model expects a 3D input [sequence_length, batch_size, features]. We reduce our 4D array to 2D
    array for compatibility. We have 4D array [video, window, frame, feature]. For our case, each video has 4 windows,
    so total window in our dataset [video*window, ...]. Again, if each frame contains 1024 features, so total features
    [frame*feature]. Basically we convert our 4D array to 2D [total_window, total_feature_in_a_window].
    
    We will feed n number of windows in our model at a time. Here, n = batch_size. Consider sequence length is 1. So, 
    in the end we will have [1, batch_size, features]. Also, we have label for each video, not each window. Since, we
    can have n number of windows per video, we get the shape[1] of embeddings and repeat the number of labels that
    many times.
    """
    # Get the number of windows each video
    label_repetition = embeddings.shape[1]
    # Change shapes
    embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], -1)
    labels = np.repeat(labels, label_repetition)

    # Split the data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(embeddings,
                                                        labels,
                                                        test_size=test_size,
                                                        random_state=42)

    # Convert the np.ndarray to torch.tensor and move to device
    train_embeddings = torch.from_numpy(x_train).to(device)
    train_labels = torch.from_numpy(y_train).to(device)
    test_embeddings = torch.from_numpy(x_test).to(device)
    test_labels = torch.from_numpy(y_test).to(device)

    # Create TensorDataset
    train_data = TensorDataset(train_embeddings, train_labels)
    test_data = TensorDataset(test_embeddings, test_labels)

    return train_data, test_data


def create_dataloader(train_data: TensorDataset,
                      test_data: TensorDataset,
                      batch_size: int = 8):
    """

    """
    # Create DataLoader from torch.utils.data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    return train_loader, test_loader
