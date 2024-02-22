import numpy as np
from sklearn.model_selection import train_test_split
import torch
from AnomalyDetection import device
from torch.utils.data import TensorDataset, DataLoader


def preprocess_data(embeddings: np.ndarray or str, labels: np.ndarray or str, test_size: float = 0.2):
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
    embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], -1)
    labels = np.repeat(labels, 4)

    # Split the data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(embeddings,
                                                        labels,
                                                        test_size=test_size,
                                                        random_state=42)

    # convert to tensor
    train_embeddings = torch.from_numpy(x_train).to(device)
    train_labels = torch.from_numpy(y_train).to(device)
    test_embeddings = torch.from_numpy(x_test).to(device)
    test_labels = torch.from_numpy(y_test).to(device)

    # Create TensorDataset
    train_data = TensorDataset(train_embeddings, train_labels)
    test_data = TensorDataset(test_embeddings, test_labels)

    return train_data, test_data


def create_dataloader(train_data: TensorDataset, test_data: TensorDataset, batch_size: int = 8):
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    return train_loader, test_loader
