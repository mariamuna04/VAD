import torch
from torch.utils.data import DataLoader
from AnomalyDetection.DetectionModel.SRUModel import SRUModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import logging
from tqdm import tqdm
import json

default_model_kwargs = {
    'num_layers': 2,
    'dropout': 0.0,
    'bidirectional': False,
    'projection_size': 0,
    'layer_norm': False,
    'highway_bias': 0.0,
    'has_skip_term': True,
    'rescale': True,
    'nn_rnn_compatible_return': False,
    'custom_m': None,
    'proj_input_to_hidden_first': False,
    'amp_recurrence_fp16': False,
    'normalize_after': False,
    'weight_c_init': None,
    'dropout_layer_prob': 0.2,
    'num_classes': 2,
    'l2_regulation_lambda': 1e-5,
}


class CreateModel:
    def __init__(self,
                 input_size,
                 hidden_size,
                 **kwargs):
        self.model = SRUModel(input_size, hidden_size, **kwargs)

    def fit(self,
            train_loader,
            epochs: int = 10,
            criterion=CrossEntropyLoss(),
            optimizer=None,
            log_report: bool = False):

        if log_report:
            logging.basicConfig(filename='training.log', level=logging.INFO)

        if optimizer is None:
            optimizer = Adam(self.model.parameters())

        for epoch in range(epochs):
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader))

            for i, (videos, labels) in enumerate(progress_bar):
                videos = videos.unsqueeze(0)
                # Forward pass
                outputs = self.model(videos)
                labels = labels.long()  # Convert labels to Long type
                loss = criterion(outputs, labels) + self.model.l2_regularization()  # calculates loss
                total_loss += loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Calculate accuracy per batch
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                batch_accuracy = 100 * total_correct / total_samples

                # Update progress bar description
                progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)

            # Log epoch statistics
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100 * total_correct / total_samples
            logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

        # Close logging
        logging.shutdown()

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Evaluating Model"):
                videos = videos.unsqueeze(0)
                outputs = self.model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f'Test Accuracy of the model on the test dataset: {test_accuracy:.2f}%')

    def predict(self, test_loader: DataLoader):
        pass

    def save(self, path: str):
        torch.save(self.model.state_dict(), path + '.pt')

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path + '.pt'))
        self.model.eval()
        print(f"Model loaded from {path}.pt")
        return self.model


def load_params(json_file: str):
    """
    Read the parameters from a JSON file and return them as a dictionary.
    """
    with open(json_file, 'r') as file:
        return json.load(file)
