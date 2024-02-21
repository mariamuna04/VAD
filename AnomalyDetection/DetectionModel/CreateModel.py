from torch.utils.data import DataLoader

from AnomalyDetection.DetectionModel.SRUModel import SRUModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

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

    def fit(self, train_loader: DataLoader, test_loader: DataLoader,
            epochs: int = 10, criterion=CrossEntropyLoss(),
            optimizer=None):
        if optimizer is None:
            optimizer = Adam(self.model.parameters())

        for epoch in range(epochs):
            for i, (videos, labels) in enumerate(train_loader):
                pass

    def predict(self, test_loader: DataLoader):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
