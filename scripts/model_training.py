from AnomalyDetection.DetectionModel.CreateModel import CreateModel
from AnomalyDetection.DetectionModel.Preprocess import preprocess_data, get_dataloader

train_data, test_data = preprocess_data('../dataset/embeddings.npy', '../dataset/labels.npy')

train_loader, test_loader = get_dataloader(train_data, test_data)

model = CreateModel(input_size=24576, hidden_size=1024)

model.fit(train_loader, epochs=10)

model.evaluate(test_loader)
