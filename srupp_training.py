"""
SRU++-based Video Anomaly Detection Training Module
This module provides a complete training pipeline for video anomaly detection using SRU++ (Simple Recurrent Unit Plus Plus).
It includes model definition, training, evaluation, and visualization components.
SRU++ is an enhanced version of SRU with improved performance and additional features.
We used this module to train SRU++ models on video embeddings for anomaly detection
"""

import logging
import os
import argparse
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import gc

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sru import SRUpp
from torch.nn import CrossEntropyLoss, Module, Dropout, Linear
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class SRUppModel(Module):
    """
    SRU++-based model for video anomaly detection.
    SRU++ is an enhanced version of SRU with improved performance.
    """
    
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        
        super(SRUppModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = kwargs.get('num_layers', 2)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_classes = kwargs.get('num_classes', 2)
        self.proj_size = kwargs.get('proj_size', 784)  # Default projection size for SRU++
        
        # SRU++ layer
        self.srupp_layers = SRUpp(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            proj_size=self.proj_size,  # SRU++ specific parameter
            dropout=kwargs.get('dropout_prob', 0.0),
            bidirectional=self.bidirectional,
            layer_norm=kwargs.get('layer_norm', False),
            highway_bias=kwargs.get('highway_bias', 0.0),
            rescale=kwargs.get('rescale', True),
            nn_rnn_compatible_return=kwargs.get('nn_rnn_compatible_return', False),
            proj_input_to_hidden_first=kwargs.get('proj_input_to_hidden_first', False),
            normalize_after=kwargs.get('normalize_after', False),
        )
        
        self.dropout = Dropout(kwargs.get('dropout_layer_prob', 0.2))        
        output_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.linear = Linear(in_features=output_size, out_features=self.num_classes)        
        self.l2_reg_lambda = kwargs.get('l2_reg_lambda', 1e-5)

    def forward(self, x):
       
        # SRU++ returns three outputs: output_states, hidden_states, cell_states
        output_states, _, _ = self.srupp_layers(x)
        # Used the last output state
        output = self.linear(self.dropout(output_states[-1]))
        return output

    def l2_regularization(self):
        
        l2_reg = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.l2_reg_lambda * l2_reg

    def get_model_info(self) -> Dict[str, Any]:
        
        return {
            'model_type': 'SRU++',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'proj_size': self.proj_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'num_classes': self.num_classes,
            'l2_reg_lambda': self.l2_reg_lambda,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class VADTrainerPlusPlus:
    
    def __init__(self, 
                 model: SRUppModel,
                 device: torch.device,
                 save_dir: str = 'models',
                 log_dir: str = 'logs'):
        
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Created directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'epoch_times': []
        }
        
        log_filename = os.path.join(log_dir, f'srupp_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, 
                    embeddings_path: str, 
                    labels_path: str,
                    test_size: float = 0.2,
                    batch_size: int = 8,
                    random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        
        print("Loading data for SRU++ training...")
        
        file_embeddings = np.load(embeddings_path)
        file_labels = np.load(labels_path)
        
        if len(file_embeddings) != len(file_labels):
            raise ValueError("The length of the embeddings and labels should be the same")
        
        if len(file_embeddings.shape) != 3:
            raise ValueError(f"The embeddings should be a 3D array [instances, sequence, features]. "
                           f"Found {len(file_embeddings.shape)}D instead.")
        
        print("Files Loaded Successfully")
        print(f"Video Embeddings Shape: {file_embeddings.shape}")
        print(f"Video Labels Shape: {file_labels.shape}")

                        
        x_train, x_test, y_train, y_test = train_test_split(
            file_embeddings, file_labels,
            test_size=test_size,
            random_state=random_state
        )
        
        train_embeddings = torch.from_numpy(x_train).to(self.device)
        train_labels = torch.from_numpy(y_train).to(self.device)
        test_embeddings = torch.from_numpy(x_test).to(self.device)
        test_labels = torch.from_numpy(y_test).to(self.device)
        
        print(f'Shape of Train Embeddings: {train_embeddings.shape}')
        print(f'Shape of Train Labels: {train_labels.shape}')
        print(f'Shape of Test Embeddings: {test_embeddings.shape}')
        print(f'Shape of Test Labels: {test_labels.shape}')
        
        train_data = TensorDataset(train_embeddings, train_labels)
        test_data = TensorDataset(test_embeddings, test_labels)
        
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
        
        del file_embeddings, file_labels, x_train, x_test, y_train, y_test
        gc.collect()
        
        return train_loader, test_loader
    
    def train(self,
              train_loader: DataLoader,
              epochs: int = 50,
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              save_best_model: bool = True) -> Dict[str, List[float]]:
        
        criterion = CrossEntropyLoss()
        
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        print(f"Starting SRU++ training for {epochs} epochs...")
        print(f"Optimizer: {optimizer_type}, Learning Rate: {learning_rate}")
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            epoch_start_time = datetime.now()
            
            progress_bar = tqdm(
                enumerate(train_loader), 
                desc=f"Epoch {epoch + 1}/{epochs}", 
                total=len(train_loader)
            )
            
            for i, (videos, labels) in progress_bar:
                # Reshaped for SRU++: (batch, sequence, features) -> (sequence, batch, features)
                videos = videos.permute(1, 0, 2)
                
                outputs = self.model(videos)
                labels = labels.long()  # Convert labels to Long type
                loss = criterion(outputs, labels) + self.model.l2_regularization()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                
                batch_accuracy = 100 * total_correct / total_samples
                progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)
            
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100 * total_correct / total_samples
            
            self.training_history['train_loss'].append(epoch_loss)
            self.training_history['train_accuracy'].append(epoch_accuracy)
            self.training_history['epoch_times'].append(epoch_time)
            
            log_message = f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s'
            print(log_message)
            self.logger.info(log_message)
            
            if save_best_model and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                self.save_model(f'best_srupp_model_epoch_{epoch+1}.pth', epoch, epoch_accuracy)
        
        print(f'SRU++ Training completed. Best accuracy: {best_accuracy:.2f}%')
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader, show_plots: bool = True) -> Dict[str, Any]:
        
        print("Evaluating SRU++ model...")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            correct = 0
            total = 0
            
            for videos, labels in tqdm(test_loader, desc="Evaluating"):
                videos = videos.permute(1, 0, 2)
                outputs = self.model(videos)                
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        test_accuracy = 100 * correct / total
        print(f'SRU++ Test Accuracy: {test_accuracy:.2f}%')
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        average_precision = average_precision_score(all_labels, all_probs)
        
        results = {
            'test_accuracy': test_accuracy,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        
        if show_plots:
            self.plot_evaluation_results(results)
        
        return results
    
    def plot_evaluation_results(self, results: Dict[str, Any]):
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("SRU++ Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        
        axes[1].plot(results['fpr'], results['tpr'], color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('SRU++ ROC Curve')
        axes[1].legend(loc="lower right")
        
        axes[2].step(results['recall'], results['precision'], color='b', alpha=0.2, where='post')
        axes[2].fill_between(results['recall'], results['precision'], step='post', alpha=0.2, color='b')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_title(f'SRU++ PR Curve (AP={results["average_precision"]:.2f})')
        
        plt.tight_layout()
        plt.show()
        
        plot_path = os.path.join(self.save_dir, f'srupp_evaluation_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"SRU++ evaluation plots saved to: {plot_path}")
    
    def plot_training_history(self):
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(self.training_history['train_loss']) + 1)        
        axes[0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0].set_title('SRU++ Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)        
        axes[1].plot(epochs, self.training_history['train_accuracy'], 'r-', label='Training Accuracy')
        axes[1].set_title('SRU++ Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        plot_path = os.path.join(self.save_dir, f'srupp_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"SRU++ training history plot saved to: {plot_path}")
    
    def save_model(self, filename: str, epoch: int, accuracy: float):
       
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'model_info': self.model.get_model_info(),
            'training_history': self.training_history
        }
        
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"SRU++ model saved to: {save_path}")
    
    def load_model(self, checkpoint_path: str) -> Dict[str, Any]:
       
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"SRU++ model loaded from: {checkpoint_path}")
        print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2f}%")
        
        return checkpoint


def setup_device():
   
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device selected: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device selected: CUDA")
        os.system("nvidia-smi")
        print(f"Device capability: {torch.cuda.get_device_capability(device)}")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")
    
    return device


def main():
   
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to embeddings numpy file')
    parser.add_argument('--labels_path', type=str, required=True,
                       help='Path to labels numpy file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    
    parser.add_argument('--input_size', type=int, default=2048,
                       help='Input feature size')
    parser.add_argument('--hidden_size', type=int, default=1024,
                       help='Hidden size in SRU++')
    parser.add_argument('--proj_size', type=int, default=784,
                       help='Projection size in SRU++ (specific to SRU++)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of SRU++ layers')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional SRU++')
    parser.add_argument('--dropout_prob', type=float, default=0.0,
                       help='Dropout probability in SRU++')
    parser.add_argument('--dropout_layer_prob', type=float, default=0.2,
                       help='Dropout probability in linear layer')
    parser.add_argument('--l2_reg_lambda', type=float, default=1e-5,
                       help='L2 regularization lambda')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer type')
    
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show evaluation plots')
    
    args = parser.parse_args()
    
    device = setup_device()
    
    model = SRUppModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        proj_size=args.proj_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout_prob=args.dropout_prob,
        dropout_layer_prob=args.dropout_layer_prob,
        l2_reg_lambda=args.l2_reg_lambda
    )
    
    print("\nSRU++ Model Architecture:")
    print(json.dumps(model.get_model_info(), indent=2))
    
    trainer = VADTrainerPlusPlus(model, device, args.save_dir, args.log_dir)
    
    train_loader, test_loader = trainer.prepare_data(
        args.embeddings_path,
        args.labels_path,
        args.test_size,
        args.batch_size
    )
    
    history = trainer.train(
        train_loader,
        args.epochs,
        args.learning_rate,
        args.optimizer
    )
    
    trainer.plot_training_history()    
    results = trainer.evaluate(test_loader, args.show_plots)
    
    trainer.save_model(
        f'final_srupp_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
        args.epochs,
        results['test_accuracy']
    )
    
    print("\nSRU++ training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
