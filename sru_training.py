"""
SRU-based Video Anomaly Detection Training Module

This module provides a complete training pipeline for video anomaly detection using SRU (Simple Recurrent Unit).
It includes model definition, training, evaluation, and visualization components.

USAGE: Use this module to train SRU models on video embeddings for anomaly detection
Author: Generated for VAD Project
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
from sru import SRU
from torch.nn import CrossEntropyLoss, Module, Dropout, Linear
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class SRUModel(Module):
    """
    SRU-based model for video anomaly detection.
    """
    
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        """
        Initialize the SRU model.
        
        Args:
            input_size: Size of input features (e.g., 2048 for ResNet features)
            hidden_size: Size of hidden state in SRU
            **kwargs: Additional SRU parameters
        """
        super(SRUModel, self).__init__()
        
        # Store model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = kwargs.get('num_layers', 2)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_classes = kwargs.get('num_classes', 2)
        
        # Main SRU layer
        self.sru_layers = SRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dropout=kwargs.get('dropout_prob', 0.0),
            bidirectional=self.bidirectional,
            layer_norm=kwargs.get('layer_norm', False),
            highway_bias=kwargs.get('highway_bias', 0.0),
            rescale=kwargs.get('rescale', True),
            nn_rnn_compatible_return=kwargs.get('nn_rnn_compatible_return', False),
            proj_input_to_hidden_first=kwargs.get('proj_input_to_hidden_first', False),
            amp_recurrence_fp16=kwargs.get('amp_recurrence_fp16', False),
            normalize_after=kwargs.get('normalize_after', False),
        )
        
        # Dropout layer
        self.dropout = Dropout(kwargs.get('dropout_layer_prob', 0.2))
        
        # Linear layer (Fully connected layer)
        output_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.linear = Linear(in_features=output_size, out_features=self.num_classes)
        
        # L2 regularization
        self.l2_reg_lambda = kwargs.get('l2_reg_lambda', 1e-5)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (sequence_length, batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        output_states, _ = self.sru_layers(x)
        # Use the last output state
        output = self.linear(self.dropout(output_states[-1]))
        return output

    def l2_regularization(self):
        """
        Calculate L2 regularization loss.
        
        Returns:
            L2 regularization loss
        """
        l2_reg = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.l2_reg_lambda * l2_reg

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'num_classes': self.num_classes,
            'l2_reg_lambda': self.l2_reg_lambda,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class VADTrainer:
    """
    Trainer class for Video Anomaly Detection using SRU.
    """
    
    def __init__(self, 
                 model: SRUModel,
                 device: torch.device,
                 save_dir: str = 'models',
                 log_dir: str = 'logs'):
        """
        Initialize the trainer.
        
        Args:
            model: SRU model instance
            device: Device to use for training
            save_dir: Directory to save models
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'epoch_times': []
        }
        
        # Setup logging
        log_filename = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
        """
        Load and prepare data for training.
        
        Args:
            embeddings_path: Path to embeddings numpy file
            labels_path: Path to labels numpy file
            test_size: Fraction of data to use for testing
            batch_size: Batch size for data loaders
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        print("Loading data...")
        
        # Load embeddings and labels
        file_embeddings = np.load(embeddings_path)
        file_labels = np.load(labels_path)
        
        # Validate data
        if len(file_embeddings) != len(file_labels):
            raise ValueError("The length of the embeddings and labels should be the same")
        
        if len(file_embeddings.shape) != 3:
            raise ValueError(f"The embeddings should be a 3D array [instances, sequence, features]. "
                           f"Found {len(file_embeddings.shape)}D instead.")
        
        print("Files Loaded Successfully")
        print(f"Video Embeddings Shape: {file_embeddings.shape}")
        print(f"Video Labels Shape: {file_labels.shape}")
        
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            file_embeddings, file_labels,
            test_size=test_size,
            random_state=random_state
        )
        
        # Convert to tensors
        train_embeddings = torch.from_numpy(x_train).to(self.device)
        train_labels = torch.from_numpy(y_train).to(self.device)
        test_embeddings = torch.from_numpy(x_test).to(self.device)
        test_labels = torch.from_numpy(y_test).to(self.device)
        
        print(f'Shape of Train Embeddings: {train_embeddings.shape}')
        print(f'Shape of Train Labels: {train_labels.shape}')
        print(f'Shape of Test Embeddings: {test_embeddings.shape}')
        print(f'Shape of Test Labels: {test_labels.shape}')
        
        # Create datasets and data loaders
        train_data = TensorDataset(train_embeddings, train_labels)
        test_data = TensorDataset(test_embeddings, test_labels)
        
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
        
        # Clean up memory
        del file_embeddings, file_labels, x_train, x_test, y_train, y_test
        gc.collect()
        
        return train_loader, test_loader
    
    def train(self,
              train_loader: DataLoader,
              epochs: int = 50,
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              save_best_model: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam' or 'sgd')
            save_best_model: Whether to save the best model
            
        Returns:
            Training history dictionary
        """
        # Setup loss function and optimizer
        criterion = CrossEntropyLoss()
        
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Optimizer: {optimizer_type}, Learning Rate: {learning_rate}")
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            epoch_start_time = datetime.now()
            
            # Create progress bar
            progress_bar = tqdm(
                enumerate(train_loader), 
                desc=f"Epoch {epoch + 1}/{epochs}", 
                total=len(train_loader)
            )
            
            for i, (videos, labels) in progress_bar:
                # Reshape for SRU: (batch, sequence, features) -> (sequence, batch, features)
                videos = videos.permute(1, 0, 2)
                
                # Forward pass
                outputs = self.model(videos)
                labels = labels.long()  # Convert labels to Long type
                loss = criterion(outputs, labels) + self.model.l2_regularization()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                batch_accuracy = 100 * total_correct / total_samples
                progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)
            
            # Calculate epoch statistics
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100 * total_correct / total_samples
            
            # Store training history
            self.training_history['train_loss'].append(epoch_loss)
            self.training_history['train_accuracy'].append(epoch_accuracy)
            self.training_history['epoch_times'].append(epoch_time)
            
            # Log epoch results
            log_message = f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s'
            print(log_message)
            self.logger.info(log_message)
            
            # Save best model
            if save_best_model and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                self.save_model(f'best_model_epoch_{epoch+1}.pth', epoch, epoch_accuracy)
        
        print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader, show_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            show_plots: Whether to show evaluation plots
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating model...")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            correct = 0
            total = 0
            
            for videos, labels in tqdm(test_loader, desc="Evaluating"):
                # Reshape for SRU
                videos = videos.permute(1, 0, 2)
                
                # Forward pass
                outputs = self.model(videos)
                
                # Get predictions and probabilities
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                
                # Accumulate results
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        # Calculate metrics
        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        average_precision = average_precision_score(all_labels, all_probs)
        
        # Create evaluation results
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
        
        # Show plots if requested
        if show_plots:
            self.plot_evaluation_results(results)
        
        return results
    
    def plot_evaluation_results(self, results: Dict[str, Any]):
        """
        Plot evaluation results including confusion matrix, ROC curve, and PR curve.
        
        Args:
            results: Dictionary containing evaluation results
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        
        # ROC Curve
        axes[1].plot(results['fpr'], results['tpr'], color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        
        # Precision-Recall Curve
        axes[2].step(results['recall'], results['precision'], color='b', alpha=0.2, where='post')
        axes[2].fill_between(results['recall'], results['precision'], step='post', alpha=0.2, color='b')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_title(f'PR Curve (AP={results["average_precision"]:.2f})')
        
        plt.tight_layout()
        plt.show()
        
        # Save plots
        plot_path = os.path.join(self.save_dir, f'evaluation_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {plot_path}")
    
    def plot_training_history(self):
        """
        Plot training history (loss and accuracy curves).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Training Loss
        axes[0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Training Accuracy
        axes[1].plot(epochs, self.training_history['train_accuracy'], 'r-', label='Training Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
    
    def save_model(self, filename: str, epoch: int, accuracy: float):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
            epoch: Current epoch
            accuracy: Current accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'model_info': self.model.get_model_info(),
            'training_history': self.training_history
        }
        
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2f}%")
        
        return checkpoint


def setup_device():
    """
    Setup and return the appropriate device for training.
    
    Returns:
        torch.device: The device to use for training
    """
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
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description="SRU-based Video Anomaly Detection Training")
    
    # Data parameters
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to embeddings numpy file')
    parser.add_argument('--labels_path', type=str, required=True,
                       help='Path to labels numpy file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=2048,
                       help='Input feature size')
    parser.add_argument('--hidden_size', type=int, default=1024,
                       help='Hidden size in SRU')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of SRU layers')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional SRU')
    parser.add_argument('--dropout_prob', type=float, default=0.0,
                       help='Dropout probability in SRU')
    parser.add_argument('--dropout_layer_prob', type=float, default=0.2,
                       help='Dropout probability in linear layer')
    parser.add_argument('--l2_reg_lambda', type=float, default=1e-5,
                       help='L2 regularization lambda')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer type')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show evaluation plots')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    
    # Create model
    model = SRUModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout_prob=args.dropout_prob,
        dropout_layer_prob=args.dropout_layer_prob,
        l2_reg_lambda=args.l2_reg_lambda
    )
    
    print("\nModel Architecture:")
    print(json.dumps(model.get_model_info(), indent=2))
    
    # Create trainer
    trainer = VADTrainer(model, device, args.save_dir, args.log_dir)
    
    # Prepare data
    train_loader, test_loader = trainer.prepare_data(
        args.embeddings_path,
        args.labels_path,
        args.test_size,
        args.batch_size
    )
    
    # Train model
    history = trainer.train(
        train_loader,
        args.epochs,
        args.learning_rate,
        args.optimizer
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    results = trainer.evaluate(test_loader, args.show_plots)
    
    # Save final model
    trainer.save_model(
        f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
        args.epochs,
        results['test_accuracy']
    )
    
    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
