"""
Video Anomaly Detection (VAD) Inference Module

This module provides complete inference pipeline for video anomaly detection:
1. Processes segmented unseen videos
2. Generates embeddings using trained models (CNN, ResNet, ViT)
3. Passes embeddings through trained SRU/SRU++ models
4. Enhances predictions using cluster-based cosine similarity weighting
5. Outputs final anomaly classification with confidence scores

USAGE: Use this module for real-time or batch inference on video segments
Author: Generated for VAD Project
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_embedding_cnn import CNNVideoFeatureExtractor
    from video_embedding_resnet import ResNetVideoFeatureExtractor  
    from video_embedding_vit import VideoFeatureExtractor as ViTVideoFeatureExtractor
    from video_clustering import VideoEmbeddingClusterer
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure all embedding modules are in the same directory")

from sklearn.metrics.pairwise import cosine_similarity


class VADInferenceEngine:
    """
    Complete Video Anomaly Detection Inference Engine.
    
    Combines video embedding generation, SRU/SRU++ prediction, and 
    cluster-based enhancement for robust anomaly detection.
    """
    
    def __init__(self, 
                 device: str = 'auto',
                 embedding_model: str = 'resnet',
                 sru_model_type: str = 'sru',
                 confidence_threshold: float = 0.5,
                 cluster_weight: float = 0.3):
        """
        Initialize the VAD inference engine.
        
        Args:
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            embedding_model: Type of embedding model ('cnn', 'resnet', 'vit')
            sru_model_type: Type of SRU model ('sru', 'srupp')
            confidence_threshold: Minimum confidence for positive classification
            cluster_weight: Weight for cluster-based enhancement (0.0-1.0)
        """
        self.device = self._setup_device(device)
        self.embedding_model_type = embedding_model.lower()
        self.sru_model_type = sru_model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.cluster_weight = cluster_weight
        
        # Model components
        self.embedding_model = None
        self.sru_model = None
        self.cluster_centers = None
        self.clusterer = None
        
        # Category mapping
        self.category_names = [
            "Normal", "Abuse", "Arson", "Assault", "Road Accident", "Burglary", 
            "Explosion", "Fighting", "Robbery", "Shooting", "Stealing", "Vandalism"
        ]
        
        # Statistics
        self.inference_stats = {
            'total_videos': 0,
            'processing_times': [],
            'confidence_scores': [],
            'cluster_enhancements': []
        }
        
        print(f"VAD Inference Engine initialized")
        print(f"Device: {self.device}")
        print(f"Embedding Model: {self.embedding_model_type}")
        print(f"SRU Model: {self.sru_model_type}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        torch_device = torch.device(device)
        print(f"Using device: {torch_device}")
        return torch_device
    
    def load_embedding_model(self, model_config: Dict[str, Any]) -> None:
        """
        Load the video embedding model.
        
        Args:
            model_config: Configuration for embedding model
        """
        print(f"Loading {self.embedding_model_type.upper()} embedding model...")
        
        if self.embedding_model_type == 'cnn':
            self.embedding_model = CNNVideoFeatureExtractor(
                device=self.device,
                **model_config.get('cnn', {})
            )
        elif self.embedding_model_type == 'resnet':
            self.embedding_model = ResNetVideoFeatureExtractor(
                device=self.device,
                **model_config.get('resnet', {})
            )
        elif self.embedding_model_type == 'vit':
            self.embedding_model = ViTVideoFeatureExtractor(
                device=self.device,
                **model_config.get('vit', {})
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_type}")
        
        print(f"{self.embedding_model_type.upper()} embedding model loaded successfully")
    
    def load_sru_model(self, model_path: str) -> None:
        """
        Load the trained SRU/SRU++ model.
        
        Args:
            model_path: Path to trained SRU model
        """
        print(f"Loading {self.sru_model_type.upper()} model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        model_config = checkpoint.get('config', {})
        input_size = model_config.get('input_size', 1024)
        hidden_size = model_config.get('hidden_size', 100)
        num_layers = model_config.get('num_layers', 2)
        num_classes = model_config.get('num_classes', 12)
        dropout = model_config.get('dropout', 0.2)
        
        # Initialize model architecture
        if self.sru_model_type == 'sru':
            self.sru_model = SRUModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
        elif self.sru_model_type == 'srupp':
            self.sru_model = SRUppModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported SRU model type: {self.sru_model_type}")
        
        # Load model weights
        self.sru_model.load_state_dict(checkpoint['model_state_dict'])
        self.sru_model.to(self.device)
        self.sru_model.eval()
        
        print(f"{self.sru_model_type.upper()} model loaded successfully")
        print(f"Model configuration: {model_config}")
    
    def load_cluster_model(self, cluster_path: str) -> None:
        """
        Load the trained clustering model and centers.
        
        Args:
            cluster_path: Path to cluster model or centers
        """
        print(f"Loading cluster model from: {cluster_path}")
        
        if cluster_path.endswith('.pkl'):
            # Load full clustering model
            self.clusterer = VideoEmbeddingClusterer()
            self.clusterer.load_model(cluster_path)
            
            # Extract class centers from the clustering model
            with open(cluster_path, 'rb') as f:
                cluster_data = pickle.load(f)
                
            # Try to get class centers from training
            if 'class_centers' in cluster_data:
                self.cluster_centers = cluster_data['class_centers']
            else:
                print("Warning: No class centers found in cluster model")
                self.cluster_centers = None
                
        elif cluster_path.endswith('.npy'):
            # Load cluster centers directly
            self.cluster_centers = np.load(cluster_path)
            print(f"Loaded cluster centers shape: {self.cluster_centers.shape}")
            
        else:
            raise ValueError(f"Unsupported cluster file format: {cluster_path}")
        
        print("Cluster model loaded successfully")
    
    def process_video(self, video_path: str) -> np.ndarray:
        """
        Process a single video and extract frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Processed video frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # Convert to numpy array and normalize
        frames = np.array(frames)
        frames = frames.astype(np.float32) / 255.0
        
        return frames
    
    def generate_embedding(self, video_frames: np.ndarray) -> np.ndarray:
        """
        Generate embedding for video frames using the selected model.
        
        Args:
            video_frames: Video frames array
            
        Returns:
            Video embedding
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Call load_embedding_model first.")
        
        # Add batch dimension if needed
        if len(video_frames.shape) == 4:
            video_frames = video_frames[np.newaxis, ...]
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.embedding_model.extract_features(video_frames)
        
        return embedding
    
    def predict_with_sru(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using SRU/SRU++ model.
        
        Args:
            embedding: Video embedding
            
        Returns:
            Tuple of (logits, probabilities)
        """
        if self.sru_model is None:
            raise ValueError("SRU model not loaded. Call load_sru_model first.")
        
        # Convert to tensor and add batch dimension if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if self.sru_model_type == 'srupp':
                logits, _, _ = self.sru_model(embedding)
            else:
                logits = self.sru_model(embedding)
            
            probabilities = F.softmax(logits, dim=-1)
        
        return logits.cpu().numpy(), probabilities.cpu().numpy()
    
    def compute_cluster_enhancement(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute cluster-based enhancement weights.
        
        Args:
            embedding: Video embedding (should be flattened)
            
        Returns:
            Enhancement weights for each class
        """
        if self.cluster_centers is None:
            print("Warning: No cluster centers available. Skipping cluster enhancement.")
            return np.ones(len(self.category_names)) / len(self.category_names)
        
        # Flatten embedding for cosine similarity
        if len(embedding.shape) > 1:
            embedding_flat = embedding.reshape(-1)
        else:
            embedding_flat = embedding
        
        # Compute cosine similarities with all cluster centers
        similarities = []
        for i in range(len(self.category_names)):
            if isinstance(self.cluster_centers, dict):
                center = self.cluster_centers.get(i)
            else:
                center = self.cluster_centers[i] if i < len(self.cluster_centers) else None
            
            if center is not None:
                similarity = cosine_similarity([embedding_flat], [center])[0][0]
                similarities.append(max(0, similarity))  # Ensure non-negative
            else:
                similarities.append(0.0)
        
        # Convert to numpy array and normalize
        similarities = np.array(similarities)
        
        # Normalize to create weights (softmax-like)
        if np.sum(similarities) > 0:
            exp_similarities = np.exp(similarities - np.max(similarities))
            weights = exp_similarities / np.sum(exp_similarities)
        else:
            weights = np.ones(len(self.category_names)) / len(self.category_names)
        
        return weights
    
    def enhance_predictions(self, 
                          sru_probabilities: np.ndarray, 
                          cluster_weights: np.ndarray) -> np.ndarray:
        """
        Enhance SRU predictions using cluster weights.
        
        Args:
            sru_probabilities: Original SRU predictions
            cluster_weights: Cluster-based enhancement weights
            
        Returns:
            Enhanced probabilities
        """
        # Combine SRU predictions with cluster weights
        enhanced_probs = (1 - self.cluster_weight) * sru_probabilities + \
                        self.cluster_weight * cluster_weights
        
        # Renormalize
        enhanced_probs = enhanced_probs / np.sum(enhanced_probs, axis=-1, keepdims=True)
        
        return enhanced_probs
    
    def predict_single_video(self, video_path: str) -> Dict[str, Any]:
        """
        Complete inference pipeline for a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Process video
            video_frames = self.process_video(video_path)
            
            # Generate embedding
            embedding = self.generate_embedding(video_frames)
            
            # SRU prediction
            sru_logits, sru_probabilities = self.predict_with_sru(embedding)
            
            # Cluster enhancement
            embedding_flat = embedding.reshape(-1) if len(embedding.shape) > 1 else embedding
            cluster_weights = self.compute_cluster_enhancement(embedding_flat)
            
            # Enhanced prediction
            enhanced_probabilities = self.enhance_predictions(sru_probabilities, cluster_weights)
            
            # Get final prediction
            predicted_class = np.argmax(enhanced_probabilities)
            confidence = enhanced_probabilities[0, predicted_class]
            
            # Determine if anomaly
            is_anomaly = predicted_class != 0  # Class 0 is "Normal"
            anomaly_confidence = 1.0 - enhanced_probabilities[0, 0]  # 1 - Normal probability
            
            result = {
                'video_path': video_path,
                'predicted_class': int(predicted_class),
                'predicted_category': self.category_names[predicted_class],
                'confidence': float(confidence),
                'is_anomaly': bool(is_anomaly),
                'anomaly_confidence': float(anomaly_confidence),
                'class_probabilities': enhanced_probabilities[0].tolist(),
                'sru_probabilities': sru_probabilities[0].tolist(),
                'cluster_weights': cluster_weights.tolist(),
                'enhancement_applied': True if self.cluster_centers is not None else False
            }
            
            # Update statistics
            self.inference_stats['total_videos'] += 1
            self.inference_stats['confidence_scores'].append(float(confidence))
            
            return result
            
        except Exception as e:
            return {
                'video_path': video_path,
                'error': str(e),
                'success': False
            }
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Batch inference for multiple videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(video_paths)} videos...")
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.predict_single_video(video_path)
            results.append(result)
        
        return results
    
    def get_inference_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for inference results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Summary statistics
        """
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful predictions'}
        
        # Calculate statistics
        anomaly_count = sum(1 for r in successful_results if r['is_anomaly'])
        normal_count = len(successful_results) - anomaly_count
        
        avg_confidence = np.mean([r['confidence'] for r in successful_results])
        avg_anomaly_confidence = np.mean([r['anomaly_confidence'] for r in successful_results])
        
        # Class distribution
        class_counts = {}
        for r in successful_results:
            class_name = r['predicted_category']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary = {
            'total_videos': len(results),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(results) - len(successful_results),
            'anomaly_count': anomaly_count,
            'normal_count': normal_count,
            'anomaly_rate': anomaly_count / len(successful_results),
            'average_confidence': float(avg_confidence),
            'average_anomaly_confidence': float(avg_anomaly_confidence),
            'class_distribution': class_counts,
            'enhancement_enabled': self.cluster_centers is not None
        }
        
        return summary


# SRU Model Definitions (from training modules)
class SRUModel(nn.Module):
    """SRU model for sequence classification."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(SRUModel, self).__init__()
        
        try:
            from sru import SRU
            self.sru = SRU(input_size, hidden_size, num_layers, dropout=dropout)
        except ImportError:
            print("Warning: SRU not available, using LSTM as fallback")
            self.sru = nn.LSTM(input_size, hidden_size, num_layers, 
                              dropout=dropout, batch_first=True)
        
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        sru_out, _ = self.sru(x)
        output = self.classifier(sru_out[:, -1, :])
        return output


class SRUppModel(nn.Module):
    """SRU++ model for sequence classification."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(SRUppModel, self).__init__()
        
        try:
            from sru import SRU
            self.sru = SRU(input_size, hidden_size, num_layers, dropout=dropout)
        except ImportError:
            print("Warning: SRU not available, using LSTM as fallback")
            self.sru = nn.LSTM(input_size, hidden_size, num_layers, 
                              dropout=dropout, batch_first=True)
        
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        sru_out, hidden = self.sru(x)
        projected = self.projection(sru_out[:, -1, :])
        output = self.classifier(projected)
        return output, sru_out, hidden


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="VAD Inference Pipeline")
    
    # Model paths
    parser.add_argument('--sru_model_path', type=str, required=True,
                       help='Path to trained SRU model')
    parser.add_argument('--cluster_model_path', type=str, required=True,
                       help='Path to cluster model or centers')
    
    # Input/Output
    parser.add_argument('--video_path', type=str,
                       help='Path to single video for inference')
    parser.add_argument('--video_dir', type=str,
                       help='Directory containing videos for batch inference')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    
    # Model configuration
    parser.add_argument('--embedding_model', type=str, default='resnet',
                       choices=['cnn', 'resnet', 'vit'],
                       help='Embedding model type')
    parser.add_argument('--sru_model_type', type=str, default='sru',
                       choices=['sru', 'srupp'],
                       help='SRU model type')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for inference')
    
    # Inference parameters
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for anomaly detection')
    parser.add_argument('--cluster_weight', type=float, default=0.3,
                       help='Weight for cluster enhancement (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video_path and not args.video_dir:
        raise ValueError("Either --video_path or --video_dir must be provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    engine = VADInferenceEngine(
        device=args.device,
        embedding_model=args.embedding_model,
        sru_model_type=args.sru_model_type,
        confidence_threshold=args.confidence_threshold,
        cluster_weight=args.cluster_weight
    )
    
    # Load models
    print("\n" + "="*50)
    print("LOADING MODELS")
    print("="*50)
    
    # Load embedding model
    embedding_config = {
        'cnn': {'output_size': 1024},
        'resnet': {'output_size': 1024},
        'vit': {'output_size': 1024}
    }
    engine.load_embedding_model(embedding_config)
    
    # Load SRU model
    engine.load_sru_model(args.sru_model_path)
    
    # Load cluster model
    engine.load_cluster_model(args.cluster_model_path)
    
    # Perform inference
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)
    
    if args.video_path:
        # Single video inference
        result = engine.predict_single_video(args.video_path)
        results = [result]
    else:
        # Batch inference
        video_paths = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(Path(args.video_dir).glob(ext))
        video_paths = [str(p) for p in video_paths]
        
        if not video_paths:
            raise ValueError(f"No video files found in {args.video_dir}")
        
        results = engine.predict_batch(video_paths)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary = engine.get_inference_summary(results)
    summary_path = os.path.join(args.output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Total videos processed: {summary['total_videos']}")
    print(f"Anomalies detected: {summary['anomaly_count']}")
    print(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
    print(f"Average confidence: {summary['average_confidence']:.4f}")


if __name__ == "__main__":
    main()
