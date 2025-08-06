"""
ResNet-based Video Embedding Generation Module

This module provides video embedding generation using ResNet50 architecture.
It processes video frames and generates embeddings for video anomaly detection tasks.

USAGE: Use this module to generate ResNet-based embeddings from video files
Author: Generated for VAD Project
"""

import cv2
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import glob
from typing import List, Tuple, Optional
from multiprocessing import Pool
import gc
import requests


class CustomResNet50(nn.Module):
    """
    Custom ResNet50 model for video embedding generation with attention mechanism.
    """
    
    def __init__(self, output_dim: int = 1024, use_attention: bool = True):
        """
        Initialize the Custom ResNet50 model.
        
        Args:
            output_dim: Output dimension of the embeddings
            use_attention: Whether to use attention mechanism
        """
        super(CustomResNet50, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet50 = models.resnet50(weights=None)
        
        # Try to load pre-trained weights
        try:
            # Download weights if not present
            weights_path = self._download_resnet_weights()
            if weights_path and os.path.exists(weights_path):
                self.resnet50.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print("Loaded pre-trained ResNet50 weights")
        except Exception as e:
            print(f"Warning: Could not load ResNet50 weights: {e}")
            print("Using randomly initialized weights")
        
        # Freeze ResNet parameters for feature extraction
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # Layer components
        self.layer0 = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool
        )
        
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Feature projection layers
        self.linear1 = nn.Linear(512, output_dim)  # For layer2 features
        self.linear2 = nn.Linear(2048, output_dim)  # For layer4 features
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.Tanh(),
                nn.Linear(output_dim, 1),
                nn.Softmax(dim=1)
            )
        
        self.output_dim = output_dim

    def _download_resnet_weights(self, weights_dir: str = './weights') -> Optional[str]:
        """Download ResNet50 weights if not present."""
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, 'resnet50.pth')
        
        if os.path.exists(weights_path):
            return weights_path
        
        try:
            url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            print("Downloading ResNet50 weights...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"ResNet50 weights downloaded to {weights_path}")
            return weights_path
        except Exception as e:
            print(f"Failed to download ResNet50 weights: {e}")
            return None

    def forward(self, x):
        """
        Forward pass of the Custom ResNet50.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature embeddings of shape (batch_size, output_dim)
        """
        x = self.layer0(x)
        x = self.layer1(x)
        
        # Extract features from layer2
        feature_1 = self.layer2(x)
        x = self.layer3(feature_1)
        
        # Extract features from layer4
        feature_2 = self.layer4(x)
        
        # Process feature_1 (from layer2)
        feature_1 = self.final_layer(feature_1)
        feature_1 = self.linear1(feature_1)
        
        # Process feature_2 (from layer4)
        feature_2 = self.final_layer(feature_2)
        feature_2 = self.linear2(feature_2)
        
        # Merge features (element-wise mean)
        x = torch.mean(torch.stack([feature_1, feature_2]), dim=0)
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(x)
            x = torch.mul(attention_weights, x)
        
        return x


class ResNetVideoEmbedder:
    """
    ResNet-based video embedding generator.
    """
    
    def __init__(self, 
                 device: torch.device,
                 output_dim: int = 1024,
                 frame_size: Tuple[int, int] = (224, 224),
                 use_attention: bool = True):
        """
        Initialize the ResNet video embedder.
        
        Args:
            device: Device to run computations on
            output_dim: Output dimension of embeddings
            frame_size: Size to resize frames to (width, height)
            use_attention: Whether to use attention mechanism
        """
        self.device = device
        self.output_dim = output_dim
        self.frame_size = frame_size
        
        # Initialize ResNet model
        self.model = CustomResNet50(output_dim, use_attention).to(device)
        self.model.eval()
        
        print(f"ResNet Video Embedder initialized with output dim: {output_dim}")
        print(f"Attention mechanism: {'Enabled' if use_attention else 'Disabled'}")
        print(f"Device: {device}")
    
    def get_frames(self, video_path: str, 
                   max_frames: Optional[int] = None, 
                   frame_skip: int = 2) -> np.ndarray:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None for all)
            frame_skip: Skip every N frames (default: extract every 2nd frame)
            
        Returns:
            Array of frames with shape (num_frames, height, width, channels)
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError(f"Error reading video file: {video_path}")

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_indices = range(0, frame_count, frame_skip)
        
        if max_frames:
            frame_indices = list(frame_indices)[:max_frames]
        
        for i in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                # Resize and normalize frame
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        video.release()
        return np.array(frames)
    
    def pad_frames(self, frames: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Pad or truncate frames to target number.
        
        Args:
            frames: Input frames array
            target_frames: Target number of frames
            
        Returns:
            Padded/truncated frames array
        """
        current_frames = frames.shape[0]
        
        if current_frames == target_frames:
            return frames
        elif current_frames > target_frames:
            # Truncate
            return frames[:target_frames]
        else:
            # Pad with zeros
            pad_shape = (target_frames - current_frames,) + frames.shape[1:]
            padding = np.zeros(pad_shape, dtype=frames.dtype)
            return np.concatenate([frames, padding], axis=0)
    
    def extract_embeddings(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from video frames using ResNet.
        
        Args:
            frames: Array of frames with shape (num_frames, height, width, channels)
            
        Returns:
            Array of embeddings with shape (num_frames, output_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for frame in frames:
                # Convert to tensor and add batch dimension
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Get embedding
                embedding = self.model(frame_tensor)
                embeddings.append(embedding.cpu().numpy().flatten())
        
        return np.array(embeddings)
    
    def process_video(self, 
                     video_path: str, 
                     max_frames: int = 150,
                     frame_skip: int = 2) -> np.ndarray:
        """
        Process a single video and return embeddings.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_skip: Skip every N frames
            
        Returns:
            Video embeddings with shape (max_frames, output_dim)
        """
        try:
            # Extract frames
            frames = self.get_frames(video_path, max_frames, frame_skip)
            
            # Pad or truncate to target size
            frames = self.pad_frames(frames, max_frames)
            
            # Extract embeddings
            embeddings = self.extract_embeddings(frames)
            
            return embeddings
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            # Return zero embeddings in case of error
            return np.zeros((max_frames, self.output_dim), dtype=np.float32)


def process_single_video_resnet(args: Tuple[str, ResNetVideoEmbedder, int, int]) -> Tuple[str, np.ndarray]:
    """
    Worker function for multiprocessing video embedding extraction.
    
    Args:
        args: Tuple containing (video_path, embedder, max_frames, frame_skip)
        
    Returns:
        Tuple of (video_path, embeddings)
    """
    video_path, embedder, max_frames, frame_skip = args
    embeddings = embedder.process_video(video_path, max_frames, frame_skip)
    return video_path, embeddings


def generate_resnet_embeddings(video_paths: List[str],
                              labels: List[int],
                              device: torch.device,
                              output_dir: str,
                              max_frames: int = 150,
                              frame_skip: int = 2,
                              output_dim: int = 1024,
                              use_attention: bool = True,
                              num_processes: int = 1) -> None:
    """
    Generate ResNet embeddings for a list of videos.
    
    Args:
        video_paths: List of video file paths
        labels: List of corresponding labels
        device: Device to run computations on
        output_dir: Directory to save embeddings
        max_frames: Maximum frames per video
        frame_skip: Frame skip interval
        output_dim: Output embedding dimension
        use_attention: Whether to use attention mechanism
        num_processes: Number of processes for multiprocessing (set to 1 to disable)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize embedder
    embedder = ResNetVideoEmbedder(device, output_dim, use_attention=use_attention)
    
    print(f"Processing {len(video_paths)} videos...")
    print(f"Max frames per video: {max_frames}")
    print(f"Frame skip: {frame_skip}")
    print(f"Output dimension: {output_dim}")
    
    all_embeddings = []
    valid_labels = []
    
    # Sequential processing (multiprocessing with CUDA can be problematic)
    for video_path, label in tqdm(zip(video_paths, labels), 
                                 total=len(video_paths), 
                                 desc="Extracting ResNet embeddings"):
        try:
            embeddings = embedder.process_video(video_path, max_frames, frame_skip)
            all_embeddings.append(embeddings)
            valid_labels.append(label)
        except Exception as e:
            print(f"Skipping video {video_path} due to error: {str(e)}")
            continue
    
    # Convert to numpy arrays
    all_embeddings = np.array(all_embeddings)
    valid_labels = np.array(valid_labels)
    
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    print(f"Generated labels shape: {valid_labels.shape}")
    
    # Save embeddings and labels
    embeddings_path = os.path.join(output_dir, 'resnet_embeddings.npy')
    labels_path = os.path.join(output_dir, 'resnet_labels.npy')
    
    np.save(embeddings_path, all_embeddings)
    np.save(labels_path, valid_labels)
    
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Labels saved to: {labels_path}")
    
    # Clean up
    del embedder
    gc.collect()


def setup_device():
    """Setup and return the appropriate device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device selected: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device selected: CUDA")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")
    
    return device


def load_video_dataset(data_path: str, dataset_type: str = 'binary') -> Tuple[List[str], List[int]]:
    """
    Load video dataset paths and labels.
    
    Args:
        data_path: Path to dataset directory
        dataset_type: Type of dataset ('binary' or 'multiclass')
        
    Returns:
        Tuple of (video_paths, labels)
    """
    video_paths = []
    labels = []
    
    if dataset_type == 'binary':
        # Binary classification (anomalous vs normal)
        pattern = os.path.join(data_path, '*', '*.mp4')
        paths = glob.glob(pattern)
        
        for path in paths:
            category = os.path.basename(os.path.dirname(path)).lower()
            label = 1 if 'anomalous' in category or 'fight' in category else 0
            video_paths.append(path)
            labels.append(label)
    
    elif dataset_type == 'multiclass':
        # Multiclass classification
        pattern = os.path.join(data_path, '*', '*.mp4')
        paths = glob.glob(pattern)
        
        category_map = {
            'normal': 0, 'abuse': 1, 'arson': 2, 'assault': 3,
            'roadaccident': 4, 'burglary': 5, 'explosion': 6,
            'fighting': 7, 'robbery': 8, 'shooting': 9,
            'stealing': 10, 'vandalism': 11
        }
        
        for path in paths:
            category = os.path.basename(os.path.dirname(path)).lower()
            label = category_map.get(category, 0)
            video_paths.append(path)
            labels.append(label)
    
    return video_paths, labels


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="ResNet-based Video Embedding Generation")
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to video dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save embeddings')
    parser.add_argument('--dataset_type', type=str, default='binary',
                       choices=['binary', 'multiclass'],
                       help='Type of dataset classification')
    
    # Processing parameters
    parser.add_argument('--max_frames', type=int, default=150,
                       help='Maximum frames per video')
    parser.add_argument('--frame_skip', type=int, default=2,
                       help='Frame skip interval')
    parser.add_argument('--output_dim', type=int, default=1024,
                       help='Output embedding dimension')
    parser.add_argument('--no_attention', action='store_true',
                       help='Disable attention mechanism')
    parser.add_argument('--num_processes', type=int, default=1,
                       help='Number of processes for multiprocessing')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    
    # Load dataset
    print("Loading dataset...")
    video_paths, labels = load_video_dataset(args.data_path, args.dataset_type)
    print(f"Found {len(video_paths)} videos")
    
    # Generate embeddings
    generate_resnet_embeddings(
        video_paths=video_paths,
        labels=labels,
        device=device,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        output_dim=args.output_dim,
        use_attention=not args.no_attention,
        num_processes=args.num_processes
    )
    
    print("ResNet embedding generation completed!")


if __name__ == "__main__":
    main()
