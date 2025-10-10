"""
CNN-based Video Embedding Generation Module
This module provides video embedding generation using a custom CNN architecture.
It processes video frames and generates embeddings for video anomaly detection tasks.
"""

import cv2
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import glob
from typing import List, Tuple, Optional
from multiprocessing import Pool
import gc


class CNN(nn.Module):
    
    def __init__(self, output_dim: int = 1024):
       
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the input size for the fully connected layer
        # For 224x224 input: ((224-2)/2-2)/2-2)/2 = 26
        self.fc1 = nn.Linear(128 * 26 * 26, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
       
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x


class CNNVideoEmbedder:
    
    def __init__(self, 
                 device: torch.device,
                 output_dim: int = 1024,
                 frame_size: Tuple[int, int] = (224, 224)):
        
        self.device = device
        self.output_dim = output_dim
        self.frame_size = frame_size        
        self.model = CNN(output_dim).to(device)
        self.model.eval()
        
        print(f"CNN Video Embedder initialized with output dim: {output_dim}")
        print(f"Device: {device}")
    
    def get_frames(self, video_path: str, 
                   max_frames: Optional[int] = None, 
                   frame_skip: int = 2) -> np.ndarray:

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
        
        current_frames = frames.shape[0]
        
        if current_frames == target_frames:
            return frames
        elif current_frames > target_frames:
            return frames[:target_frames]
        else:
            pad_shape = (target_frames - current_frames,) + frames.shape[1:]
            padding = np.zeros(pad_shape, dtype=frames.dtype)
            return np.concatenate([frames, padding], axis=0)
    
    def extract_embeddings(self, frames: np.ndarray) -> np.ndarray:
        
        embeddings = []
        
        with torch.no_grad():
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)                
                embedding = self.model(frame_tensor)
                embeddings.append(embedding.cpu().numpy().flatten())
        
        return np.array(embeddings)
    
    def process_video(self, 
                     video_path: str, 
                     max_frames: int = 100,
                     frame_skip: int = 2) -> np.ndarray:
        
        try:
            frames = self.get_frames(video_path, max_frames, frame_skip)            
            frames = self.pad_frames(frames, max_frames)            
            embeddings = self.extract_embeddings(frames)
            
            return embeddings
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return np.zeros((max_frames, self.output_dim), dtype=np.float32)


def process_single_video_cnn(args: Tuple[str, CNNVideoEmbedder, int, int]) -> Tuple[str, np.ndarray]:
    
    video_path, embedder, max_frames, frame_skip = args
    embeddings = embedder.process_video(video_path, max_frames, frame_skip)
    return video_path, embeddings


def generate_cnn_embeddings(video_paths: List[str],
                           labels: List[int],
                           device: torch.device,
                           output_dir: str,
                           max_frames: int = 100,
                           frame_skip: int = 2,
                           output_dim: int = 1024,
                           num_processes: int = 1) -> None:
  
    os.makedirs(output_dir, exist_ok=True)    
    embedder = CNNVideoEmbedder(device, output_dim)
    
    print(f"Processing {len(video_paths)} videos...")
    print(f"Max frames per video: {max_frames}")
    print(f"Frame skip: {frame_skip}")
    print(f"Output dimension: {output_dim}")
    
    all_embeddings = []
    valid_labels = []
    
    if num_processes > 1:
        # Use multiprocessing (Note: This might not work well with CUDA)
        print(f"Using multiprocessing with {num_processes} processes...")
        args_list = [(path, embedder, max_frames, frame_skip) for path in video_paths]
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_video_cnn, args_list),
                total=len(args_list),
                desc="Extracting embeddings"
            ))
        
        for i, (video_path, embeddings) in enumerate(results):
            if embeddings is not None and embeddings.shape[0] > 0:
                all_embeddings.append(embeddings)
                valid_labels.append(labels[i])
            else:
                print(f"Skipping video {video_path} due to processing error")
    
    else:
        # Sequential processing
        for video_path, label in tqdm(zip(video_paths, labels), 
                                     total=len(video_paths), 
                                     desc="Extracting embeddings"):
            try:
                embeddings = embedder.process_video(video_path, max_frames, frame_skip)
                all_embeddings.append(embeddings)
                valid_labels.append(label)
            except Exception as e:
                print(f"Skipping video {video_path} due to error: {str(e)}")
                continue
    
    all_embeddings = np.array(all_embeddings)
    valid_labels = np.array(valid_labels)
    
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    print(f"Generated labels shape: {valid_labels.shape}")
    
    embeddings_path = os.path.join(output_dir, 'cnn_embeddings.npy')
    labels_path = os.path.join(output_dir, 'cnn_labels.npy')
    
    np.save(embeddings_path, all_embeddings)
    np.save(labels_path, valid_labels)
    
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Labels saved to: {labels_path}")
    
    del embedder
    gc.collect()


def setup_device():
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
    parser = argparse.ArgumentParser(description="CNN-based Video Embedding Generation")
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to video dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save embeddings')
    parser.add_argument('--dataset_type', type=str, default='binary',
                       choices=['binary', 'multiclass'],
                       help='Type of dataset classification')
    
    parser.add_argument('--max_frames', type=int, default=100,
                       help='Maximum frames per video')
    parser.add_argument('--frame_skip', type=int, default=2,
                       help='Frame skip interval')
    parser.add_argument('--output_dim', type=int, default=1024,
                       help='Output embedding dimension')
    parser.add_argument('--num_processes', type=int, default=1,
                       help='Number of processes for multiprocessing')
    
    args = parser.parse_args()
    
    device = setup_device()
    
    print("Loading dataset...")
    video_paths, labels = load_video_dataset(args.data_path, args.dataset_type)
    print(f"Found {len(video_paths)} videos")
    
    generate_cnn_embeddings(
        video_paths=video_paths,
        labels=labels,
        device=device,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        output_dim=args.output_dim,
        num_processes=args.num_processes
    )
    
    print("CNN embedding generation completed!")


if __name__ == "__main__":
    main()
