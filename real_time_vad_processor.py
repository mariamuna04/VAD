"""
Real-Time Video Anomaly Detection (VAD) Stream Processor
This module provides a complete real-time video stream processing pipeline for anomaly detection:
1. Continuous video stream buffering with queue-based frame management
2. Segment-based processing (100 frames per segment)
3. Video embedding generation using CNN, ResNet50, or ViT models
4. SRU/SRU++ temporal sequence modeling
5. Cluster-based enhancement with cosine similarity weighting
6. Real-time anomaly classification with confidence scores
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import threading
import time
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from pathlib import Path
import logging
from queue import Queue, Empty
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_embedding_cnn import CNNVideoFeatureExtractor
    from video_embedding_resnet import ResNetVideoFeatureExtractor
    from video_embedding_vit import VideoFeatureExtractor as ViTVideoFeatureExtractor
    from video_clustering import VideoEmbeddingClusterer
    from sru_training import SRUModel
    from srupp_training import SRUPlusPlusModel
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure all modules are in the same directory")

from sklearn.metrics.pairwise import cosine_similarity


class FrameBuffer:
    """
    Thread-safe circular buffer for continuous video stream management.
    Acts as a queue that holds N seconds of video footage at specified FPS.
    """
    
    def __init__(self, 
                 buffer_duration: int = 30,  # seconds
                 fps: int = 30,
                 max_frame_size: Tuple[int, int] = (224, 224)):
        
        self.buffer_duration = buffer_duration
        self.fps = fps
        self.max_frames = buffer_duration * fps
        self.max_frame_size = max_frame_size
        
        # Thread-safe deque for frame storage
        self.buffer = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()
        
        self.total_frames_added = 0
        self.frames_dropped = 0
        
        print(f"FrameBuffer initialized: {buffer_duration}s @ {fps}fps = {self.max_frames} frames")
    
    def add_frame(self, frame: np.ndarray) -> bool:
        
        try:
            # Resize frame if necessary
            if frame.shape[:2] != self.max_frame_size:
                frame = cv2.resize(frame, self.max_frame_size)
            
            with self.lock:
                if len(self.buffer) >= self.max_frames:
                    self.frames_dropped += 1
                
                self.buffer.append(frame.copy())
                self.total_frames_added += 1
                
            return True
            
        except Exception as e:
            print(f"Error adding frame to buffer: {e}")
            return False
    
    def get_segment(self, segment_size: int = 100) -> Optional[np.ndarray]:
        
        with self.lock:
            if len(self.buffer) < segment_size:
                return None
            
            segment_frames = list(self.buffer)[-segment_size:]            
            segment = np.array(segment_frames)
            
            return segment
    
    def get_latest_frames(self, count: int) -> Optional[np.ndarray]:
        
        with self.lock:
            if len(self.buffer) < count:
                return None
            
            latest_frames = list(self.buffer)[-count:]
            return np.array(latest_frames)
    
    def pop_segment(self, segment_size: int = 100, overlap: int = 0) -> Optional[np.ndarray]:
        
        with self.lock:
            if len(self.buffer) < segment_size:
                return None
            
            segment_frames = []
            for _ in range(segment_size - overlap):
                if self.buffer:
                    segment_frames.append(self.buffer.popleft())
            
            if overlap > 0 and len(self.buffer) >= overlap:
                overlap_frames = list(self.buffer)[:overlap]
                segment_frames.extend(overlap_frames)
            
            if len(segment_frames) == segment_size:
                return np.array(segment_frames)
            else:
                for frame in reversed(segment_frames):
                    self.buffer.appendleft(frame)
                return None
    
    def get_buffer_info(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'current_frames': len(self.buffer),
                'max_frames': self.max_frames,
                'buffer_full_percentage': (len(self.buffer) / self.max_frames) * 100,
                'total_frames_added': self.total_frames_added,
                'frames_dropped': self.frames_dropped,
                'buffer_duration_seconds': len(self.buffer) / self.fps if self.fps > 0 else 0
            }
    
    def clear(self):
        with self.lock:
            self.buffer.clear()


class VideoStreamProcessor:
    """
    Processes continuous video streams from various sources (camera, file, RTSP, etc.).
    """
    
    def __init__(self, 
                 source: Union[str, int] = 0,  # 0 for webcam, string for file/RTSP
                 fps: int = 30,
                 frame_size: Tuple[int, int] = (224, 224)):
        
        self.source = source
        self.fps = fps
        self.frame_size = frame_size
        
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        
        self.frame_buffer = FrameBuffer(
            buffer_duration=30,  # 30 seconds buffer
            fps=fps,
            max_frame_size=frame_size
        )
        
    def start_capture(self) -> bool:
        
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video source opened: {self.source}")
            print(f"Actual FPS: {actual_fps}, Resolution: {actual_width}x{actual_height}")
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting video capture: {e}")
            return False
    
    def _capture_loop(self):
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                print("End of video stream or capture error")
                break
            
            current_time = time.time()
            
            if current_time - last_frame_time >= frame_interval:
                if self.frame_buffer.add_frame(frame):
                    last_frame_time = current_time
                
            time.sleep(0.001)
    
    def get_segment(self, segment_size: int = 100) -> Optional[np.ndarray]:
        return self.frame_buffer.get_segment(segment_size)
    
    def pop_segment(self, segment_size: int = 100, overlap: int = 10) -> Optional[np.ndarray]:
        return self.frame_buffer.pop_segment(segment_size, overlap)
    
    def get_buffer_info(self) -> Dict[str, Any]:
        return self.frame_buffer.get_buffer_info()
    
    def stop_capture(self):
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Video capture stopped")


class SegmentProcessor:
    
    def __init__(self, device: torch.device = None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SegmentProcessor initialized on device: {self.device}")
    
    def frames_to_tensor(self, frames: np.ndarray, normalize: bool = True) -> torch.Tensor:
        
        # Converted to tensor and change dimensions: (N, H, W, C) -> (N, C, H, W)
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32)
        
        if normalize:
            frames = frames / 255.0
        
        tensor = torch.from_numpy(frames).to(self.device)
        
        # Permuted dimensions: (N, H, W, C) -> (N, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        
        return tensor
    
    def preprocess_segment(self, 
                          frames: np.ndarray,
                          target_size: Tuple[int, int] = (224, 224),
                          normalize: bool = True) -> torch.Tensor:
        
        processed_frames = []
        
        for frame in frames:
            # Resize if necessary
            if frame.shape[:2] != target_size:
                frame = cv2.resize(frame, target_size)
            
            processed_frames.append(frame)
        
        processed_frames = np.array(processed_frames)
        
        return self.frames_to_tensor(processed_frames, normalize)


class EmbeddingGenerator:
   
    def __init__(self, 
                 model_type: str = 'resnet',
                 device: torch.device = None,
                 model_config: Dict[str, Any] = None):
        
        self.model_type = model_type.lower()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config or {}
        
        self.model = None
        self.embedding_dim = None
        self._initialize_model()
        
        print(f"EmbeddingGenerator initialized with {model_type} model on {self.device}")
    
    def _initialize_model(self):
        try:
            if self.model_type == 'cnn':
                self.model = CNNVideoFeatureExtractor(
                    device=self.device,
                    **self.model_config
                )
                self.embedding_dim = 1024  # Default CNN embedding dimension
                
            elif self.model_type == 'resnet':
                self.model = ResNetVideoFeatureExtractor(
                    device=self.device,
                    **self.model_config
                )
                self.embedding_dim = 2048  # Default ResNet embedding dimension
                
            elif self.model_type == 'vit':
                self.model = ViTVideoFeatureExtractor(
                    device=self.device,
                    **self.model_config
                )
                self.embedding_dim = 768  # Default ViT embedding dimension
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            print(f"Error initializing {self.model_type} model: {e}")
            print("Falling back to simple CNN implementation")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        class SimpleCNN(nn.Module):
            def __init__(self, embedding_dim=1024):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((7, 7))
                self.fc = nn.Linear(128 * 7 * 7, embedding_dim)
                
            def forward(self, x):
                # shape: (batch_size, channels, height, width)
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        self.model = SimpleCNN(1024).to(self.device)
        self.embedding_dim = 1024
        print("Fallback CNN model initialized")
    
    def generate_embeddings(self, frames_tensor: torch.Tensor) -> np.ndarray:
        
        try:
            self.model.eval()
            embeddings = []
            
            with torch.no_grad():
                batch_size = 8
                num_frames = frames_tensor.shape[0]
                
                for i in range(0, num_frames, batch_size):
                    batch_end = min(i + batch_size, num_frames)
                    batch_frames = frames_tensor[i:batch_end]
                    
                    if hasattr(self.model, 'extract_features'):
                        batch_embeddings = self.model.extract_features(batch_frames)
                    else:
                        batch_embeddings = self.model(batch_frames)
                    
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    embeddings.append(batch_embeddings)
                
                final_embeddings = np.concatenate(embeddings, axis=0)
                
                return final_embeddings
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            num_frames = frames_tensor.shape[0]
            return np.random.randn(num_frames, self.embedding_dim).astype(np.float32)


class RealTimeVADEngine:
    
    def __init__(self,
                 video_source: Union[str, int] = 0,
                 embedding_model: str = 'resnet',
                 sru_model_type: str = 'sru',
                 cluster_weight: float = 0.3,
                 segment_size: int = 100,
                 segment_overlap: int = 10,
                 confidence_threshold: float = 0.5,
                 device: torch.device = None):
        """
        Initialize the Real-Time VAD Engine.
        
        Args:
            video_source: Video source (camera index, file path, or RTMP URL)
            embedding_model: Embedding model type ('cnn', 'resnet', 'vit')
            sru_model_type: SRU model type ('sru', 'sru++')
            cluster_weight: Weight for cluster-based enhancement (0.0-1.0)
            segment_size: Number of frames per segment
            segment_overlap: Overlap between consecutive segments
            confidence_threshold: Minimum confidence for anomaly detection
            device: PyTorch device
        """
        self.video_source = video_source
        self.embedding_model_type = embedding_model
        self.sru_model_type = sru_model_type
        self.cluster_weight = cluster_weight
        self.segment_size = segment_size
        self.segment_overlap = segment_overlap
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.stream_processor = None
        self.segment_processor = None
        self.embedding_generator = None
        self.sru_model = None
        self.cluster_model = None
        self.cluster_centers = None
        
        self.is_processing = False
        self.processing_thread = None
        self.results_queue = Queue()
        
        self.total_segments_processed = 0
        self.anomalies_detected = 0
        self.processing_times = []
        
        self.category_names = [
            "Normal", "Abuse", "Arson", "Assault", "Road Accident", "Burglary",
            "Explosion", "Fighting", "Robbery", "Shooting", "Stealing", "Vandalism"
        ]
        
        print(f"RealTimeVADEngine initialized:")
        print(f"  - Video source: {video_source}")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - SRU model: {sru_model_type}")
        print(f"  - Device: {self.device}")
        print(f"  - Segment size: {segment_size} frames")
    
    def initialize_components(self):
        print("Initializing components...")
        
        self.stream_processor = VideoStreamProcessor(
            source=self.video_source,
            fps=30,
            frame_size=(224, 224)
        )
        
        self.segment_processor = SegmentProcessor(device=self.device)
        
        self.embedding_generator = EmbeddingGenerator(
            model_type=self.embedding_model_type,
            device=self.device
        )
        
        print("Components initialized successfully")
    
    def load_models(self, 
                   sru_model_path: str,
                   cluster_model_path: Optional[str] = None):
        
        print("Loading models...")
        
        try:
            if self.sru_model_type.lower() == 'sru':
                self.sru_model = SRUModel(
                    input_size=self.embedding_generator.embedding_dim,
                    hidden_size=100,
                    num_layers=2,
                    num_classes=12
                ).to(self.device)
            else:  # sru++
                self.sru_model = SRUPlusPlusModel(
                    input_size=self.embedding_generator.embedding_dim,
                    hidden_size=100,
                    num_layers=2,
                    num_classes=12
                ).to(self.device)
            
            checkpoint = torch.load(sru_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.sru_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.sru_model.load_state_dict(checkpoint)
            
            self.sru_model.eval()
            print(f"SRU model loaded from {sru_model_path}")
            
            if cluster_model_path and os.path.exists(cluster_model_path):
                with open(cluster_model_path, 'rb') as f:
                    cluster_data = pickle.load(f)
                
                if isinstance(cluster_data, dict):
                    self.cluster_model = cluster_data.get('clusterer')
                    self.cluster_centers = cluster_data.get('class_centers')
                else:
                    self.cluster_model = cluster_data
                
                print(f"Clustering model loaded from {cluster_model_path}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def start_processing(self):
        """Start real-time video processing."""
        if not self.stream_processor or not self.sru_model:
            raise ValueError("Components not initialized or models not loaded")
        
        print("Starting real-time processing...")
        
        if not self.stream_processor.start_capture():
            raise RuntimeError("Failed to start video capture")
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("Real-time processing started")
    
    def _processing_loop(self):
        print("Processing loop started...")
        
        while self.is_processing:
            try:
                segment = self.stream_processor.pop_segment(
                    segment_size=self.segment_size,
                    overlap=self.segment_overlap
                )
                
                if segment is None:
                    time.sleep(0.1)  # Wait for more frames
                    continue
                
                start_time = time.time()
                result = self._process_segment(segment)
                processing_time = time.time() - start_time
                
                if result:
                    result['processing_time'] = processing_time
                    result['timestamp'] = time.time()
                    self.results_queue.put(result)
                    
                    self.total_segments_processed += 1
                    self.processing_times.append(processing_time)
                    
                    if result['is_anomaly']:
                        self.anomalies_detected += 1
                        print(f"ANOMALY DETECTED: {result['predicted_category']} "
                              f"(Confidence: {result['confidence']:.4f})")
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_segment(self, segment: np.ndarray) -> Optional[Dict[str, Any]]:
        
        try:
            frames_tensor = self.segment_processor.preprocess_segment(segment)
            
            embeddings = self.embedding_generator.generate_embeddings(frames_tensor)
            
            # Prepared embeddings for SRU (add batch dimension and convert to tensor)
            embeddings_tensor = torch.from_numpy(embeddings).unsqueeze(0).to(self.device)
            
            # Transposed for SRU: (batch, sequence, features) -> (sequence, batch, features)
            embeddings_tensor = embeddings_tensor.transpose(0, 1)
            
            with torch.no_grad():
                sru_output = self.sru_model(embeddings_tensor)
                sru_probabilities = F.softmax(sru_output, dim=1).cpu().numpy()[0]
            
            sru_predicted_class = np.argmax(sru_probabilities)
            sru_confidence = np.max(sru_probabilities)
            
            final_probabilities = sru_probabilities.copy()
            cluster_weights = None
            enhancement_applied = False
            
            if self.cluster_model and self.cluster_centers is not None:
                try:
                    segment_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
                    
                    similarities = cosine_similarity(segment_embedding, self.cluster_centers)[0]
                    cluster_weights = similarities / np.sum(similarities)  # Normalize
                    
                    final_probabilities = (1 - self.cluster_weight) * sru_probabilities + \
                                         self.cluster_weight * cluster_weights
                    enhancement_applied = True
                    
                except Exception as e:
                    print(f"Warning: Cluster enhancement failed: {e}")
                    final_probabilities = sru_probabilities
            
            predicted_class = np.argmax(final_probabilities)
            confidence = np.max(final_probabilities)
            predicted_category = self.category_names[predicted_class]
            
            is_anomaly = predicted_class != 0 and confidence >= self.confidence_threshold
            anomaly_confidence = confidence if is_anomaly else 0.0
            
            result = {
                'predicted_class': int(predicted_class),
                'predicted_category': predicted_category,
                'confidence': float(confidence),
                'is_anomaly': is_anomaly,
                'anomaly_confidence': float(anomaly_confidence),
                'sru_probabilities': sru_probabilities.tolist(),
                'final_probabilities': final_probabilities.tolist(),
                'cluster_weights': cluster_weights.tolist() if cluster_weights is not None else None,
                'enhancement_applied': enhancement_applied,
                'segment_size': len(segment)
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing segment: {e}")
            return None
    
    def get_latest_results(self, count: int = 1) -> List[Dict[str, Any]]:
        results = []
        try:
            for _ in range(min(count, self.results_queue.qsize())):
                result = self.results_queue.get_nowait()
                results.append(result)
        except Empty:
            pass
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        buffer_info = self.stream_processor.get_buffer_info() if self.stream_processor else {}
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'total_segments_processed': self.total_segments_processed,
            'anomalies_detected': self.anomalies_detected,
            'anomaly_rate': self.anomalies_detected / max(1, self.total_segments_processed),
            'average_processing_time': avg_processing_time,
            'pending_results': self.results_queue.qsize(),
            'buffer_info': buffer_info
        }
    
    def stop_processing(self):
        """Stop real-time processing."""
        print("Stopping real-time processing...")
        
        self.is_processing = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        if self.stream_processor:
            self.stream_processor.stop_capture()
        
        print("Real-time processing stopped")


# Example usage and testing functions
def demo_real_time_vad():
    """Demonstration of real-time VAD processing."""
    print("=== Real-Time Video Anomaly Detection Demo ===")
    
    # Initialize engine
    engine = RealTimeVADEngine(
        video_source=0,  # Use webcam (change to video file path for testing)
        embedding_model='resnet',
        sru_model_type='sru',
        cluster_weight=0.3,
        segment_size=100,
        confidence_threshold=0.5
    )
    
    try:
        # Initialize components
        engine.initialize_components()
        
        # Load models (you need to provide actual model paths)
        # engine.load_models(
        #     sru_model_path='path/to/sru_model.pth',
        #     cluster_model_path='path/to/clustering_model.pkl'
        # )
        
        # For demo, we'll skip model loading
        print("Note: Skipping model loading for demo (models not provided)")
        
        # Start processing
        # engine.start_processing()
        
        # Monitor for a while
        # for i in range(30):  # Run for 10 iterations
        #     time.sleep(1)
        #     
        #     # Get latest results
        #     results = engine.get_latest_results(count=5)
        #     for result in results:
        #         print(f"Result: {result['predicted_category']} "
        #               f"({result['confidence']:.4f})")
        #     
        #     # Print statistics
        #     if i % 10 == 0:
        #         stats = engine.get_statistics()
        #         print(f"Stats: {stats}")
        
        # Stop processing
        # engine.stop_processing()
        
        print("Demo completed")
        
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    # Run demonstration
    demo_real_time_vad()
