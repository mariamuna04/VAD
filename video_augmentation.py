"""
Video Augmentation Module for Video Anomaly Detection (VAD)

This module provides comprehensive video augmentation functionality for video anomaly detection tasks.
It supports various augmentation techniques including:
- Horizontal and vertical flipping
- Brightness and contrast adjustment
- Gaussian noise addition
- Temporal augmentations

USAGE: Use this module to augment video segments for training data enhancement
Author: Generated for VAD Project
"""

import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from typing import Tuple, List, Optional, Dict, Any
import random


class VideoAugmentor:
    """
    A comprehensive video augmentation class that provides various augmentation techniques
    for video anomaly detection tasks.
    """
    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float, float] = (0.7, 1.3),
                 noise_std_range: Tuple[float, float] = (0, 25),
                 flip_probability: float = 0.5,
                 augment_probability: float = 0.8):
        """
        Initialize the VideoAugmentor with augmentation parameters.
        
        Args:
            brightness_range: Range for brightness adjustment (min, max)
            contrast_range: Range for contrast adjustment (min, max)
            noise_std_range: Range for Gaussian noise standard deviation (min, max)
            flip_probability: Probability of applying flip augmentations
            augment_probability: Probability of applying any augmentation
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std_range = noise_std_range
        self.flip_probability = flip_probability
        self.augment_probability = augment_probability
    
    def adjust_brightness_contrast(self, frame: np.ndarray, 
                                 brightness: float, contrast: float) -> np.ndarray:
        """
        Adjust brightness and contrast of a frame.
        
        Args:
            frame: Input frame (H, W, C)
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            
        Returns:
            Augmented frame
        """
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        # Apply brightness and contrast: new_pixel = contrast * pixel + brightness_offset
        brightness_offset = (brightness - 1.0) * 128  # Center around middle gray
        adjusted = contrast * frame_float + brightness_offset
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def add_gaussian_noise(self, frame: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Add Gaussian noise to a frame.
        
        Args:
            frame: Input frame (H, W, C)
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy frame
        """
        if noise_std <= 0:
            return frame
            
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, frame.shape).astype(np.float32)
        
        # Add noise to frame
        noisy_frame = frame.astype(np.float32) + noise
        
        # Clip values to valid range
        noisy_frame = np.clip(noisy_frame, 0, 255)
        
        return noisy_frame.astype(np.uint8)
    
    def flip_frame(self, frame: np.ndarray, flip_type: str) -> np.ndarray:
        """
        Flip a frame horizontally or vertically.
        
        Args:
            frame: Input frame (H, W, C)
            flip_type: 'horizontal', 'vertical', or 'both'
            
        Returns:
            Flipped frame
        """
        if flip_type == 'horizontal':
            return cv2.flip(frame, 1)  # Horizontal flip
        elif flip_type == 'vertical':
            return cv2.flip(frame, 0)  # Vertical flip
        elif flip_type == 'both':
            return cv2.flip(frame, -1)  # Both horizontal and vertical
        else:
            return frame
    
    def augment_frame(self, frame: np.ndarray, augmentation_params: Dict[str, Any]) -> np.ndarray:
        """
        Apply augmentations to a single frame.
        
        Args:
            frame: Input frame (H, W, C)
            augmentation_params: Dictionary of augmentation parameters
            
        Returns:
            Augmented frame
        """
        augmented_frame = frame.copy()
        
        # Apply brightness and contrast adjustment
        if 'brightness' in augmentation_params and 'contrast' in augmentation_params:
            augmented_frame = self.adjust_brightness_contrast(
                augmented_frame, 
                augmentation_params['brightness'], 
                augmentation_params['contrast']
            )
        
        # Add Gaussian noise
        if 'noise_std' in augmentation_params:
            augmented_frame = self.add_gaussian_noise(
                augmented_frame, 
                augmentation_params['noise_std']
            )
        
        # Apply flipping
        if 'flip_type' in augmentation_params:
            augmented_frame = self.flip_frame(
                augmented_frame, 
                augmentation_params['flip_type']
            )
        
        return augmented_frame
    
    def generate_augmentation_params(self) -> Dict[str, Any]:
        """
        Generate random augmentation parameters based on configured ranges.
        
        Returns:
            Dictionary of augmentation parameters
        """
        params = {}
        
        # Randomly decide whether to apply augmentations
        if random.random() > self.augment_probability:
            return params  # Return empty dict for no augmentation
        
        # Brightness and contrast
        params['brightness'] = random.uniform(*self.brightness_range)
        params['contrast'] = random.uniform(*self.contrast_range)
        
        # Gaussian noise
        params['noise_std'] = random.uniform(*self.noise_std_range)
        
        # Flipping
        if random.random() < self.flip_probability:
            flip_options = ['horizontal', 'vertical', 'both']
            params['flip_type'] = random.choice(flip_options)
        
        return params
    
    def augment_video(self, input_path: str, output_path: str, 
                     augmentation_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Augment a single video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to output augmented video
            augmentation_params: Optional specific augmentation parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {input_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"Error: Could not create output video {output_path}")
                cap.release()
                return False
            
            # Generate augmentation parameters if not provided
            if augmentation_params is None:
                augmentation_params = self.generate_augmentation_params()
            
            # Process each frame
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply augmentations
                if augmentation_params:  # Only augment if params are not empty
                    augmented_frame = self.augment_frame(frame, augmentation_params)
                else:
                    augmented_frame = frame  # No augmentation
                
                # Write frame to output video
                out.write(augmented_frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            return True
            
        except Exception as e:
            print(f"Error processing video {input_path}: {str(e)}")
            return False


def augment_single_video(args: Tuple[str, str, VideoAugmentor, Dict[str, Any]]) -> Tuple[str, bool]:
    """
    Worker function for multiprocessing video augmentation.
    
    Args:
        args: Tuple containing (input_path, output_path, augmentor, params)
        
    Returns:
        Tuple of (input_path, success_status)
    """
    input_path, output_path, augmentor, augmentation_params = args
    success = augmentor.augment_video(input_path, output_path, augmentation_params)
    return input_path, success


def create_augmented_dataset(input_dir: str, output_dir: str, 
                           augmentor: VideoAugmentor,
                           augmentations_per_video: int = 3,
                           num_processes: int = 4) -> None:
    """
    Create an augmented dataset from input videos.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save augmented videos
        augmentor: VideoAugmentor instance
        augmentations_per_video: Number of augmented versions per original video
        num_processes: Number of processes for multiprocessing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of video files
    video_extensions = ['.mp4']
    video_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Creating {augmentations_per_video} augmentations per video")
    
    # Prepare arguments for multiprocessing
    args_list = []
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        
        # Create original copy (no augmentation)
        original_output = os.path.join(output_dir, f"{base_name}_original.mp4")
        args_list.append((input_path, original_output, augmentor, {}))
        
        # Create augmented versions
        for i in range(augmentations_per_video):
            augmented_output = os.path.join(output_dir, f"{base_name}_aug_{i:03d}.mp4")
            args_list.append((input_path, augmented_output, augmentor, None))
    
    # Process videos using multiprocessing
    print(f"Processing {len(args_list)} videos using {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(augment_single_video, args_list),
            total=len(args_list),
            desc="Augmenting videos"
        ))
    
    # Report results
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"\nAugmentation completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed videos:")
        for input_path, success in results:
            if not success:
                print(f"  - {input_path}")


def main():
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description="Video Augmentation for VAD")
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save augmented videos')
    
    # Augmentation parameters
    parser.add_argument('--brightness_min', type=float, default=0.7,
                       help='Minimum brightness factor')
    parser.add_argument('--brightness_max', type=float, default=1.3,
                       help='Maximum brightness factor')
    parser.add_argument('--contrast_min', type=float, default=0.7,
                       help='Minimum contrast factor')
    parser.add_argument('--contrast_max', type=float, default=1.3,
                       help='Maximum contrast factor')
    parser.add_argument('--noise_min', type=float, default=0,
                       help='Minimum noise standard deviation')
    parser.add_argument('--noise_max', type=float, default=25,
                       help='Maximum noise standard deviation')
    parser.add_argument('--flip_prob', type=float, default=0.5,
                       help='Probability of applying flip augmentation')
    parser.add_argument('--augment_prob', type=float, default=0.8,
                       help='Probability of applying any augmentation')
    
    # Processing parameters
    parser.add_argument('--augmentations_per_video', type=int, default=3,
                       help='Number of augmented versions per original video')
    parser.add_argument('--num_processes', type=int, default=4,
                       help='Number of processes for multiprocessing')
    
    args = parser.parse_args()
    
    # Create augmentor instance
    augmentor = VideoAugmentor(
        brightness_range=(args.brightness_min, args.brightness_max),
        contrast_range=(args.contrast_min, args.contrast_max),
        noise_std_range=(args.noise_min, args.noise_max),
        flip_probability=args.flip_prob,
        augment_probability=args.augment_prob
    )
    
    # Create augmented dataset
    create_augmented_dataset(
        args.input_dir,
        args.output_dir,
        augmentor,
        args.augmentations_per_video,
        args.num_processes
    )


if __name__ == "__main__":
    main()
