"""
Test script for Real-Time VAD Buffer and Queue System

This script tests the buffer mechanism, segment extraction, and queue operations
without requiring actual video input or trained models.

USAGE: this script validate the buffer system
Author: Generated for VAD Project
"""

import numpy as np
import time
import threading
from collections import deque
from typing import List, Tuple, Optional
import random

class FrameBufferTest:
    """Test class for frame buffer functionality."""
    
    def __init__(self):
        self.buffer_duration = 10  # seconds (smaller for testing)
        self.fps = 30
        self.max_frames = self.buffer_duration * self.fps
        self.frame_size = (224, 224, 3)
        
        # Initialize buffer
        self.buffer = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()
        
        # Statistics
        self.frames_added = 0
        self.frames_dropped = 0
        
        print(f"Test buffer initialized: {self.buffer_duration}s @ {self.fps}fps = {self.max_frames} frames")
    
    def generate_mock_frame(self, frame_id: int) -> np.ndarray:
        """Generate a mock video frame for testing."""
        # Create a frame with unique pattern based on frame_id
        frame = np.zeros(self.frame_size, dtype=np.uint8)
        
        # Add some pattern to make frames distinguishable
        pattern_value = (frame_id % 255)
        frame[:, :, 0] = pattern_value  # Red channel
        frame[:, :, 1] = (pattern_value * 2) % 255  # Green channel  
        frame[:, :, 2] = (pattern_value * 3) % 255  # Blue channel
        
        # Add frame ID in corner (for debugging)
        frame[0:10, 0:50] = frame_id % 255
        
        return frame
    
    def add_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """Add frame to buffer (thread-safe)."""
        try:
            with self.lock:
                if len(self.buffer) >= self.max_frames:
                    self.frames_dropped += 1
                
                # Store frame with metadata
                frame_data = {
                    'frame': frame,
                    'frame_id': frame_id,
                    'timestamp': time.time()
                }
                
                self.buffer.append(frame_data)
                self.frames_added += 1
                
            return True
            
        except Exception as e:
            print(f"Error adding frame {frame_id}: {e}")
            return False
    
    def get_segment(self, segment_size: int = 100) -> Optional[List[dict]]:
        """Extract segment from buffer."""
        with self.lock:
            if len(self.buffer) < segment_size:
                return None
            
            # Get last segment_size frames
            segment = list(self.buffer)[-segment_size:]
            return segment
    
    def pop_segment(self, segment_size: int = 100, overlap: int = 10) -> Optional[List[dict]]:
        """Pop segment with overlap."""
        with self.lock:
            if len(self.buffer) < segment_size:
                return None
            
            # Extract frames
            segment_frames = []
            for _ in range(segment_size - overlap):
                if self.buffer:
                    segment_frames.append(self.buffer.popleft())
            
            # Add overlap frames back
            if overlap > 0 and len(self.buffer) >= overlap:
                overlap_frames = list(self.buffer)[:overlap]
                segment_frames.extend(overlap_frames)
            
            if len(segment_frames) == segment_size:
                return segment_frames
            else:
                # Put frames back if incomplete
                for frame_data in reversed(segment_frames):
                    self.buffer.appendleft(frame_data)
                return None
    
    def get_buffer_stats(self) -> dict:
        """Get buffer statistics."""
        with self.lock:
            oldest_frame_id = self.buffer[0]['frame_id'] if self.buffer else None
            newest_frame_id = self.buffer[-1]['frame_id'] if self.buffer else None
            
            return {
                'current_frames': len(self.buffer),
                'max_frames': self.max_frames,
                'fill_percentage': (len(self.buffer) / self.max_frames) * 100,
                'frames_added': self.frames_added,
                'frames_dropped': self.frames_dropped,
                'oldest_frame_id': oldest_frame_id,
                'newest_frame_id': newest_frame_id
            }


def test_basic_buffer_operations():
    """Test basic buffer add/get operations."""
    print("\n Testing Basic Buffer Operations")
    print("=" * 50)
    
    buffer_test = FrameBufferTest()
    
    # Add some frames
    print("Adding frames to buffer...")
    for i in range(50):
        frame = buffer_test.generate_mock_frame(i)
        success = buffer_test.add_frame(frame, i)
        
        if i % 10 == 0:
            stats = buffer_test.get_buffer_stats()
            print(f"Frame {i}: Buffer {stats['current_frames']}/{stats['max_frames']} "
                  f"({stats['fill_percentage']:.1f}%)")
    
    # Get segment
    print("\nExtracting segment...")
    segment = buffer_test.get_segment(20)
    if segment:
        print(f"Extracted segment with {len(segment)} frames")
        print(f"Segment range: frame {segment[0]['frame_id']} to {segment[-1]['frame_id']}")
    else:
        print("Could not extract segment")
    
    # Final stats
    final_stats = buffer_test.get_buffer_stats()
    print(f"\nFinal buffer stats: {final_stats}")


def test_buffer_overflow():
    """Test buffer behavior when exceeding capacity."""
    print("\n Testing Buffer Overflow Behavior")
    print("=" * 50)
    
    buffer_test = FrameBufferTest()
    
    # Add more frames than buffer can hold
    total_frames = buffer_test.max_frames + 100
    print(f"Adding {total_frames} frames (buffer capacity: {buffer_test.max_frames})")
    
    for i in range(total_frames):
        frame = buffer_test.generate_mock_frame(i)
        buffer_test.add_frame(frame, i)
        
        if (i + 1) % 100 == 0:
            stats = buffer_test.get_buffer_stats()
            print(f"Added {i+1} frames | Buffer: {stats['current_frames']}/{stats['max_frames']} "
                  f"| Oldest: {stats['oldest_frame_id']} | Newest: {stats['newest_frame_id']} "
                  f"| Dropped: {stats['frames_dropped']}")
    
    # Final verification
    final_stats = buffer_test.get_buffer_stats()
    print(f"\nOverflow test completed:")
    print(f"  Total frames added: {final_stats['frames_added']}")
    print(f"  Frames dropped: {final_stats['frames_dropped']}")
    print(f"  Buffer size: {final_stats['current_frames']}")
    print(f"  Expected oldest frame: {total_frames - buffer_test.max_frames}")
    print(f"  Actual oldest frame: {final_stats['oldest_frame_id']}")


def test_segment_extraction_overlap():
    """Test segment extraction with overlap."""
    print("\n Testing Segment Extraction with Overlap")
    print("=" * 50)
    
    buffer_test = FrameBufferTest()
    
    # Fill buffer with frames
    for i in range(200):
        frame = buffer_test.generate_mock_frame(i)
        buffer_test.add_frame(frame, i)
    
    print(f"Buffer filled with {buffer_test.get_buffer_stats()['current_frames']} frames")
    
    # Extract overlapping segments
    segment_size = 30
    overlap = 10
    segments_extracted = 0
    
    print(f"Extracting segments (size: {segment_size}, overlap: {overlap})...")
    
    while True:
        segment = buffer_test.pop_segment(segment_size, overlap)
        if segment is None:
            break
        
        segments_extracted += 1
        start_id = segment[0]['frame_id']
        end_id = segment[-1]['frame_id']
        
        print(f"Segment {segments_extracted}: frames {start_id} to {end_id} "
              f"(length: {len(segment)})")
        
        # Check for expected overlap
        if segments_extracted > 1:
            expected_start = previous_end - overlap + 1
            if start_id != expected_start:
                print(f" Expected overlap! Expected start: {expected_start}, Actual: {start_id}")
        
        previous_end = end_id
        
        if segments_extracted >= 5:  # Limit for demo
            break
    
    print(f"\nExtracted {segments_extracted} overlapping segments")
    remaining_stats = buffer_test.get_buffer_stats()
    print(f"Remaining frames in buffer: {remaining_stats['current_frames']}")


def test_concurrent_access():
    """Test thread-safe concurrent access."""
    print("\n Testing Concurrent Access (Thread Safety)")
    print("=" * 50)
    
    buffer_test = FrameBufferTest()
    
    # Shared state for threads
    producer_running = True
    consumer_running = True
    segments_consumed = 0
    
    def producer_thread():
        """Producer thread: continuously adds frames."""
        frame_id = 0
        while producer_running:
            frame = buffer_test.generate_mock_frame(frame_id)
            buffer_test.add_frame(frame, frame_id)
            frame_id += 1
            time.sleep(1/30)  # 30 FPS
    
    def consumer_thread():
        """Consumer thread: continuously extracts segments."""
        nonlocal segments_consumed
        while consumer_running:
            segment = buffer_test.get_segment(50)
            if segment:
                segments_consumed += 1
                start_id = segment[0]['frame_id']
                end_id = segment[-1]['frame_id']
                print(f"Consumed segment {segments_consumed}: frames {start_id}-{end_id}")
            time.sleep(1.0)  # Process every second
    
    # Start threads
    print("Starting producer and consumer threads...")
    producer = threading.Thread(target=producer_thread, daemon=True)
    consumer = threading.Thread(target=consumer_thread, daemon=True)
    
    producer.start()
    consumer.start()
    
    # Run for a few seconds
    time.sleep(10)
    
    # Stop threads
    producer_running = False
    consumer_running = False
    
    time.sleep(1)  # Wait for threads to finish
    
    final_stats = buffer_test.get_buffer_stats()
    print(f"\nConcurrent access test completed:")
    print(f"  Frames added: {final_stats['frames_added']}")
    print(f"  Segments consumed: {segments_consumed}")
    print(f"  Final buffer size: {final_stats['current_frames']}")


def test_memory_usage():
    """Test memory usage estimation."""
    print("\n Testing Memory Usage Estimation")
    print("=" * 50)
    
    # Calculate theoretical memory usage
    frame_size = 224 * 224 * 3  # Width × Height × Channels
    bytes_per_frame = frame_size * 4  # 4 bytes per float32
    buffer_duration = 30  # seconds
    fps = 30
    total_frames = buffer_duration * fps
    
    total_memory_bytes = total_frames * bytes_per_frame
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    total_memory_gb = total_memory_mb / 1024
    
    print(f"Theoretical Memory Usage:")
    print(f"  Frame size: {frame_size:,} pixels")
    print(f"  Bytes per frame: {bytes_per_frame:,} bytes")
    print(f"  Buffer duration: {buffer_duration} seconds")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total memory: {total_memory_bytes:,} bytes")
    print(f"  Total memory: {total_memory_mb:.2f} MB")
    print(f"  Total memory: {total_memory_gb:.3f} GB")
    
    # Test with actual buffer (smaller size for testing)
    print(f"\nTesting with smaller buffer (10 seconds)...")
    buffer_test = FrameBufferTest()  # 10-second buffer
    
    # Fill buffer
    for i in range(buffer_test.max_frames):
        frame = buffer_test.generate_mock_frame(i)
        buffer_test.add_frame(frame, i)
    
    stats = buffer_test.get_buffer_stats()
    actual_frames = stats['current_frames']
    actual_memory_mb = (actual_frames * bytes_per_frame) / (1024 * 1024)
    
    print(f"Actual buffer test:")
    print(f"  Frames stored: {actual_frames:,}")
    print(f"  Estimated memory: {actual_memory_mb:.2f} MB")


def run_all_tests():
    """Run all buffer tests."""
    print(" Real-Time VAD Buffer System Tests")
    print("=" * 60)
    
    try:
        test_basic_buffer_operations()
        test_buffer_overflow()
        test_segment_extraction_overlap()
        test_concurrent_access()
        test_memory_usage()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
