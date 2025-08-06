"""
Simple Demo Script for Real-Time Video Anomaly Detection

This script provides a simple demonstration of the real-time VAD system
without requiring actual trained models (uses mock models for demo).

USAGE: Run this script to see how the system works
Author: Generated for VAD Project
"""

import time
import numpy as np
import threading
from typing import Dict, Any
from real_time_vad_config import *

class MockVADDemo:
    """
    Mock VAD system for demonstration purposes.
    Simulates the real-time processing without requiring actual models.
    """
    
    def __init__(self):
        self.is_running = False
        self.frame_count = 0
        self.segment_count = 0
        self.anomaly_count = 0
        
        # Simulate different anomaly types with probabilities
        self.anomaly_scenarios = [
            {"category": "Normal", "prob": 0.7},
            {"category": "Fighting", "prob": 0.1},
            {"category": "Robbery", "prob": 0.08},
            {"category": "Vandalism", "prob": 0.05},
            {"category": "Assault", "prob": 0.04},
            {"category": "Explosion", "prob": 0.03},
        ]
    
    def simulate_frame_capture(self):
        """Simulate continuous frame capture from video stream."""
        print("📹 Starting simulated video capture...")
        
        while self.is_running:
            # Simulate frame capture at 30 FPS
            time.sleep(1/30)  # 30 FPS
            self.frame_count += 1
            
            # Process segment every 100 frames
            if self.frame_count % VIDEO_CONFIG['segment_size'] == 0:
                self.process_segment()
    
    def process_segment(self):
        """Simulate processing of a video segment."""
        self.segment_count += 1
        
        # Simulate processing time
        processing_time = np.random.uniform(0.1, 0.5)  # Random processing time
        time.sleep(processing_time)
        
        # Simulate anomaly detection
        result = self.simulate_anomaly_detection()
        
        # Display result
        self.display_result(result, processing_time)
    
    def simulate_anomaly_detection(self) -> Dict[str, Any]:
        """Simulate the anomaly detection process."""
        
        # Step 1: Simulate embedding generation
        embedding_dim = MODEL_CONFIG['embedding_models']['resnet']['output_size']
        mock_embeddings = np.random.randn(VIDEO_CONFIG['segment_size'], embedding_dim)
        
        # Step 2: Simulate SRU model prediction
        sru_probs = np.random.dirichlet(np.ones(len(CATEGORY_NAMES)))
        sru_predicted_class = np.argmax(sru_probs)
        
        # Step 3: Simulate cluster enhancement
        cluster_weights = np.random.dirichlet(np.ones(len(CATEGORY_NAMES)))
        
        # Combine SRU and cluster predictions
        cluster_weight = PROCESSING_CONFIG['cluster_weight']
        final_probs = (1 - cluster_weight) * sru_probs + cluster_weight * cluster_weights
        
        # Final prediction
        predicted_class = np.argmax(final_probs)
        confidence = np.max(final_probs)
        predicted_category = CATEGORY_NAMES[predicted_class]
        
        # Determine if anomaly
        is_anomaly = predicted_class != 0 and confidence >= PROCESSING_CONFIG['confidence_threshold']
        
        if is_anomaly:
            self.anomaly_count += 1
        
        return {
            'segment_id': self.segment_count,
            'predicted_class': predicted_class,
            'predicted_category': predicted_category,
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'sru_probabilities': sru_probs,
            'cluster_weights': cluster_weights,
            'final_probabilities': final_probs,
            'enhancement_applied': True
        }
    
    def display_result(self, result: Dict[str, Any], processing_time: float):
        """Display processing result."""
        timestamp = time.strftime("%H:%M:%S")
        category = result['predicted_category']
        confidence = result['confidence']
        is_anomaly = result['is_anomaly']
        
        # Choose display format based on anomaly status
        if is_anomaly:
            status_emoji = "🚨"
            status_text = "ANOMALY DETECTED"
        else:
            status_emoji = "✅"
            status_text = "Normal Activity"
        
        print(f"[{timestamp}] {status_emoji} {status_text}")
        print(f"  └─ Category: {category}")
        print(f"  └─ Confidence: {confidence:.4f}")
        print(f"  └─ Processing Time: {processing_time:.3f}s")
        print(f"  └─ Segment: {result['segment_id']}")
        
        # Show top 3 probabilities
        probs = result['final_probabilities']
        top_indices = np.argsort(probs)[-3:][::-1]
        print("  └─ Top Predictions:")
        for i, idx in enumerate(top_indices):
            print(f"     {i+1}. {CATEGORY_NAMES[idx]}: {probs[idx]:.4f}")
        print()
    
    def display_statistics(self):
        """Display running statistics."""
        print(f"\n{'='*50}")
        print(f"📊 REAL-TIME VAD STATISTICS")
        print(f"{'='*50}")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Segments Processed: {self.segment_count}")
        print(f"Anomalies Detected: {self.anomaly_count}")
        if self.segment_count > 0:
            anomaly_rate = (self.anomaly_count / self.segment_count) * 100
            print(f"Anomaly Detection Rate: {anomaly_rate:.2f}%")
        print(f"Processing Rate: {self.segment_count / max(1, time.time() - self.start_time):.2f} segments/sec")
        print(f"{'='*50}\n")
    
    def run_demo(self, duration: int = 60):
        """Run the demo for specified duration."""
        print("🎬 Starting Real-Time Video Anomaly Detection Demo")
        print(f"⏱️  Duration: {duration} seconds")
        print(f"📐 Segment Size: {VIDEO_CONFIG['segment_size']} frames")
        print(f"🎯 Confidence Threshold: {PROCESSING_CONFIG['confidence_threshold']}")
        print(f"⚖️  Cluster Weight: {PROCESSING_CONFIG['cluster_weight']}")
        print("\n" + "="*60)
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start frame capture simulation in separate thread
        capture_thread = threading.Thread(target=self.simulate_frame_capture, daemon=True)
        capture_thread.start()
        
        # Statistics display loop
        last_stats_time = time.time()
        
        try:
            while time.time() - self.start_time < duration:
                current_time = time.time()
                
                # Display statistics every 15 seconds
                if current_time - last_stats_time >= 15:
                    self.display_statistics()
                    last_stats_time = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        
        # Stop demo
        self.is_running = False
        
        # Final statistics
        print("\n🏁 Demo completed!")
        self.display_statistics()
        
        # Summary
        runtime = time.time() - self.start_time
        print(f"📈 DEMO SUMMARY")
        print(f"   Runtime: {runtime:.1f} seconds")
        print(f"   Avg Frames/sec: {self.frame_count / runtime:.1f}")
        print(f"   Avg Segments/sec: {self.segment_count / runtime:.2f}")
        if self.segment_count > 0:
            print(f"   Final Anomaly Rate: {(self.anomaly_count / self.segment_count) * 100:.2f}%")


def demonstrate_buffer_mechanism():
    """Demonstrate the buffer mechanism separately."""
    print("\n🔄 DEMONSTRATING BUFFER MECHANISM")
    print("="*50)
    
    # Simulate the buffer
    from collections import deque
    
    buffer_size = VIDEO_CONFIG['buffer_duration'] * VIDEO_CONFIG['default_fps']  # 30s * 30fps = 900 frames
    frame_buffer = deque(maxlen=buffer_size)
    
    print(f"Buffer capacity: {buffer_size} frames ({VIDEO_CONFIG['buffer_duration']} seconds)")
    
    # Simulate adding frames
    for i in range(1000):  # Add more frames than buffer can hold
        frame_buffer.append(f"frame_{i}")
        
        if i % 100 == 0:
            current_buffer_size = len(frame_buffer)
            buffer_percentage = (current_buffer_size / buffer_size) * 100
            print(f"Added {i+1} frames | Buffer: {current_buffer_size}/{buffer_size} ({buffer_percentage:.1f}%)")
    
    print(f"\nBuffer demonstration complete!")
    print(f"Final buffer size: {len(frame_buffer)}")
    print(f"Oldest frame: {frame_buffer[0]}")
    print(f"Newest frame: {frame_buffer[-1]}")
    
    # Demonstrate segment extraction
    segment_size = VIDEO_CONFIG['segment_size']
    if len(frame_buffer) >= segment_size:
        segment = list(frame_buffer)[-segment_size:]
        print(f"\nExtracted segment of {len(segment)} frames:")
        print(f"Segment start: {segment[0]}")
        print(f"Segment end: {segment[-1]}")


def demonstrate_processing_pipeline():
    """Demonstrate the complete processing pipeline steps."""
    print("\n⚙️  DEMONSTRATING PROCESSING PIPELINE")
    print("="*60)
    
    steps = [
        "1. 📹 Continuous video stream capture",
        "2. 🗃️  Frame buffering (30 seconds @ 30fps = 900 frames)",
        "3. 📦 Segment extraction (100 frames per segment)",
        "4. 🖼️  Frame preprocessing and tensor conversion",
        "5. 🧠 Video embedding generation (CNN/ResNet/ViT)",
        "6. 🔗 SRU/SRU++ temporal sequence modeling",
        "7. 🎯 Clustering-based cosine similarity enhancement",
        "8. ⚖️  Weighted prediction combination",
        "9. 🚨 Anomaly classification and confidence scoring",
        "10. 📊 Results output and statistics tracking"
    ]
    
    for step in steps:
        print(step)
        time.sleep(0.5)  # Dramatic pause
    
    print(f"\n✨ Complete pipeline demonstrated!")


if __name__ == "__main__":
    print("🎯 Real-Time Video Anomaly Detection - Demo Mode")
    print("=" * 60)
    
    # Show configuration
    print("📋 SYSTEM CONFIGURATION:")
    print(f"   Video FPS: {VIDEO_CONFIG['default_fps']}")
    print(f"   Frame Size: {VIDEO_CONFIG['frame_size']}")
    print(f"   Buffer Duration: {VIDEO_CONFIG['buffer_duration']} seconds")
    print(f"   Segment Size: {VIDEO_CONFIG['segment_size']} frames")
    print(f"   Confidence Threshold: {PROCESSING_CONFIG['confidence_threshold']}")
    print(f"   Cluster Weight: {PROCESSING_CONFIG['cluster_weight']}")
    
    # Run demonstrations
    demonstrate_buffer_mechanism()
    demonstrate_processing_pipeline()
    
    # Run main demo
    demo = MockVADDemo()
    demo.run_demo(duration=30)  # Run for 30 seconds
    
    print("\n🎉 All demonstrations completed!")
