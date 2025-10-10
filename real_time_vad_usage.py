"""
Real-Time Video Anomaly Detection (VAD) Usage Example
This script demonstrates how to use the real-time VAD processor for continuous
video stream analysis with queue-based buffering and segment processing.
"""

import os
import time
import argparse
import json
from pathlib import Path

from real_time_vad_processor import RealTimeVADEngine

def main():
    parser = argparse.ArgumentParser(description='Real-Time Video Anomaly Detection')
    
    # Video source arguments
    parser.add_argument('--video_source', type=str, default='0',
                        help='Video source: camera index (0,1,2...) or video file path')
    parser.add_argument('--rtsp_url', type=str, default=None,
                        help='RTSP stream URL (e.g., rtsp://user:pass@ip:port/stream)')
    
    # Model arguments
    parser.add_argument('--sru_model_path', type=str, required=True,
                        help='Path to trained SRU model (.pth file)')
    parser.add_argument('--cluster_model_path', type=str, default=None,
                        help='Path to trained clustering model (.pkl file)')
    parser.add_argument('--embedding_model', type=str, default='resnet',
                        choices=['cnn', 'resnet', 'vit'],
                        help='Embedding model type')
    parser.add_argument('--sru_model_type', type=str, default='sru',
                        choices=['sru', 'sru++'],
                        help='SRU model type')
    
    # Processing arguments
    parser.add_argument('--segment_size', type=int, default=100,
                        help='Number of frames per segment')
    parser.add_argument('--segment_overlap', type=int, default=10,
                        help='Overlap between consecutive segments')
    parser.add_argument('--cluster_weight', type=float, default=0.3,
                        help='Weight for cluster-based enhancement (0.0-1.0)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum confidence for anomaly detection')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='real_time_results',
                        help='Directory to save results')
    parser.add_argument('--save_results', action='store_true',
                        help='Save processing results to file')
    parser.add_argument('--display_video', action='store_true',
                        help='Display video stream with annotations')
    
    # Runtime arguments
    parser.add_argument('--duration', type=int, default=60,
                        help='Processing duration in seconds (0 for infinite)')
    parser.add_argument('--stats_interval', type=int, default=10,
                        help='Statistics display interval in seconds')
    
    args = parser.parse_args()
    
    video_source = args.video_source
    if args.rtsp_url:
        video_source = args.rtsp_url
    elif args.video_source.isdigit():
        video_source = int(args.video_source)
    
    print("=== Real-Time Video Anomaly Detection ===")
    print(f"Video source: {video_source}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"SRU model: {args.sru_model_type}")
    print(f"Segment size: {args.segment_size} frames")
    print(f"Cluster weight: {args.cluster_weight}")
    print("=" * 50)
    
    try:
        engine = RealTimeVADEngine(
            video_source=video_source,
            embedding_model=args.embedding_model,
            sru_model_type=args.sru_model_type,
            cluster_weight=args.cluster_weight,
            segment_size=args.segment_size,
            segment_overlap=args.segment_overlap,
            confidence_threshold=args.confidence_threshold
        )
        
        print("Initializing components...")
        engine.initialize_components()
        
        print("Loading models...")
        engine.load_models(
            sru_model_path=args.sru_model_path,
            cluster_model_path=args.cluster_model_path
        )
        
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, f'real_time_results_{int(time.time())}.json')
            all_results = []
        
        print("Starting real-time processing...")
        engine.start_processing()
        
        print("Waiting for video buffer to fill...")
        time.sleep(5)
        
        start_time = time.time()
        last_stats_time = start_time
        
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if args.duration > 0 and elapsed_time >= args.duration:
                    print(f"Reached duration limit of {args.duration} seconds")
                    break
                
                results = engine.get_latest_results(count=10)
                
                for result in results:
                    timestamp = result.get('timestamp', current_time)
                    category = result['predicted_category']
                    confidence = result['confidence']
                    is_anomaly = result['is_anomaly']
                    processing_time = result.get('processing_time', 0)
                    
                    status = "ANOMALY" if is_anomaly else "Normal"
                    print(f"[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] "
                          f"{status} | {category} | Confidence: {confidence:.4f} | "
                          f"Process Time: {processing_time:.3f}s")
                    
                    if args.save_results:
                        all_results.append(result)
                
                if current_time - last_stats_time >= args.stats_interval:
                    stats = engine.get_statistics()
                    print(f"\n--- Statistics (Elapsed: {elapsed_time:.1f}s) ---")
                    print(f"Segments processed: {stats['total_segments_processed']}")
                    print(f"Anomalies detected: {stats['anomalies_detected']}")
                    print(f"Anomaly rate: {stats['anomaly_rate']:.3f}")
                    print(f"Avg processing time: {stats['average_processing_time']:.3f}s")
                    print(f"Buffer usage: {stats['buffer_info'].get('buffer_full_percentage', 0):.1f}%")
                    print(f"Pending results: {stats['pending_results']}")
                    print("-" * 50)
                    
                    last_stats_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        print("Stopping processing...")
        engine.stop_processing()
        
        if args.save_results and all_results:
            print(f"Saving {len(all_results)} results to {results_file}")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        final_stats = engine.get_statistics()
        print(f"\n=== Final Statistics ===")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")
        print(f"Segments processed: {final_stats['total_segments_processed']}")
        print(f"Anomalies detected: {final_stats['anomalies_detected']}")
        print(f"Anomaly rate: {final_stats['anomaly_rate']:.3f}")
        print(f"Average processing time: {final_stats['average_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

def test_with_sample_video():
    """Test the system with a sample video file."""
    print("=== Testing with Sample Video ===")
    
    sample_video = "test_video.mp4"  
    sru_model_path = "models/sru_model.pth" 
    cluster_model_path = "models/clustering_model.pkl" 
    if not os.path.exists(sample_video):
        print(f"Sample video not found: {sample_video}")
        return
    
    if not os.path.exists(sru_model_path):
        print(f"SRU model not found: {sru_model_path}")
        return
    
    engine = RealTimeVADEngine(
        video_source=sample_video,
        embedding_model='resnet',
        sru_model_type='sru',
        segment_size=100,
        confidence_threshold=0.5
    )
    
    try:
        engine.initialize_components()
        engine.load_models(sru_model_path, cluster_model_path)
        engine.start_processing()
        
        time.sleep(30)
        
        engine.stop_processing()
        
        stats = engine.get_statistics()
        print(f"Test completed - Processed {stats['total_segments_processed']} segments")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()
