"""
Configuration file for Real-Time Video Anomaly Detection system
"""

# Video Stream Configuration
VIDEO_CONFIG = {
    'default_fps': 30,
    'frame_size': (224, 224),  # (width, height)
    'buffer_duration': 30,      # seconds
    'segment_size': 100,        # frames per segment
    'segment_overlap': 10,      # frames overlap between segments
}

# Model Configuration
MODEL_CONFIG = {
    'embedding_models': {
        'cnn': {
            'output_size': 1024,
            'batch_size': 8,
        },
        'resnet': {
            'output_size': 2048,
            'batch_size': 8,
            'use_attention': True,
        },
        'vit': {
            'output_size': 768,
            'batch_size': 4,
            'model_name': 'vit_base_patch16_224',
        }
    },
    'sru_models': {
        'hidden_size': 100,
        'num_layers': 2,
        'num_classes': 12,
        'dropout': 0.2,
    }
}

# Processing Configuration
PROCESSING_CONFIG = {
    'confidence_threshold': 0.5,
    'cluster_weight': 0.3,
    'anomaly_threshold': 0.5,
    'max_queue_size': 1000,
}

# Category Labels
CATEGORY_NAMES = [
    "Normal", "Abuse", "Arson", "Assault", "Road Accident", "Burglary",
    "Explosion", "Fighting", "Robbery", "Shooting", "Stealing", "Vandalism"
]

# Device Configuration
DEVICE_CONFIG = {
    'prefer_gpu': True,
    'gpu_memory_fraction': 0.8,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'log_file': 'real_time_vad.log',
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_results': True,
    'results_format': 'json',
    'display_video': False,
    'statistics_interval': 10,  # seconds
}
