## Multi-layer Embedded Video Anomaly Detection using Attention Driven Recurrence

[Ummay Maria Muna](mailto:umuna201429@bscse.uiu.ac.bd),
[Shanta Biswas](mailto:sbiswas201418@bscse.uiu.ac.bd),
[Syed Abu Ammar Muhammad Zarif](mailto:szarif202009@bscse.uiu.ac.bd),
[Philip Jefferson Deori](mailto:pdeori202111@bscse.uiu.ac.bd),
[Tauseef Tajwar](mailto:tauseef@cse.uiu.ac.bd), and
[Dr. Swakkhar Shatabda](mailto:swakkhar@cse.uiu.ac.bd)

Automated Video Anomaly Detection (VAD) is a challenging task due to its context-dependent and sporadic nature. Recent
deep learning advancements offer promising solutions. In this paper, we propose a spatio-temporal analysis-based video
anomaly detection method where we address challenges such as lengthy videos and anomaly sparsity in an anomalous video
by segmenting and labeling anomalous parts, integrating a sliding window system, and employing multilevel embedding
creation techniques. We enhance feature representation using customized ResNet50 and introduce the parameter-efficient
SRU++ recurrent model with an attention mechanism for the efficient processing of embedding sequences. Additionally, a
cluster-based weighing mechanism was also incorporated to further enhance the prediction capability. Extensive
evaluation utilizing different approaches on the UCF Crime dataset demonstrates our approach's superior performance
compared to state-of-the-art methods, making it suitable for real-world surveillance scenarios.


# Video Embedding Generation Modules

This directory contains three specialized modules for generating video embeddings using different deep learning architectures. These modules are designed to be part of your Video Anomaly Detection (VAD) pipeline.

## Overview

### 1. CNN-based Embeddings (`video_embedding_cnn.py`)
- **Architecture**: Custom 3-layer CNN with attention mechanism
- **Output Dimension**: 1024 (configurable)
- **Default Frames**: 100 per video
- **Features**: 
  - Lightweight and fast
  - Good for basic feature extraction
  - Lower computational requirements

### 2. ResNet-based Embeddings (`video_embedding_resnet.py`)
- **Architecture**: ResNet50 with custom attention mechanism
- **Output Dimension**: 1024 (configurable)  
- **Default Frames**: 150 per video
- **Features**:
  - Pre-trained on ImageNet
  - Multi-level feature extraction (layer2 + layer4)
  - Attention mechanism for enhanced features
  - Automatic weight downloading

### 3. ViT-based Embeddings (`video_embedding_vit.py`)
- **Architecture**: Vision Transformer (ViT) with spatial projection
- **Output Dimension**: 2048 (configurable)
- **Default Frames**: 75 per video
- **Features**:
  - State-of-the-art transformer architecture
  - Excellent for complex visual patterns
  - Requires `timm` library
  - Fallback CNN if ViT unavailable

## Installation Requirements

```bash
# Basic requirements (for all modules)
pip install torch torchvision opencv-python numpy tqdm

# Additional for ResNet module
pip install requests

# Additional for ViT module  
pip install timm
```

## Usage Examples

### 1. CNN Embeddings
```bash
python utils/video_embedding_cnn.py \
    --data_path /path/to/videos \
    --output_dir /path/to/output \
    --max_frames 100 \
    --frame_skip 2 \
    --output_dim 1024 \
    --dataset_type binary
```

### 2. ResNet Embeddings
```bash
python utils/video_embedding_resnet.py \
    --data_path /path/to/videos \
    --output_dir /path/to/output \
    --max_frames 150 \
    --frame_skip 2 \
    --output_dim 1024 \
    --dataset_type binary
```

### 3. ViT Embeddings
```bash
python utils/video_embedding_vit.py \
    --data_path /path/to/videos \
    --output_dir /path/to/output \
    --max_frames 75 \
    --frame_skip 2 \
    --output_dim 2048 \
    --vit_model vit_small_patch16_224 \
    --dataset_type binary
```

## Dataset Structure

The modules expect your dataset to be organized as follows:

### Binary Classification
```
dataset/
├── anomalous/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── normal/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### Multiclass Classification
```
dataset/
├── normal/
├── abuse/
├── arson/
├── assault/
├── roadaccident/
├── burglary/
├── explosion/
├── fighting/
├── robbery/
├── shooting/
├── stealing/
└── vandalism/
```

## Output Format

Each module generates two files:
- `{method}_embeddings.npy`: Shape `(num_videos, max_frames, output_dim)`
- `{method}_labels.npy`: Shape `(num_videos,)`

Example output shapes:
- CNN: `(1000, 100, 1024)` for 1000 videos
- ResNet: `(1000, 150, 1024)` for 1000 videos  
- ViT: `(1000, 75, 2048)` for 1000 videos

## Key Features

### Frame Processing
- **Automatic resizing**: All frames resized to 224x224
- **Normalization**: Pixel values normalized to [0, 1]
- **Padding/Truncation**: Videos padded or truncated to target frame count
- **Color space**: BGR to RGB conversion for consistency

### Error Handling
- **Robust processing**: Continues on individual video errors
- **Zero padding**: Failed videos get zero embeddings
- **Progress tracking**: Real-time progress bars
- **Comprehensive logging**: Detailed error messages

### Memory Management
- **Garbage collection**: Automatic memory cleanup
- **GPU optimization**: Efficient GPU memory usage
- **Batch processing**: Frame-by-frame processing to avoid OOM

## Performance Recommendations

### For Speed (CNN)
```bash
python utils/video_embedding_cnn.py \
    --max_frames 50 \
    --frame_skip 3 \
    --output_dim 512
```

### For Quality (ViT)
```bash
python utils/video_embedding_vit.py \
    --max_frames 100 \
    --frame_skip 1 \
    --output_dim 2048 \
    --vit_model vit_base_patch16_224
```

### For Balance (ResNet)
```bash
python utils/video_embedding_resnet.py \
    --max_frames 100 \
    --frame_skip 2 \
    --output_dim 1024
```

## Integration with VAD Pipeline

These embeddings can be directly used with your SRU/SRU++ training modules:

```bash
# 1. Generate embeddings
python utils/video_embedding_resnet.py --data_path /data --output_dir /embeddings

# 2. Train SRU model
python utils/sru_training.py \
    --embeddings_path /embeddings/resnet_embeddings.npy \
    --labels_path /embeddings/resnet_labels.npy
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_frames`
- Increase `frame_skip`
- Use CPU: `CUDA_VISIBLE_DEVICES="" python ...`

### ViT Model Issues
- Install timm: `pip install timm`
- Use fallback CNN if ViT fails
- Try smaller ViT models: `vit_tiny_patch16_224`

### Video Reading Errors
- Check OpenCV installation: `pip install opencv-python`
- Verify video file formats (MP4 recommended)
- Check file permissions and paths
