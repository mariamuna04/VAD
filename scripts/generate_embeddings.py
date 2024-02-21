"""
USAGE: This generate_embedding.py is used to generate embeddings for a given folder with a specific structure

DATASET STRUCTURE: The folder should contain 2 sub-folders. For example, dataset folder should contain i) dataset/anomaly and
ii) dataset/non-anomaly. This code will take all the videos ending with .mp4 from both the folders and generate
embeddings for each video.

dataset:
| - anomaly
| - non-anomaly

"""

# Importing the required libraries, This is a custom library, see the library for more details...
from AnomalyDetection.CreateEmbeddings import Dataset

# Creating the dataset
"""
Dataset.create_dataset() will take the folder path and class_label_index as input and will return the embeddings and
labels. We can also specify the true_class_name, shuffle_data and video_ext. The default values are as follows:

true_class_name = 'anomaly'
shuffle_data = True
video_ext = '.mp4'

The class_label_index is the index of the class name in the file path. For example, if the file path is
'dataset/anomaly/1.mp4', then the class_label_index will be 1. The index starts from 0.

The embeddings will be of shape [x, 4, 24, 1024] and labels will be of shape [x]. The labels will be 0 for non-anomaly
and 1 for anomaly. The embeddings and labels will be saved in the same folder with the name embeddings.npy and 
labels.npy.
"""
embeddings, labels = Dataset.create_dataset(
    '../dataset/',
    class_label_index=2,
    true_class_name='anomaly'
)

# Print to see the shapes
# Embeddings Shape:  [x, 4, 24, 1024]
# Labels Shape:  [x]
# Where x ~ 3655
print("Embeddings Shape: ", embeddings.shape)
print("Labels Shape: ", labels.shape)

# NOTE: x may differ due to shape mismatch in the videos. Some videos may have less than 4 windows or 24 frames
# or 1024 features. This is fixed by padding the videos with zeros to make them of the same shape using np.pad.
