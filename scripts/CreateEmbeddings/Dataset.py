"""
This file contains the function to create the dataset from the videos. Dataset means the embeddings and labels. The
embeddings are the features extracted from the videos and the labels are the class labels for the videos. The class
labels are 0 for non-anomaly and 1 for anomaly.
"""

# Importing the required libraries
from random import shuffle
from glob import glob
import numpy as np
from tqdm import tqdm
from .Window import Window, WindowEmbedded


# Function to create the dataset
def create_dataset(file_path: str,
                   class_label_index: int,
                   true_class_name: str = 'anomaly',
                   shuffle_data: bool = True,
                   video_ext: str = 'mp4',
                   save: bool = True):
    """
    This function will take the folder path and class_label_index as input and will return the embeddings and
    labels. We can also specify the true_class_name, shuffle_data and video_ext.

    Parameters
    ----------
    file_path : str
        The folder path containing the videos.
    class_label_index : int
        The index of the class name in the file path. For example, if the file path is
        'dataset/anomaly/1.mp4', then the class_label_index will be 1. The index starts from 0.
    true_class_name : str, optional
        The true class name, where the class label is 1.
        Default: 'anomaly'
    shuffle_data : bool, optional
        Whether to shuffle the data or not.
        Default: True
    video_ext : str, optional
        The extension of the videos.
        Default: 'mp4'
    save : bool, optional
        Whether to save the embeddings and labels or not.

    Notes
    ----------
    The embeddings will be of shape [x, 4, 24, 1024] and labels will be of shape [x]. The labels will be 0 for non-anomaly
    and 1 for anomaly. The embeddings and labels will be saved in the same folder with the name embeddings.npy and
    labels.npy. Some videos may have less than 4 frames or 24 features. This problem can be solved by padding the videos
    with zeros to make them of the same shape. Will be done in the future.

    Returns
    ----------
    all_embeddings : np.ndarray
        The embeddings and labels of the videos.
    all_labels : np.ndarray
        The labels of the videos.
    """

    # Create glob path for the videos
    rgx = file_path + f'/*/*.{video_ext}'
    # extract the paths of the videos
    paths = glob(rgx)
    # Randomize the order of the videos
    if shuffle_data:
        shuffle(paths)

    # Embeddings and Labels
    all_embeddings = []
    all_labels = []

    # Iterate over the paths and extract the embeddings
    for video_path in tqdm(paths):
        try:
            window = Window(video_path, class_label_index, true_class_name=true_class_name)
            window_embed_object = WindowEmbedded(window)
            embeddings = window_embed_object.window_embeddings
            # Try to append the embeddings and labels
            all_embeddings.append(embeddings)
            all_labels.append(window.class_label)
        except ValueError:
            # print error in red
            print(f"\n\033[91mError windowing video: {video_path}\033[0m")
            continue

    # Save the embeddings and labels and return
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)
    if save:
        np.save('embeddings.npy', all_embeddings)
        np.save('labels.npy', all_labels)
    return all_embeddings, all_labels
