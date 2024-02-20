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
def create_dataset(directory_path: str,
                   class_label_index: int,
                   true_class_name: str = 'anomaly',
                   shuffle_data: bool = True,
                   video_ext: str = 'mp4',
                   save: bool = True,
                   checkpoints_count: int = 0):
    """
    This function will take the folder path and class_label_index as input and will return the embeddings and
    labels. We can also specify the true_class_name, shuffle_data and video_ext.

    Parameters
    ----------
    directory_path : str
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
        Default: True
    checkpoints_count : int, optional
        The number of videos to save the embeddings and labels after. 0 means No checkpoints.
        Default: 0

    Notes
    ----------
    The embeddings will be of shape [x, 4, 24, 1024] and labels will be of shape [x]. The labels will be 0 for
    non-anomaly and 1 for anomaly. The embeddings and labels will be saved in the same folder with the name
    embeddings.npy and labels.npy. Some videos may have less than 4 frames or 24 features. This problem is solved by
    padding the videos with zeros to make them of the same shape using np.pad.

    Returns
    ----------
    all_embeddings : np.ndarray
        The embeddings and labels of the videos.
    all_labels : np.ndarray
        The labels of the videos.
    """

    # If the file path does not exist, raise an error
    if not glob(directory_path):
        raise FileNotFoundError(f"The folder path '{directory_path}' does not exist.")

    # If the file is empty, raise an error
    if not glob(directory_path + f'/*/*.{video_ext}'):
        raise FileNotFoundError(f"The folder '{directory_path}' does not contain any videos with extension '{video_ext}'.")

    # Create glob path for the videos
    rgx = directory_path + f'/*/*.{video_ext}'
    # extract the paths of the videos
    paths = glob(rgx)

    # Randomize the order of the videos
    if shuffle_data:
        shuffle(paths)

    # Embeddings and Labels
    all_embeddings = []
    all_labels = []

    # Variables for checkpoint
    # Save the embeddings and labels after every 100 videos
    count = 0
    checkpoints = 0
    # Iterate over the paths and extract the embeddings
    for video_path in tqdm(paths, desc="Extracting Embeddings"):
        if checkpoints_count != 0:
            if count == checkpoints:
                # create checkpoint
                np.save(f'embeddings_{checkpoints}.npy', all_embeddings)
                np.save(f'labels_{checkpoints}.npy', all_labels)
                checkpoints += 1
                count = 0
        try:
            window = Window(video_path, class_label_index, true_class_name=true_class_name)
            window_embed_object = WindowEmbedded(window)
            embeddings = window_embed_object.window_embeddings
            # Try to append the embeddings and labels
            all_embeddings.append(embeddings)
            all_labels.append(window.class_label)
            count += 1
        except ValueError:
            # print error in red
            print(f"\n\033[91mError windowing video: {video_path}\033[0m")
            continue

    # Validate the embeddings and labels
    assert len(all_embeddings) == len(all_labels)

    """
    Now we have created the list of embeddings and the list of labels. But, there might be cases where the number of
    windows, frames, or features in the embeddings is not the same for all the embeddings. Normally the shape should
    be [x, 4, 24, 1024], where x is the number of videos.
    
    To solve this problem, we will pad the embeddings with 0 where the number of windows, frames, or features is less
    than the maximum. This will make all the embeddings of the same shape.
    
    We will use np.pad to pad the embeddings with 0. keras.preprocessing.sequence.pad_sequences can also be used but
    it is slower than np.pad and also it is not recommended for 3D arrays.
    """

    # Calculating the maximum number of windows, frames, and features
    max_windows = max([len(embeddings) for embeddings in all_embeddings])
    max_frames = max([len(embeddings[0]) for embeddings in all_embeddings])
    max_features = max([len(embeddings[0][0]) for embeddings in all_embeddings])

    # Pad the embeddings with 0 where the number of windows, frames, or features is less than the maximum
    padded_embeddings = []
    for embedding in tqdm(all_embeddings, desc="Padding Embeddings where necessary"):
        # If the number of windows, frames, or features is less than the maximum, pad the embeddings with 0
        # else, append the embeddings as it is
        if embedding.shape[0] < max_windows or embedding.shape[1] < max_frames or embedding.shape[2] < max_features:
            pad_widths = [(0, max_windows - embedding.shape[0]),
                          (0, max_frames - embedding.shape[1]),
                          (0, max_features - embedding.shape[2])]
            padded_embedding = np.pad(embedding, pad_widths, mode='constant', constant_values=0.0)
            padded_embeddings.append(padded_embedding)
        else:
            padded_embeddings.append(embedding)

    # Convert the embeddings and labels to numpy arrays
    padded_arrays = np.asarray(padded_embeddings, dtype='float32')

    # Convert the embeddings to np.ndarray
    all_embeddings = np.array(padded_arrays)
    all_labels = np.array(all_labels)
    # If argument save is True, save the embeddings and labels
    if save:
        np.save('embeddings.npy', all_embeddings)
        np.save('labels.npy', all_labels)

    # Return the embeddings and labels (np.ndarray)
    return all_embeddings, all_labels
