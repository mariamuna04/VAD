from random import shuffle
from glob import glob
import numpy as np
from tqdm import tqdm

__all__ = ['create_dataset']

from .Window import Window, WindowEmbedded


def create_dataset(file_path: str, class_label_index: int,
                   true_class_name: str = 'anomaly', shuffle_data: bool = True, video_ext: str = 'mp4'):

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

    # Save the embeddings and labels
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)
    np.save('embeddings.npy', all_embeddings)
    np.save('labels.npy', all_labels)
    return all_embeddings, all_labels
