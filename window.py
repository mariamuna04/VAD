"""
This is an implementation of window class
------Discarded because of low accuracy ------

author: @mariamuna04, @tinykishore
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

# Select Device According to Availability
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Device selected:", device)


class Window:
    def __init__(self, video_path, window_size=4, stride=2, frame_average_count=5):
        """
        A class to manage sliding windows over video frames.
        Parameters:
            video_path (str): The path to the video file.
            window_size (int): The size of the sliding window.
            stride (int): The step size for moving the window.
            frame_average_count (int): The number of frames to be averaged for each group.
        """ 
        self.__video_duration = None
        self.__window_size = window_size
        self.__stride = stride
        self.__frame_average_count = frame_average_count
        self.__video_path = video_path
        self.__total_frames = self.__get_frames()
        self.__averaged_frames = self.__get_average_frame()
        self.__windows = self.__prepare_window()
        self.__window_index = 0

    def next(self):
        """
        This method strides to the next window in the windowed clip.

        Returns:
            numpy.ndarray: The next window in the windowed clip.
        """
        if self.has_next():
            self.__window_index += 1
            return self.__windows[self.__window_index - 1]
        else:
            print("No next window")
            return None

    def has_next(self):
        
        return self.__window_index < len(self.__windows)

    def current_window(self):
    
        return self.__windows[self.__window_index]

    def previous(self):
    
        if self.__window_index > 0:
            self.__window_index -= 1
            return self.__windows[self.__window_index]
        else:
            print("No previous window")
            return None

    def reset(self):
        """
        This method resets the window index to the beginning of the windowed clip.
        """
        self.__window_index = 0

    def __prepare_window(self):
        """
        This method prepares a sliding window of frames from the average frame of the video and divides the average frame into overlapping windows of 'window_size' frames.
        The stride parameter determines the step size for moving the window.
        The number of steps is calculated based on the average frame shape and video duration and the resulting windows are stored in a numpy array

        Will retuen a sliding window of frames.
        """
        window = []
        fps = self.__averaged_frames.shape[0] // self.__video_duration
        steps = int(round((fps * self.__window_size) // self.__stride))

        for i in range(0, len(self.__averaged_frames) - steps, steps):
            window.append(self.__averaged_frames[i: i + steps * 2])
        return np.array(window)

    def __get_average_frame(self):
        """
        This method computes the average frame of the video by taking the mean of consecutive frames. This method divides the video
        frames into groups, each containing 'avg_no' frames. For each group, it computes the mean frame by averaging
        the pixel values of all frames in the group.

        Returns the average frame of the video.
        """
        reduced_frames = []
        for i in range(0, len(self.__total_frames), self.__frame_average_count):
            frames = self.__total_frames[i:i + self.__frame_average_count]
            mean = np.mean(frames, axis=0)
            reduced_frames.append(mean)
        return np.array(reduced_frames)

    def __get_frames(self):
        """
        This method reads the video file and returns the frames and returns an array containing processed frames from the video.
        """
        video = cv2.VideoCapture(self.__video_path)
        if not video.isOpened():
            raise IOError("Error reading video file")

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        self.__video_duration = frame_count // fps
        frames = []
        for i in range(0, frame_count):
            video.set(1, i)
            ret, frame = video.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype("float32") / 255.0
                frames.append(frame)
        video.release()
        return np.array(frames)

    def __iter__(self):
      
        return self  # Return self to make the object iterable

    def __next__(self):
    
        if self.has_next():
            self.__window_index += 1
            return self.__windows[self.__window_index - 1]
        else:
            raise StopIteration

    def get_current_window_stats(self):
        """
        This method returns the stats of the current window.

        Returns a dictionary containing the statistics of the current window.
        """
        return {
            "index": self.__window_index,
            "frame_count": len(self.__windows[self.__window_index]),
            "frame_shape": self.__windows[self.__window_index].shape,
            "frame_dtype": self.__windows[self.__window_index].dtype,
        }

    def __repr__(self):
        return (f"Window (\n\tVideo Path = {self.__video_path}\n"
                f"\tTotal Window = {self.__window_size}\n"
                f"\tStride = {self.__stride}\n"
                f"\tNo. Frames Averaged = {self.__frame_average_count}\n"
                f"\tTotal Frames = {len(self.__total_frames)}\n"
                f"\tAveraged Frames = {len(self.__averaged_frames)}\n"
                f"\tShape = {self.__windows.shape}\n)")

    @property
    def shape(self):
        return self.__windows.shape


class WindowEmbedded:
    """
    This class extracts embeddings from the sliding windows using a CNN model.
    Parameters:
        windows (Window): An object of the Window class.
    """

    def __init__(self, windows: Window):
        self.__windows = windows
        self.__embedding_model = CNN().to(device)
        self.window_embeddings = self.__get_embeddings()

    def __get_embeddings(self):
        embeddings_list = []
        # for window in tqdm.tqdm(self.__windows, desc="Extracting Embeddings", total=self.__windows.shape[0]):
        for window in self.__windows:
            frames = torch.tensor(window).permute(0, 3, 1, 2).to(device)
            frame_embeddings_list = []
            for frame in frames:
                frame = frame.unsqueeze(0)
                frame = frame.to(device)
                frame_embeddings = self.__embedding_model(frame)
                frame_embeddings_list.append(frame_embeddings.flatten().cpu().detach().numpy())
            embeddings_list.append(frame_embeddings_list)
        self.window_embeddings = np.array(embeddings_list)
        return self.window_embeddings


class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for extracting embeddings from video frames. Sample Model
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 0, device=device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, device=device)
        self.d = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, device=device)
        # Pooling layer, All are same
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 26 * 26, 1024,
                             device=device)  # Adjust input size based on your frame size

    def forward(self, x):
        x = x.to(device)
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        # For batch size 1
        x = x.view(-1, 128 * 26 * 26)
        # For batch size > 1
        # x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        return x
