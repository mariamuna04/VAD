import cv2
import numpy as np
import torch
from AnomalyDetection.CreateEmbeddings.EmbeddingModel import CNN
from AnomalyDetection import device, this_os


# This class takes a video path and return a `Window` object. A `Window` object is a
# generator object which can be used to iterate over the video frames in a sliding window fashion.

class Window:
    def __init__(self, video_path,
                 class_label_index,
                 true_class_name='anomaly', window_size=4, stride=2,
                 frame_average_count=5):
        """
        A class to manage sliding windows over video frames.

        :param video_path (str): The path to the video file.
        :param window_size (int): The size of the sliding window (default is 4).
        :param stride (int): The step size for moving the window (default is 2).
        :param frame_average_count (int): The number of frames to be averaged for each group (default is 5).
        """
        # The duration of the video in seconds
        self.__video_duration = None
        # The size of the sliding window
        self.__window_size = window_size
        # The step size for moving the window
        self.__stride = stride
        # The number of frames to be averaged for each group
        self.__frame_average_count = frame_average_count
        # The path to the video file
        self.__video_path = video_path
        # An array containing processed frames from the video
        self.__total_frames = self.__get_frames()
        # The average frame of the video
        self.__averaged_frames = self.__get_average_frame()
        # A sliding window of frames
        self.__windows = self.__prepare_window()
        # The index of the current window in the windowed clip
        self.__window_index = 0
        # Class label of the video path
        if this_os == "nt":
            self.class_label = 1 if video_path.split('\\')[class_label_index] == true_class_name else 0
        else:
            self.class_label = 1 if video_path.split('/')[class_label_index] == true_class_name else 0

    def next(self):
        """
        This method strides to the next window in the windowed clip.
        :return: numpy.ndarray: The next window in the windowed clip.
        """
        if self.has_next():
            self.__window_index += 1
            return self.__windows[self.__window_index - 1]
        else:
            print("No next window")
            return None

    def has_next(self):
        """
        This method checks if there is a next window in the windowed clip.
        :return: bool: True if there is a next window, False otherwise.
        """
        return self.__window_index < len(self.__windows)

    def current_window(self):
        """
        This method returns the current window in the windowed clip.
        :return: numpy.ndarray: The current window in the windowed clip.
        """
        return self.__windows[self.__window_index]

    def previous(self):
        """
        This method strides to the previous window in the windowed clip.
        :return: numpy.ndarray: The previous window in the windowed clip.
        """
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
        Prepares a sliding window of frames from the average frame of the video.

        :returns:
            numpy.ndarray: A sliding window of frames.

        Notes:
            This method divides the average frame into overlapping windows of 'window_size' frames.
            The stride parameter determines the step size for moving the window.
            The number of steps is calculated based on the average frame shape and video duration.
            The resulting windows are stored in a numpy array.
        """
        window = []
        fps = self.__averaged_frames.shape[0] // self.__video_duration
        steps = int(round((fps * self.__window_size) // self.__stride))

        for i in range(0, len(self.__averaged_frames) - steps, steps):
            window.append(self.__averaged_frames[i: i + steps * 2])
        return np.array(window)

    def __get_average_frame(self):
        """
        Computes the average frame of the video by taking the mean of consecutive frames.

        :returns:
            numpy.ndarray: The average frame of the video.

        Notes:
            This method divides the video frames into groups, each containing 'avg_no' frames.
            For each group, it computes the mean frame by averaging the pixel values of all frames in the group.
        """
        reduced_frames = []
        for i in range(0, len(self.__total_frames), self.__frame_average_count):
            frames = self.__total_frames[i:i + self.__frame_average_count]
            mean = np.mean(frames, axis=0)
            reduced_frames.append(mean)
        return np.array(reduced_frames)

    def __get_frames(self):
        """
        This method reads the video file and returns the frames

        :return:
            numpy.ndarray: An array containing processed frames from the video.
        :raises:
            IOError: If the video file cannot be read or does not exist.
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
        """
        This method makes the object iterable.
        :return: window: The object itself.
        """
        return self  # Return self to make the object iterable

    def __next__(self):
        """
        This method returns the next window in the windowed clip. (used for iteration)
        :return: numpy.ndarray: The next window in the windowed clip.
        """
        if self.has_next():
            self.__window_index += 1
            return self.__windows[self.__window_index - 1]
        else:
            raise StopIteration

    def get_current_window_stats(self):
        """
        This method returns the stats of the current window.

        :return:
            dict: A dictionary containing the statistics of the current window.
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


# This class takes a window object and returns the embeddings of the window using the CNN model.
# It maintains the sliding window of frames and extracts the embeddings of each frame using the CNN model.
class WindowEmbedded:
    def __init__(self, windows: Window):
        self.__windows = windows
        self.__embedding_model = CNN().to(device)
        self.window_embeddings = self.__get_embeddings()
        self.class_label = windows.class_label

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
