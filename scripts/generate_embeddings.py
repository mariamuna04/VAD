"""
USAGE: This generate_embedding.py is used to generate embeddings for a given folder with a specific structure

"""

from CreateEmbeddings import Dataset, Window

x = Window.Window('../dataset/anomaly/10_Arson002_x264_010.mp4',
                  class_label_index=2, true_class_name='anomaly')

print(x.class_label)

embeddings, labels = Dataset.create_dataset(
    '../dataset/',
    class_label_index=2,
    true_class_name='anomaly'
)

# This need to be checked
