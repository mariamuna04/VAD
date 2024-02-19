"""
USAGE: This generate_embedding.py is used to generate embeddings for a given folder with a specific structure

"""

from CreateEmbeddings import Dataset, Window

embeddings, labels = Dataset.create_dataset(
    '../dataset/',
    class_label_index=2,
    true_class_name='anomaly'
)

print("Embeddings Shape: ", embeddings.shape)
print("Labels Shape: ", labels.shape)

# This need to be checked
