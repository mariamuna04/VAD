"""
Video Embedding Clustering Module for Video Anomaly Detection (VAD)

This module provides K-means clustering functionality for video embeddings.
It trains clusters using true labels and embeddings, then creates cluster centers
for enhanced classification in the VAD pipeline.

USAGE: Use this module to create clusters from video embeddings for improved classification
Author: Generated for VAD Project
"""

import os
import argparse
import numpy as np
import torch
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc


class VideoEmbeddingClusterer:
    """
    K-means clustering for video embeddings with multiple similarity metrics.
    """
    
    def __init__(self, 
                 n_clusters: int = 12,
                 random_state: int = 42,
                 max_iter: int = 300,
                 normalize_embeddings: bool = True,
                 use_pca: bool = False,
                 pca_components: int = 512):
        """
        Initialize the clustering module.
        
        Args:
            n_clusters: Number of clusters (should match number of classes)
            random_state: Random state for reproducibility
            max_iter: Maximum iterations for K-means
            normalize_embeddings: Whether to normalize embeddings before clustering
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.normalize_embeddings = normalize_embeddings
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # Initialize components
        self.kmeans = None
        self.scaler = None
        self.pca = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.embedding_shape = None
        
        # Category mapping for multiclass
        self.category_names = [
            "Normal", "Abuse", "Arson", "Assault", "Road Accident", "Burglary", 
            "Explosion", "Fighting", "Robbery", "Shooting", "Stealing", "Vandalism"
        ]
        
        # Statistics
        self.training_stats = {}
        
        print(f"VideoEmbeddingClusterer initialized with {n_clusters} clusters")
    
    def prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Prepare embeddings for clustering by reshaping and normalizing.
        
        Args:
            embeddings: Video embeddings of shape (n_videos, n_frames, n_features)
            
        Returns:
            Processed embeddings ready for clustering
        """
        print(f"Original embeddings shape: {embeddings.shape}")
        
        # Store original shape for later use
        self.embedding_shape = embeddings.shape
        
        # Flatten the embeddings (n_videos, n_frames * n_features)
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        print(f"Flattened embeddings shape: {flattened_embeddings.shape}")
        
        # Apply normalization if requested
        if self.normalize_embeddings:
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized_embeddings = self.scaler.fit_transform(flattened_embeddings)
                print("Applied StandardScaler normalization")
            else:
                normalized_embeddings = self.scaler.transform(flattened_embeddings)
        else:
            normalized_embeddings = flattened_embeddings
        
        # Apply PCA if requested
        if self.use_pca:
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                pca_embeddings = self.pca.fit_transform(normalized_embeddings)
                print(f"Applied PCA: {normalized_embeddings.shape} -> {pca_embeddings.shape}")
                print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
            else:
                pca_embeddings = self.pca.transform(normalized_embeddings)
            
            return pca_embeddings
        
        return normalized_embeddings
    
    def train_clusters(self, 
                      embeddings: np.ndarray, 
                      labels: np.ndarray) -> Dict[str, Any]:
        """
        Train K-means clusters using embeddings and true labels.
        
        Args:
            embeddings: Video embeddings of shape (n_videos, n_frames, n_features)
            labels: True labels for videos
            
        Returns:
            Dictionary containing training statistics and results
        """
        print("Starting cluster training...")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
        # Prepare embeddings
        processed_embeddings = self.prepare_embeddings(embeddings)
        
        # Initialize and train K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10
        )
        
        print("Training K-means clustering...")
        cluster_predictions = self.kmeans.fit_predict(processed_embeddings)
        
        # Store cluster centers and labels
        self.cluster_centers = self.kmeans.cluster_centers_
        self.cluster_labels = cluster_predictions
        
        # Calculate training statistics
        silhouette_avg = silhouette_score(processed_embeddings, cluster_predictions)
        ari_score = adjusted_rand_score(labels, cluster_predictions)
        inertia = self.kmeans.inertia_
        
        # Analyze cluster-label correspondence
        cluster_label_analysis = self._analyze_cluster_label_correspondence(
            cluster_predictions, labels
        )
        
        # Create label-to-cluster mapping
        label_to_cluster_map = self._create_label_cluster_mapping(
            cluster_predictions, labels
        )
        
        # Store training statistics
        self.training_stats = {
            'silhouette_score': silhouette_avg,
            'adjusted_rand_index': ari_score,
            'inertia': inertia,
            'n_clusters': self.n_clusters,
            'n_samples': len(embeddings),
            'cluster_label_analysis': cluster_label_analysis,
            'label_to_cluster_map': label_to_cluster_map
        }
        
        print(f"Clustering completed!")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Adjusted Rand Index: {ari_score:.4f}")
        print(f"Inertia: {inertia:.2f}")
        
        return self.training_stats
    
    def _analyze_cluster_label_correspondence(self, 
                                           cluster_predictions: np.ndarray, 
                                           true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the correspondence between clusters and true labels.
        
        Args:
            cluster_predictions: Predicted cluster labels
            true_labels: True class labels
            
        Returns:
            Dictionary with correspondence analysis
        """
        analysis = {}
        
        # Create confusion matrix between clusters and labels
        confusion_matrix = np.zeros((self.n_clusters, len(self.category_names)))
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_predictions == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                for label in cluster_true_labels:
                    if label < len(self.category_names):
                        confusion_matrix[cluster_id, label] += 1
        
        analysis['confusion_matrix'] = confusion_matrix
        
        # Find best label for each cluster
        cluster_best_labels = {}
        for cluster_id in range(self.n_clusters):
            best_label = np.argmax(confusion_matrix[cluster_id, :])
            confidence = confusion_matrix[cluster_id, best_label] / np.sum(confusion_matrix[cluster_id, :])
            cluster_best_labels[cluster_id] = {
                'best_label': int(best_label),
                'confidence': float(confidence),
                'category_name': self.category_names[best_label]
            }
        
        analysis['cluster_best_labels'] = cluster_best_labels
        
        return analysis
    
    def _create_label_cluster_mapping(self, 
                                    cluster_predictions: np.ndarray, 
                                    true_labels: np.ndarray) -> Dict[int, int]:
        """
        Create a mapping from true labels to their most common cluster.
        
        Args:
            cluster_predictions: Predicted cluster labels
            true_labels: True class labels
            
        Returns:
            Dictionary mapping label -> most_common_cluster
        """
        label_cluster_map = {}
        
        for label in range(len(self.category_names)):
            label_mask = true_labels == label
            if np.sum(label_mask) > 0:
                label_clusters = cluster_predictions[label_mask]
                most_common_cluster = np.bincount(label_clusters).argmax()
                label_cluster_map[label] = int(most_common_cluster)
        
        return label_cluster_map
    
    def compute_class_centers(self, 
                            embeddings: np.ndarray, 
                            labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute centers for each class using true labels.
        
        Args:
            embeddings: Video embeddings
            labels: True labels
            
        Returns:
            Dictionary mapping class_id -> class_center
        """
        print("Computing class centers...")
        
        # Prepare embeddings
        processed_embeddings = self.prepare_embeddings(embeddings)
        
        class_centers = {}
        
        for class_id in range(len(self.category_names)):
            class_mask = labels == class_id
            if np.sum(class_mask) > 0:
                class_embeddings = processed_embeddings[class_mask]
                class_center = np.mean(class_embeddings, axis=0)
                class_centers[class_id] = class_center
                print(f"Class {class_id} ({self.category_names[class_id]}): {np.sum(class_mask)} samples")
            else:
                print(f"Warning: No samples found for class {class_id} ({self.category_names[class_id]})")
        
        return class_centers
    
    def predict_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: Video embeddings to cluster
            
        Returns:
            Cluster predictions
        """
        if self.kmeans is None:
            raise ValueError("Model not trained. Call train_clusters first.")
        
        processed_embeddings = self.prepare_embeddings(embeddings)
        return self.kmeans.predict(processed_embeddings)
    
    def compute_cosine_similarities(self, 
                                  embedding: np.ndarray, 
                                  centers: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Compute cosine similarities between an embedding and class centers.
        
        Args:
            embedding: Single embedding vector
            centers: Dictionary of class centers
            
        Returns:
            Dictionary mapping class_id -> cosine_similarity
        """
        similarities = {}
        
        for class_id, center in centers.items():
            similarity = cosine_similarity([embedding], [center])[0][0]
            similarities[class_id] = float(similarity)
        
        return similarities
    
    def visualize_clusters(self, 
                          embeddings: np.ndarray, 
                          labels: np.ndarray, 
                          save_path: Optional[str] = None,
                          method: str = 'pca') -> None:
        """
        Visualize clusters using dimensionality reduction.
        
        Args:
            embeddings: Video embeddings
            labels: True labels
            save_path: Path to save the plot
            method: Dimensionality reduction method ('pca' or 'tsne')
        """
        print(f"Visualizing clusters using {method.upper()}...")
        
        # Prepare embeddings
        processed_embeddings = self.prepare_embeddings(embeddings)
        
        # Apply dimensionality reduction for visualization
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        
        embeddings_2d = reducer.fit_transform(processed_embeddings)
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        # Define colors for each category
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.category_names)))
        
        # Plot points for each category
        for class_id in range(len(self.category_names)):
            class_mask = labels == class_id
            if np.sum(class_mask) > 0:
                plt.scatter(
                    embeddings_2d[class_mask, 0], 
                    embeddings_2d[class_mask, 1],
                    label=self.category_names[class_id],
                    alpha=0.6,
                    s=50,
                    color=colors[class_id]
                )
        
        # Plot cluster centers if available
        if self.cluster_centers is not None:
            if method.lower() == 'pca':
                centers_2d = reducer.transform(self.cluster_centers)
            else:
                # For t-SNE, we need to include centers in the original transformation
                combined_data = np.vstack([processed_embeddings, self.cluster_centers])
                combined_2d = reducer.fit_transform(combined_data)
                centers_2d = combined_2d[-len(self.cluster_centers):]
            
            plt.scatter(
                centers_2d[:, 0], 
                centers_2d[:, 1], 
                marker='x', 
                color='red', 
                s=200, 
                linewidth=3,
                label='Cluster Centers'
            )
        
        plt.title(f'Video Embedding Clusters ({method.upper()} Visualization)')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained clustering model and components.
        
        Args:
            save_path: Path to save the model
        """
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_centers': self.cluster_centers,
            'cluster_labels': self.cluster_labels,
            'embedding_shape': self.embedding_shape,
            'training_stats': self.training_stats,
            'config': {
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'max_iter': self.max_iter,
                'normalize_embeddings': self.normalize_embeddings,
                'use_pca': self.use_pca,
                'pca_components': self.pca_components
            },
            'category_names': self.category_names
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained clustering model.
        
        Args:
            load_path: Path to load the model from
        """
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.cluster_centers = model_data['cluster_centers']
        self.cluster_labels = model_data['cluster_labels']
        self.embedding_shape = model_data['embedding_shape']
        self.training_stats = model_data['training_stats']
        self.category_names = model_data['category_names']
        
        # Update config
        config = model_data['config']
        self.n_clusters = config['n_clusters']
        self.random_state = config['random_state']
        self.max_iter = config['max_iter']
        self.normalize_embeddings = config['normalize_embeddings']
        self.use_pca = config['use_pca']
        self.pca_components = config['pca_components']
        
        print(f"Model loaded from: {load_path}")
        print(f"Loaded model with {self.n_clusters} clusters")


def load_embeddings_and_labels(embeddings_path: str, 
                              labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and labels from numpy files.
    
    Args:
        embeddings_path: Path to embeddings .npy file
        labels_path: Path to labels .npy file
        
    Returns:
        Tuple of (embeddings, labels)
    """
    print(f"Loading embeddings from: {embeddings_path}")
    print(f"Loading labels from: {labels_path}")
    
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    return embeddings, labels


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Video Embedding Clustering for VAD")
    
    # Data parameters
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to embeddings numpy file')
    parser.add_argument('--labels_path', type=str, required=True,
                       help='Path to labels numpy file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save clustering results')
    
    # Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=12,
                       help='Number of clusters')
    parser.add_argument('--max_iter', type=int, default=300,
                       help='Maximum iterations for K-means')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # Preprocessing parameters
    parser.add_argument('--no_normalize', action='store_true',
                       help='Disable embedding normalization')
    parser.add_argument('--use_pca', action='store_true',
                       help='Apply PCA for dimensionality reduction')
    parser.add_argument('--pca_components', type=int, default=512,
                       help='Number of PCA components')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                       help='Create cluster visualizations')
    parser.add_argument('--vis_method', type=str, default='pca',
                       choices=['pca', 'tsne'],
                       help='Visualization method')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    embeddings, labels = load_embeddings_and_labels(args.embeddings_path, args.labels_path)
    
    # Initialize clusterer
    clusterer = VideoEmbeddingClusterer(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        max_iter=args.max_iter,
        normalize_embeddings=not args.no_normalize,
        use_pca=args.use_pca,
        pca_components=args.pca_components
    )
    
    # Train clusters
    print("\n" + "="*50)
    print("TRAINING CLUSTERS")
    print("="*50)
    training_stats = clusterer.train_clusters(embeddings, labels)
    
    # Compute class centers
    print("\n" + "="*50)
    print("COMPUTING CLASS CENTERS")
    print("="*50)
    class_centers = clusterer.compute_class_centers(embeddings, labels)
    
    # Save results
    model_path = os.path.join(args.output_dir, 'clustering_model.pkl')
    clusterer.save_model(model_path)
    
    # Save class centers separately
    centers_path = os.path.join(args.output_dir, 'class_centers.npy')
    centers_dict_path = os.path.join(args.output_dir, 'class_centers.pkl')
    
    # Convert class centers to array for numpy save
    centers_array = np.array([class_centers.get(i, np.zeros(clusterer.cluster_centers.shape[1])) 
                             for i in range(len(clusterer.category_names))])
    np.save(centers_path, centers_array)
    
    # Save as dictionary for easy loading
    with open(centers_dict_path, 'wb') as f:
        pickle.dump(class_centers, f)
    
    print(f"Class centers saved to: {centers_path}")
    print(f"Class centers dict saved to: {centers_dict_path}")
    
    # Save training statistics
    stats_path = os.path.join(args.output_dir, 'clustering_stats.json')
    with open(stats_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_stats = {}
        for key, value in training_stats.items():
            if isinstance(value, np.ndarray):
                json_stats[key] = value.tolist()
            elif isinstance(value, dict):
                json_stats[key] = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) 
                                  for k, v in value.items()}
            else:
                json_stats[key] = value
        json.dump(json_stats, f, indent=2)
    
    print(f"Training statistics saved to: {stats_path}")
    
    # Create visualizations if requested
    if args.visualize:
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        vis_path = os.path.join(args.output_dir, f'cluster_visualization_{args.vis_method}.png')
        clusterer.visualize_clusters(embeddings, labels, vis_path, args.vis_method)
    
    print("\n" + "="*50)
    print("CLUSTERING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Results saved in: {args.output_dir}")
    print(f"Silhouette Score: {training_stats['silhouette_score']:.4f}")
    print(f"Adjusted Rand Index: {training_stats['adjusted_rand_index']:.4f}")


if __name__ == "__main__":
    main()
