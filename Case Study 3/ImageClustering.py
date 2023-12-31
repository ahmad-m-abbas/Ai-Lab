import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

class ClusteringAlgorithm(Enum):
    KMEANS = 'kmeans'
    GMM = 'gmm'
    DBSCAN = 'dbscan'

class EvaluationMetric(Enum):
    SILHOUETTE = 'silhouette'
    CALINSKI_HARABASZ = 'calinski_harabasz'
    DAVIES_BOULDIN = 'davies_bouldin'
    
class ImageClustering:
    def __init__(self, image_path, algorithm):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.algorithm = ClusteringAlgorithm[algorithm.upper()]
        self.resize_image

    def resize_image(self):
        pil_image = Image.fromarray(self.image)

        resized_pil_image = pil_image.resize((300, 200))

        self.image = np.array(resized_pil_image)
        
    def preprocess_image(self, use_standardization=True, use_normalization=False, augment_with_coordinates=False):
        pixels = self.image.reshape((-1, 3))

        if augment_with_coordinates:
            x_coords, y_coords = np.meshgrid(np.arange(self.image.shape[1]), np.arange(self.image.shape[0]))
            coords = np.stack((x_coords, y_coords), axis=-1).reshape((-1, 2))
            pixels = np.concatenate((pixels, coords), axis=1)

        if use_standardization:
            scaler = StandardScaler()
            pixels = scaler.fit_transform(pixels)
        elif use_normalization:
            normalizer = MinMaxScaler()
            pixels = normalizer.fit_transform(pixels)

        return pixels

    def cluster_image(self, n_clusters=None, dbscan_params={}, preprocess_kwargs={}):
        pixels = self.preprocess_image(**preprocess_kwargs)
        original_pixels = self.image.reshape((-1, 3))
        labels = None

        if self.algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(n_clusters=n_clusters, n_init='auto')
            labels = model.fit_predict(pixels)
        elif self.algorithm == ClusteringAlgorithm.GMM:
            model = GaussianMixture(n_components=n_clusters)
            labels = model.fit_predict(pixels)
        elif self.algorithm == ClusteringAlgorithm.DBSCAN:
            model = DBSCAN(**dbscan_params)
            labels = model.fit_predict(pixels) 


        cluster_colors = np.zeros((np.max(labels) + 1, 3), dtype=np.uint8)
        for label in np.unique(labels):
            cluster_colors[label] = np.mean(original_pixels[labels == label], axis=0)

        clustered_image = np.array([cluster_colors[label] for label in labels])
        clustered_image = clustered_image.reshape(self.image.shape)

        return clustered_image

    def evaluate_clustering(self, n_clusters=None, dbscan_params={}, preprocess_kwargs={}, metric=EvaluationMetric.SILHOUETTE):
        pixels = self.preprocess_image(**preprocess_kwargs)
        labels = None

        # Cluster the image
        if self.algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(n_clusters=n_clusters, n_init='auto')
        elif self.algorithm == ClusteringAlgorithm.GMM:
            model = GaussianMixture(n_components=n_clusters)
        elif self.algorithm == ClusteringAlgorithm.DBSCAN:
            model = DBSCAN(**dbscan_params)

        labels = model.fit_predict(pixels)

        print('Start Evaluation:')
        print('Number of labels:', len(labels))
        print('Number of Pixels:', pixels.shape)
        if metric == EvaluationMetric.SILHOUETTE:
            score = silhouette_score(pixels, labels) if np.unique(labels).size > 1 else None
        elif metric == EvaluationMetric.CALINSKI_HARABASZ:
            score = calinski_harabasz_score(pixels, labels) if np.unique(labels).size > 1 else None
        elif metric == EvaluationMetric.DAVIES_BOULDIN:
            score = davies_bouldin_score(pixels, labels) if np.unique(labels).size > 1 else None

        return score

    def hyperparameter_tuning(self, param_grid, preprocess_kwargs={}, metric=EvaluationMetric.SILHOUETTE):
        results = []
        for i, params in enumerate(param_grid):
            print(f"Processing {i+1}/{len(param_grid)}: {params}")
            if self.algorithm in [ClusteringAlgorithm.KMEANS, ClusteringAlgorithm.GMM]:
                n_clusters = params.get('n_clusters', None)
                score = self.evaluate_clustering(n_clusters=n_clusters, preprocess_kwargs=preprocess_kwargs, metric=metric)
            elif self.algorithm == ClusteringAlgorithm.DBSCAN:
                score = self.evaluate_clustering(dbscan_params=params, preprocess_kwargs=preprocess_kwargs, metric=metric)
    
            results.append((params, score))
    
        return results

    def find_best_params(self, tuning_results):
        valid_results = [result for result in tuning_results if result[1] is not None]
    
        if not valid_results:
            raise ValueError("No valid results found. All scores are None.")
    
        best_params, _ = min(valid_results, key=lambda x: x[1])
        return best_params


    def display_clustered_image(self, clustered_image):
        plt.imshow(clustered_image)
        plt.axis('off')
        plt.show()
