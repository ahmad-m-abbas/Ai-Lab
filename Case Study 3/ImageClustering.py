import numpy as np
import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from HierarchicalClustering import DivisiveClustering


class ClusteringAlgorithm(Enum):
    KMEANS = 'kmeans'
    GMM = 'gmm'
    AGG = 'agg'
    HC = 'hc'

class EvaluationMetric(Enum):
    SILHOUETTE = 'silhouette'
    DAVIES_BOULDIN = 'davies_bouldin'
    
class ImageClustering:
    def __init__(self, image_path, algorithm):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.resized_image = self.resize_image()
        self.algorithm = ClusteringAlgorithm[algorithm.upper()]
        

    def resize_image(self):
        pil_image = Image.fromarray(self.image)

        resized_pil_image = pil_image.resize((128, 128))

        return np.array(resized_pil_image)
        
    def preprocess_image(self, use_standardization=True, use_normalization=False, augment_with_coordinates=False, resized=False):
        
        if resized:   
            used_image = self.resized_image
        else:
            used_image = self.image
            
        pixels = used_image.reshape((-1, 3))


        if augment_with_coordinates:
            x_coords, y_coords = np.meshgrid(np.arange(used_image.shape[1]), np.arange(used_image.shape[0]))
            coords = np.stack((x_coords, y_coords), axis=-1).reshape((-1, 2))
            pixels = np.concatenate((pixels, coords), axis=1)

        if use_standardization:
            scaler = StandardScaler()
            pixels = scaler.fit_transform(pixels)
        elif use_normalization:
            normalizer = MinMaxScaler()
            pixels = normalizer.fit_transform(pixels)

        return pixels

    def cluster_image(self, n_clusters=None, preprocess_kwargs={}):
        if self.algorithm == ClusteringAlgorithm.AGG or self.algorithm == ClusteringAlgorithm.HC:
            pixels = self.preprocess_image(**preprocess_kwargs, resized=True)
            used_image = self.resized_image
        else:
            pixels = self.preprocess_image(**preprocess_kwargs)
            used_image = self.image
        
        original_pixels = used_image.reshape((-1, 3))
            
        labels = None

        if self.algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(n_clusters=n_clusters, n_init='auto')
            labels = model.fit_predict(pixels)
        elif self.algorithm == ClusteringAlgorithm.GMM:
            model = GaussianMixture(n_components=n_clusters)
            labels = model.fit_predict(pixels)
        elif self.algorithm == ClusteringAlgorithm.AGG:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            labels = model.fit_predict(pixels)
        else:
            model = DivisiveClustering(n_clusters=n_clusters, n_init='auto')
            labels = model.fit_predict(pixels, used_image)
        
        

        # Chatgpt colors
        colors = { 0: [255, 0, 0], 1: [0, 0, 255], 2: [255, 255, 0], 3: [0, 255, 0], 4: [255, 165, 0], 5: [128, 0, 128], 6: [0, 255, 255], 
                   7: [132, 248, 207], 8: [111, 71, 144], 9: [75, 158, 50], 10: [37, 169, 241], 11: [161, 104, 244], 12: [0, 252, 170], 13: [72, 229, 46],
                  14: [55, 154, 149], 5: [147, 227, 46], 16: [197, 162, 123], 17: [66, 76, 19], 18: [190, 87, 170], 19: [37, 13, 63], 20: [94, 63, 245],
                  21: [154, 179, 223], 22: [240, 86, 104], 23: [29, 81, 82], 24: [175, 128, 60]
        }

        labels = labels.astype(int)
        cluster_colors = np.zeros((np.max(labels) + 1, 3), dtype=np.uint8)

        for label in np.unique(labels):
            if label != -1:
                cluster_colors[label] = colors[label]
            else:
                cluster_colors[label] = [255, 255, 255]

        clustered_image = np.array([cluster_colors[label] for label in labels])
        clustered_image = clustered_image.reshape(used_image.shape)

        for label in np.unique(labels):
            cluster_colors[label] = np.mean(original_pixels[labels == label], axis=0)
        colored_cluster = np.array([cluster_colors[label] for label in labels])
        colored_cluster = colored_cluster.reshape(used_image.shape)
        
        return clustered_image, colored_cluster

    def evaluate_clustering(self, n_clusters=None, preprocess_kwargs={}, metric=EvaluationMetric.SILHOUETTE):
        if metric == EvaluationMetric.SILHOUETTE:
            pixels = self.preprocess_image(**preprocess_kwargs, resized=True)
            used_image = self.resized_image
        else:
            pixels = self.preprocess_image(**preprocess_kwargs)
            used_image = self.image
            
        labels = None

        if self.algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(n_clusters=n_clusters, n_init='auto')
            labels = model.fit_predict(pixels)
        elif self.algorithm == ClusteringAlgorithm.GMM:
            model = GaussianMixture(n_components=n_clusters)
            labels = model.fit_predict(pixels)
        elif self.algorithm == ClusteringAlgorithm.AGG:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            labels = model.fit_predict(pixels)
        else:
            model = DivisiveClustering(n_clusters=n_clusters, n_init='auto')
            labels = model.fit_predict(pixels, used_image)
            
        
                                    
        if metric == EvaluationMetric.SILHOUETTE:
            score = silhouette_score(pixels, labels) if np.unique(labels).size > 1 else None
        elif metric == EvaluationMetric.DAVIES_BOULDIN:
            score = davies_bouldin_score(pixels, labels) if np.unique(labels).size > 1 else None
        return score
    