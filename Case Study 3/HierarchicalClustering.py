from sklearn.cluster import KMeans
import numpy as np
from scipy.ndimage import label

class DivisiveClustering:
    def __init__(self, n_clusters=2, n_init=10):
        self.n_clusters = n_clusters
        self.n_init = n_init

    def fit_predict(self, X, used_image):
        labels = np.zeros(len(X))
        current_cluster = 1
        augment_with_coordinates = True
        
        X_temp = X
        x_coords, y_coords = np.meshgrid(np.arange(used_image.shape[1]), np.arange(used_image.shape[0]))
        coords = np.stack((x_coords, y_coords), axis=-1).reshape((-1, 2))
        X_temp = np.concatenate((X_temp, coords), axis=1)
        
        while current_cluster < self.n_clusters:
            largest_cluster = self.find_largest_cluster(labels, used_image.shape)
            indices = np.where(labels == largest_cluster)            
            data_subset = X[indices]
            
            if current_cluster > 1:
                data_subset = X_temp[indices]
            
            kmeans = KMeans(n_clusters=2, n_init=self.n_init, random_state=42)
            new_labels = kmeans.fit_predict(data_subset)

            new_cluster_label = current_cluster + 1
            labels[indices] = new_labels + largest_cluster

            labels[labels == largest_cluster + 1] = new_cluster_label

            current_cluster += 1

        return labels



    def find_largest_cluster(self, labels, image_shape):
        reshaped_labels = labels.reshape(image_shape[:2])
        structure = np.ones((3, 3))
        labeled, ncomponents = label(reshaped_labels, structure)

        largest_component = 0
        max_size = 0
        for component in range(1, ncomponents + 1):
            component_size = np.sum(labeled == component)
            if component_size > max_size:
                max_size = component_size
                largest_component = component

        largest_cluster_label = reshaped_labels[np.where(labeled == largest_component)][0]
        return largest_cluster_label