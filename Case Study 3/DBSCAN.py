import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from skimage import io
from PIL import Image
from itertools import product
from kneed import KneeLocator
class ImageSegmentation:
    def __init__(self, image_path, resize_dims=(150, 150), augment_with_coordinates=False):
        self.image_path = image_path
        self.resize_dims = resize_dims
        self.image = self.load_and_preprocess_image(image_path, augment_with_coordinates=augment_with_coordinates)
        self.eps = None
        self.min_samples = None

    def load_and_preprocess_image(self, image_path, augment_with_coordinates=False):
        image = io.imread(image_path)
        pil_image = Image.fromarray(image)
            
        resized_pil_image = pil_image.resize(self.resize_dims)
        image_array = np.array(resized_pil_image).reshape(-1, 3)

        if augment_with_coordinates:
            x_coords, y_coords = np.meshgrid(np.arange(self.resize_dims[1]), np.arange(self.resize_dims[0]))
            coords = np.stack((x_coords, y_coords), axis=-1).reshape((-1, 2))
            image_array = np.concatenate((image_array, coords), axis=1)
            
        scaler = StandardScaler()
        image_normalized = scaler.fit_transform(image_array)
        return image_normalized

    def estimate_eps_using_derivative(self, show_image=False, n_neighbors=2):
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(self.image)
        distances, indices = nbrs.kneighbors(self.image)
        distances = np.sort(distances, axis=0)[:, 1]

        derivatives = np.gradient(distances)

        max_derivative_index = np.argmax(derivatives)

        estimated_eps = distances[max_derivative_index]

        if show_image:
            plt.figure(figsize=(14, 6))
            plt.plot(distances, label='K-distance')
            plt.plot(derivatives, label='Derivative', linestyle='--')
            plt.axvline(x=max_derivative_index, color='r', linestyle=':', label='Max Derivative (elbow)')
            plt.xlabel('Points sorted by distance')
            plt.ylabel('Distance to 2nd nearest neighbor / Derivative')
            plt.title('k-Distance Graph and Derivative')
            plt.legend()
            plt.show()

        return estimated_eps
    
    def estimate_eps_with_kneelocator(self, neighbors=6, show=False):
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(self.image)
        distances, indices = nbrs.kneighbors(self.image)
        distances = distances[:, neighbors - 1]
        distances = sorted(distances, reverse=True)
        
        kneedle = KneeLocator(range(1, len(distances) + 1), distances, curve='convex', direction='decreasing', S=1.0)
        if show:
            kneedle.plot_knee_normalized()

        estimated_eps = kneedle.knee_y
        print(f"The estimated eps is: {estimated_eps}")
        
        return estimated_eps
