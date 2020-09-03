from cluster_algorithm import ClusterAlgorithm
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

class CustomKMeans(ClusterAlgorithm):
    """docstring for ClusterAlgorithm"""
    def __init__(self, k_means):
        self.k_means = k_means

    def get_segmentations(self, image):
        old_x, old_y = image.shape[0], image.shape[1]
        image = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
        k_segmentations = []
        for k in self.k_means:
            km = MiniBatchKMeans(k, batch_size=200)
            km.fit(image)
            segmentation = km.labels_.reshape(old_x, old_y)
            k_segmentations.append(segmentation)

        return k_segmentations
