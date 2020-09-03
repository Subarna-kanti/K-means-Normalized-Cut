import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
import metrics
from data_utils import get_truthes

class ClusterAlgorithm(object):
    """docstring for ClusterAlgorithm"""
    def __init__(self):
        pass

    def train(self, x_train, y_train, save_path=None, verbose=False):
        """save_path: if not None, segmentation results will be saved under specified folder
            verbose: set to true if you want to print details of the calculated performance
        """
        if (len(x_train) != len(y_train)):
            raise Exception("Unmatched data sizes: X={} vs y={}".format(len(x_train), len(y_train)))

        f_per_k = []
        c_per_k = []
        for i in range(len(x_train)):
            print("Image#{}".format(i+1))
            k_segmentations = self.get_segmentations(x_train[i])
            if save_path != None:
                self.save_to_path(k_segmentations, save_path)
            ground_truths = get_truthes(y_train[i])
            f_measures, cond_entropies = self.calc_performance(k_segmentations, ground_truths, verbose=verbose)
            f_per_k.append(f_measures)
            c_per_k.append(cond_entropies)

        f_average = np.array(f_per_k).mean(axis=0)
        c_average = np.array(c_per_k).mean(axis=0)
        for i in range(len(f_average)):
            k = self.k_means[i]
            f = f_average[i]
            c = c_average[i]
            print("K={}: Total average f-measure={:.4f}, Total average cond_entropies={:.4f}".format(k, f, c))

        return f_average, c_average 

    
    def get_segmentations(self, image, k_means):
        pass


    def calc_performance(self, segmentations, ground_truths, verbose=False):
        num_truthes = len(ground_truths)
        f_per_seg = []
        c_per_seg = []
        for i, seg in enumerate(segmentations):
            k = self.k_means[i]
            f = 0
            c=0
            for j in range(num_truthes):
                fm = metrics.f_measure(seg, ground_truths[j])
                cond_entropy = v_measure_score(seg.flat, ground_truths[j].flat)
                f += fm
                c += cond_entropy
                if verbose:
                    print("k:{} VS gorund-truth:{} => f={:.4f}, c={:.4f}".format(k,j+1, fm, cond_entropy))
            f /= num_truthes
            c /= num_truthes
            f_per_seg.append(f)
            c_per_seg.append(c)
            print("k={}: Avg. f-measue={:.4f} , Avg. conditional entropy={:.4f}\n".format(k, f, c))
        return f_per_seg, c_per_seg


    def save_to_path(self, path):
        pass
