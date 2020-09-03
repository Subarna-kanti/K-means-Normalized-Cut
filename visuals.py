import matplotlib.pyplot as plt
import numpy as np
from data_utils import get_truthes

def plot_img_with_gnd(image, img, grnd_truth, size=(30,20)):
    fig = plt.figure(figsize=size)
    ground_truthes = get_truthes(grnd_truth)
    num_columns = len(ground_truthes) + 2
    fig.add_subplot(1, num_columns, 1)
    plt.imshow(image, cmap = 'gray')
    plt.axis('off')
    fig.add_subplot(1, num_columns, 2)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    for i in range(len(ground_truthes)):
        ax = fig.add_subplot(1, num_columns, i+3)
        ax.set_title("Ground truth_{}".format(i+1))
        # plt.imshow(ground_truthes[i], cmap='Greys_r')
        plt.imshow(ground_truthes[i], cmap = 'gray')
        plt.axis('off')


def plot_images_truths(images, img, ground_truth, size=(30,20)):
    for i in range(len(images)):
        plot_img_with_gnd(images[i], img[i], ground_truth[i], size)


def plot_k_performance(k_means, f_average, c_average, normalized=False, title=""):
    # data to plot
    n_groups = len(k_means)
    f_measures = f_average
    cond_entropies = c_average
    
    if normalized:
        f_measures = (np.array(f_measures) - np.min(f_measures))/(np.max(f_measures) - np.min(f_measures))
        cond_entropies = (np.array(cond_entropies) - np.min(cond_entropies))/(np.max(cond_entropies) - np.min(cond_entropies))
        title += " (Normalized)"
    else:
        title += " (Not Normalized)"
    # create plot
    fig, ax = plt.subplots(figsize=(12,8))
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
     
    rects1 = plt.bar(index, f_measures, bar_width,
                     alpha=opacity,
                     color='b',
                     label='f_measure')
     
    rects2 = plt.bar(index + bar_width, cond_entropies, bar_width,
                     alpha=opacity,
                     color='g',
                     label='cond. entropy')
     
    plt.xlabel('K')
    plt.ylabel('Scores')
    plt.title(title)
    plt.xticks(index + bar_width, tuple(k_means) )
    plt.legend()
     
    plt.tight_layout()
    plt.show()
