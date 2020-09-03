import numpy as np

def f_measure(seg, grnd):
    clusters = np.unique(seg).tolist()
    F = 0
    for c in clusters:
        indeces = np.where(seg==c)
        # precision
        (values, counts) = np.unique(grnd[indeces], return_counts=True)
        ind = np.argmax(counts)
        max_val = values[ind]
        occurances = counts[ind]
        prec = occurances/ len(indeces[0])
        # recall
        rec = occurances / len(grnd[grnd==max_val])
        f = 2*prec*rec / (prec+rec)
        F += f
    F /= len(clusters)
    return F
