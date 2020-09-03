# imports
from imports import *
from test import *

def labeling(label_data, class_nbr, weights, x_label, index_label, sigma):    
    dist_sum = np.zeros((len(weights), class_nbr))
    nbr_digits = np.zeros((class_nbr,))

    # accumulate the normalized gaussian distance for the labeling dataset
    for (x, y) in zip(x_label, index_label):
        nbr_digits[y] += 1
        dist_neuron = np.exp(-np.linalg.norm(x - weights, axis=1)/sigma)
        dist_bmu = np.max(dist_neuron)
        for i, distn in enumerate(dist_neuron):
            dist_sum[i][y] += distn/dist_bmu

    # normalize the activities on the number of samples per class
    for i, dists in enumerate(dist_sum):
        dist_sum[i] = dists/nbr_digits

    # assign the neurons labels
    neuron_label = np.argmax(dist_sum, axis=1)
    print("Neurons labels = ")
    print(neuron_label)

    return neuron_label

def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def run_labeling(weights, label_data, x_tr, index_tr, x_ts, index_ts):
    class_nbr = 10
    sigma_kernel = 1.0
    for i in range(10):
        x_tr, index_tr = unison_shuffle(x_tr, index_tr)
        x_lb = np.copy(x_tr[:label_data,:])
        index_lb = np.copy(index_tr[:label_data])

        # label the network
        neuron_label = labeling(label_data, class_nbr, weights, x_lb, index_lb, sigma_kernel)

        # test the network
        accuracy = test(class_nbr, weights, x_ts, index_ts, neuron_label, sigma_kernel)