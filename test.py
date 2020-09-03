# imports
from imports import *

def test(class_nbr, weights, x_test, index_test, neuron_label, sigma):
    neuron_index = np.zeros((len(x_test),), dtype=int)

    # calculate the BMUs for the test dataset
    for i, x in enumerate(x_test):
        dist_neuron = np.linalg.norm(x - weights, axis=1)
        neuron_index[i] = np.argmin(dist_neuron)
    
    # compare the BMUs labels and the samples labels
    accuracy = 0
    for p, t in zip(neuron_index, index_test):
        if neuron_label[p] == t:
            accuracy += 1
    accuracy = (float(accuracy)/len(x_test))*100
    print("SOM test accuracy = %.2f\n" % accuracy)

    return accuracy