# imports
from imports import *
from tensorflow.keras.datasets import mnist

# fuction to load data from file
def load_data(name): 
    data = np.load(name, allow_pickle=True)
    print("Data.shape = ", data.shape)
    return data

def get_dataset(train_data, label_data, test_data):
    # importing dataset
    (x_train_all, index_train_all), (x_test_all, index_test_all) = mnist.load_data()
    x_train_all = x_train_all.astype('float32') / 255.
    x_test_all = x_test_all.astype('float32') / 255.
    x_train_all = x_train_all.reshape((60000, 784))
    x_test_all = x_test_all.reshape((10000, 784))

    # constructing dataset
    x_train = np.copy(x_train_all[:train_data,:])
    index_train = np.copy(index_train_all[:train_data])
    x_label = np.copy(x_train_all[:label_data,:])
    index_label = np.copy(index_train_all[:label_data])
    x_test = np.copy(x_test_all[:test_data,:])
    index_test = np.copy(index_test_all[:test_data])

    return x_train, index_train, x_label, index_label, x_test, index_test, label_data