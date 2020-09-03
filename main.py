# -*- coding: utf-8 -*-

# ####################################################################################################
# GPU-based Self-Organizing-Map by Lyes Khacef.
# Reference: L. Khacef, V. Gripon, and B. Miramond, “GPU-based self-organizing-maps 
# for post-labeled few-shot unsupervised learning”, in International Conference On 
# Neural Information Processing (ICONIP), 2020.
# ####################################################################################################

# imports
from imports import *
from gpu_check import *
from data_load import *
from tf_ksom import *
from label import *
from test import *

# hyper-parameters
train_data = 60000
label_data = 600
test_data = 10000
input_dim = 784
map_wth = 10
map_hgt = 10
class_nbr = 10
nbr_epochs = 10
eps_i_list = [1.0]
eps_f_list = [0.01]
eta_i_list = [10.0]
eta_f_list = [0.01]
sigma_kernel = 1.0

# GPU name
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError("GPU device not found!")
print('Found GPU at: {}'.format(device_name))

# load dataset
x_train, index_train, x_label, index_label, x_test, index_test, label_data = get_dataset(train_data, label_data, test_data)

def run_som(eps_i, eps_f, eta_i, eta_f):
    print("\nHyper-parameters:   # eps_i = %f   # eps_f = %f   # eta_i = %f   # eta_f = %f" % (eps_i, eps_f, eta_i, eta_f))
    # train the network
    som = KSOM(map_wth, map_hgt, input_dim)
    start_time = timeit.default_timer()
    som.train(nbr_epochs, eps_i, eps_f, eta_i, eta_f, x_train, x_label, index_label, x_test, index_test)
    end_time = timeit.default_timer()
    print("\nSOM training time = ", end_time - start_time)
    weights = som.get_weights().numpy()

    # save the weights
    np.save("weights/som_weights.npy", weights)
    
    # label the network
    neuron_label = labeling(label_data, class_nbr, weights, x_label, index_label, sigma_kernel)

    # test the network
    accuracy = test(class_nbr, weights, x_test, index_test, neuron_label, sigma_kernel)

    return weights, accuracy

# hyper-parameters grid search
hyper_param_list, accuracy_list = [], []
for eps_i in eps_i_list:
    for eps_f in eps_f_list:
        for eta_i in eta_i_list:
            for eta_f in eta_f_list:
                hyper_param_list.append([eps_i, eps_f, eta_i, eta_f])
                weights, accuracy = run_som(eps_i, eps_f, eta_i, eta_f)
                accuracy_list.append(accuracy)

# best hyper-parameters
best_accuracy = np.max(accuracy_list)
best_hyper_param = hyper_param_list[np.argmax(accuracy_list)]
print("Best accuracy = ", best_accuracy)
print("Best hyper-parameters:   # eps_i = %f   # eps_f = %f   # sig_i = %f   # sig_f = %f" % (best_hyper_param[0], best_hyper_param[1], best_hyper_param[2], best_hyper_param[3]))

label_list = [label_data]
for label_data in label_list:
    print("\n---------- Labels = %d ----------" % label_data)
    run_labeling(weights, label_data, x_train, index_train, x_test, index_test)

# GPU memory check
gpu_report()

# display neurons weights as mnist digits
som_grid = plt.figure(figsize=(10, 10)) # width, height in inches
for n in range(map_wth*map_hgt):
    image = weights[n].reshape([28,28]) # x_train[num] is the 784 normalized pixel values
    sub = som_grid.add_subplot(map_wth, map_hgt, n + 1)
    sub.set_axis_off()
    clr = sub.imshow(image, cmap = plt.get_cmap("jet"), interpolation = "nearest")
    #plt.colorbar(clr)
plt.savefig("plots/som_weights.png")
#plt.show()