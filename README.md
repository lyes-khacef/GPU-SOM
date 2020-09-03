# GPU-SOM

GPU-based Self-Organizing Map with TensorFlow.

We propose a novel GPU-based implementation of the Self-Organzing Map (SOM) with TensorFlow. This simulator includes the training, labeling and test codes for fast experimentation of classification tasks based on post-labeled unsupervised learning.

The GPU-SOM is about 100 times faster than the classical CPU implementation. For example, a 1024 neurons SOM training on MNIST database exceeds 7000 images/s. It is also possible to visualize the SOM neurons weights at the end of training for visual assessment of the learning convergence.
