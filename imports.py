# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit

# import packages
import os, sys, humanize, psutil, GPUtil

# diable gpu
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# disable tensorflow logs
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# import tensorflow
import tensorflow as tf
print("Tensorflow version = ", tf.__version__)
if tf.executing_eagerly():
    print("Eager execution!")
#tf.debugging.set_log_device_placement(True)