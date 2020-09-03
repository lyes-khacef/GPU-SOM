# imports
from imports import *

class KSOM():
    def __init__(self, m, n, dim):
        self.m = m
        self.n = n
        self.dim = dim
        self.map_wgt =  tf.Variable(
                            tf.random.uniform(
                                shape = [m*n, dim],
                                minval = 0.0,
                                maxval = 1.0,
                                dtype = tf.float32,
                                seed = 23
                            )
                        )
        self.map_loc =  tf.constant(
                            np.array(
                                list(self.neuron_locs(m, n))
                            )
                        )

    def neuron_locs(self, m, n):
        # nested iterations over both dimensions to yield one by one the 2-d locations of the individual neurons in the SOM
        for i in range(m):
            for j in range(n):
                yield np.array([i,j], dtype=np.float32)
    
    def compute_winner(self, sample):
            self.sample = sample

            # compute the squared euclidean distance between the input and the neurons
            self.squared_distance = tf.reduce_sum(
                                        tf.square(
                                            tf.subtract(
                                                self.map_wgt, # [m*n, dim]
                                                tf.expand_dims(
                                                    self.sample, # [dim] -> [1, dim]
                                                    axis=0
                                                )
                                            )
                                        ), 
                                        axis=1
                                    )
            
            # find the bmu's index
            self.bmu_idx =  tf.argmin(
                                    input=self.squared_distance, 
                                    axis=0
                                )
            
            # extract the bmu's 2-d location
            self.bmu_loc =  tf.gather(
                                self.map_loc, 
                                self.bmu_idx
                            )
    
    def update_network(self, epsilon, eta):
        # compute the squared manhattan distance between the bmu and the neurons
        self.bmu_distance_squares = tf.reduce_sum(
                                        tf.square(
                                            tf.subtract(
                                                self.map_loc, # [m*n, 2]
                                                tf.expand_dims(
                                                    self.bmu_loc, # [2] -> [1, 2]
                                                    axis=0
                                                )
                                            )
                                        ), 
                                        axis=1
                                    )

        # compute the neighborhood function
        self.neighbourhood_func = tf.exp(
                                      tf.negative(
                                          tf.math.divide(
                                              self.bmu_distance_squares,
                                              tf.multiply(
                                                  tf.square(
                                                      eta,
                                                  ),
                                                  2.0
                                              )
                                          )
                                      )
                                  )

        # compute the overall learning of each neuron
        self.learning = tf.multiply(
                            self.neighbourhood_func, 
                            epsilon
                        )
        
        # compute the difference between the neurons weights and the input
        self.delta_wgt =  tf.subtract(
                              tf.expand_dims(
                                  self.sample, # [dim] -> [1, dim]
                                  axis=0
                              ),
                              self.map_wgt, # [m*n, dim]
                          )

        # compute the weights update according to the learning and delta_wgt and update the weights
        tf.compat.v1.assign_add(
            self.map_wgt,
            tf.multiply(
                tf.expand_dims(
                    self.learning, # [m*n] -> [m*n, 1]
                    axis=-1
                ),
                self.delta_wgt # [m*n, dim]
            )
        )
    
    def get_weights(self):
        return self.map_wgt

    @tf.function
    def train(self, nbr_epochs, epsilon_i, epsilon_f, eta_i, eta_f, x_train, x_label, index_label, x_test, index_test):
        with tf.device('/device:gpu:0'):
            for epoch in tf.range(nbr_epochs):
                tf.print("---------- Epoch", epoch + 1, "----------")

                # update the learning rate epsilon
                epsilon_t =  tf.multiply(
                                    epsilon_i,
                                    tf.pow(
                                        tf.math.divide(
                                            epsilon_f, 
                                            epsilon_i
                                        ),
                                        tf.cast(
                                            tf.math.divide(
                                                epoch,
                                                nbr_epochs - 1
                                            ), 
                                            dtype=tf.float32
                                        )
                                    )
                                )
                
                # update the gaussian neighborhood witdh eta
                eta_t =  tf.multiply(
                                  eta_i, 
                                  tf.pow(
                                      tf.math.divide(
                                          eta_f, 
                                          eta_i
                                      ),
                                      tf.cast(
                                          tf.math.divide(
                                              epoch,
                                              nbr_epochs - 1
                                          ), 
                                          dtype=tf.float32
                                      )
                                  )
                              )
                
                # shuffle the training dataset
                tf.random.shuffle(x_train)

                # bmu computing and network update for each sample
                for x_trn in x_train:
                    sample = tf.cast(x_trn, dtype=tf.float32)
                    self.compute_winner(sample)
                    self.update_network(epsilon_t, eta_t)