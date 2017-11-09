__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np
import tensorflow as tf

class GPRegressorGPU:
    '''Gaussian Process regressor on GPU.

    Takes a pre-fitted Gaussian Process regressor (CPU) and prepares a TensorFlow graph for
    prediction using pre-computed values.
    '''

    def __init__(self, gp_cpu, x):
        '''Initialize regressor.

        Params
        ------
        gp_cpu : sensor_correction.GPRegressor
            Pre-fitted Gaussing Process regressor
        x : Nx4 tensor
            Feature vectors on GPU.
        '''

        # Prepare constants
        signal_var = tf.constant(gp_cpu.signal_std**2, dtype=tf.float32)
        mean_y =  tf.constant(gp_cpu.gpr.y_train_mean, dtype=tf.float32)
        w = tf.constant(np.reciprocal(gp_cpu.length_scale), dtype=tf.float32)
        
        # Pre-scale train and test
        xtrain = tf.constant(gp_cpu.gpr.X_train_, dtype=tf.float32) * w
        xtest = x * w
        alpha = tf.expand_dims(tf.constant(gp_cpu.gpr.alpha_, dtype=tf.float32), -1)

        # Compute pairwise squared distance
        a = tf.matmul(
            tf.expand_dims(tf.reduce_sum(tf.square(xtrain), 1), 1),
            tf.ones(shape=(1, xtest.shape[0]))
        )
        
        b = tf.transpose(tf.matmul(
            tf.reshape(tf.reduce_sum(tf.square(xtest), 1), shape=[-1, 1]),
            tf.ones(shape=(xtrain.shape[0], 1)),
            transpose_b=True
        ))

        d = tf.add(a, b) - 2 * tf.matmul(xtrain, xtest, transpose_b=True)

        # Eval kernel
        k = signal_var * tf.exp(-0.5 * d)

        # Predict
        self.predict = tf.matmul(tf.transpose(k), alpha) + mean_y

