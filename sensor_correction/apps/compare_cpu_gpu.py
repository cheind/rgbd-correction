__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np
import tensorflow as tf
from  tensorflow.contrib.staging import StagingArea
from tensorflow.python.ops import data_flow_ops
import time
import math

from sensor_correction.gp_cpu import GPRegressor
from sensor_correction.gp_gpu import GPRegressorGPU
from sensor_correction.utils import sensor_unproject, create_batches

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Gaussian Process')
    parser.add_argument('regressor', type=str, help='Trained GP')
    parser.add_argument('depth', type=str, help='Preprocessed depth')       
    parser.add_argument('intrinsics', type=str, help='Camera intrinsics')
    parser.add_argument('--batch-sizes', nargs='*', type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    args = parser.parse_args() 

    data = np.load(args.depth)    
    temps = data['temps']
    poses = data['poses']
    all_depths_ir = data['depth_ir'][()]
    all_depths_rgb = data['depth_rgb'][()]
    
    h, w = all_depths_ir[(poses[0], temps[0])].shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x, y)
    xy = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))

    # Load regressor
    r_cpu = GPRegressor()
    r_cpu.load(args.regressor)


    # Load intrinsics
    K = np.loadtxt(args.intrinsics).reshape(3,3)
    Kinv = np.linalg.inv(K)

    total_time_cpu = 0.
    total_time_gpu = 0.
    total_count = 0
                
    depth_ir = all_depths_ir[(poses[0], temps[0])]

    xyz = sensor_unproject(xy, depth_ir.ravel(), Kinv)
    xyzt = np.column_stack((xyz, np.ones(xyz.shape[0])*temps[0]))

    times_gpu = []
    times_cpu = []

    for bs in args.batch_sizes:
        batches = create_batches(xyzt, bs)        

        # gpu without staging
        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                xyzt_gpu = tf.placeholder(dtype=tf.float32, shape=[bs, 4])
                r_gpu = GPRegressorGPU(r_cpu, xyzt_gpu)

                # warm-up
                for b in batches:
                    deltae_gpu = sess.run(r_gpu.predict, feed_dict={xyzt_gpu : b})   

                start_time = time.time()
                for b in batches:
                    deltae_gpu = sess.run(r_gpu.predict, feed_dict={xyzt_gpu : b})           
                t = time.time() - start_time
                times_gpu.append(t)
                print('GPU {:.3f}sec / Batch size {}'.format(t, bs))
    
        start_time = time.time()
        for b in batches:
            deltae_cpu = r_cpu.predict(b)  
        t = time.time() - start_time    
        print('CPU {:.3f}sec / Batch size {}'.format(t, bs))
        times_cpu.append(t)

    np.savez('times.npz', batches=args.batch_sizes, cpu=times_cpu, gpu=times_gpu)