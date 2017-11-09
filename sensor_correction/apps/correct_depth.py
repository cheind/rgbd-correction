__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import glob
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sensor_correction.gp_cpu import GPRegressor
from sensor_correction.utils import sensor_unproject, create_batches

def crop(img, border):
    return img[border[1]:-border[1], border[0]:-border[0]]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Gaussian Process')
    parser.add_argument('regressor', type=str, help='Trained GP')
    parser.add_argument('depth', type=str, help='Preprocessed depth')       
    parser.add_argument('intrinsics', type=str, help='Camera intrinsics')
    parser.add_argument('--output', type=str, help='Result file', default='corrected_depths.npz')    
    parser.add_argument('--gpu', action='store_true', help='Use GPU') 
    args = parser.parse_args() 

    matplotlib.rcParams.update({'font.size': 20})

    # Load depth data
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
    r = GPRegressor()
    r.load(args.regressor)

    if args.gpu:
        import tensorflow as tf
        from sensor_correction.gp_gpu import GPRegressorGPU
        sess = tf.Session()
        xfeed = tf.placeholder(dtype=tf.float32, shape=[16384 ,4])
        r = GPRegressorGPU(r, xfeed)


    # Load intrinsics
    K = np.loadtxt(args.intrinsics).reshape(3,3)
    Kinv = np.linalg.inv(K)

    all_depths = {}
    all_deltae = {}

    total_time = 0.
    total_count = 0

    for p in poses:
        for t in temps:
            print('Processing pos {}, temperature {}'.format(p, t))            
            
            depth_ir = all_depths_ir[(p, t)] # Actual

            start_time = time.time()
            xyz = sensor_unproject(xy, depth_ir.ravel(), Kinv)
            xyzt = np.column_stack((xyz, np.ones(xyz.shape[0])*t))
            batches = create_batches(xyzt, 16384, pad=True)

            deltae = []
            for b in batches:
                if args.gpu:
                    br = sess.run(r.predict, feed_dict={xfeed : b})
                else:
                    br = r.predict(b)
                deltae.append(br)
            deltae = np.concatenate(deltae)[:xyzt.shape[0]].reshape(depth_ir.shape)
            
            depth_corr = depth_ir + deltae
            
            total_time += (time.time() - start_time)
            total_count += 1

            all_deltae[(p, t)] = deltae
            all_depths[(p, t)] = depth_corr

    print('Processing took {:.3f}sec total, {:.3f}sec on average'.format(total_time, total_time / total_count))

    np.savez(args.output, depth_corrected=all_depths, depth_deltae=all_deltae, temps=temps, poses=poses)
