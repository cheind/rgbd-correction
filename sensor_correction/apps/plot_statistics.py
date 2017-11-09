__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import glob
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib

from sensor_correction.utils import mask_outliers
from sensor_correction.utils import sensor_unproject

import seaborn as sbn
sbn.set_context('paper')
sbn.set(font_scale=2)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Plot error statistics')
    parser.add_argument('depth', type=str, help='Preprocessed depth')       
    parser.add_argument('corrected', type=str, help='Corrected depth')
    parser.add_argument('intrinsics', type=str, help='Camera intrinsics')
    parser.add_argument('--no-show', action='store_true', help='Do not display results, just save image')
    parser.add_argument('--temps', nargs='*', type=int)
    parser.add_argument('--poses', nargs='*', type=int)
    args = parser.parse_args() 

    # Load depth data
    data = np.load(args.depth)    
    temps = data['temps']
    poses = data['poses']    
    if args.temps:
        temps = np.array(args.temps)
    if args.poses:
        poses = np.array(args.poses)

    all_depths_ir = data['depth_ir'][()]
    all_depths_rgb = data['depth_rgb'][()]

    data = np.load(args.corrected)
    all_corrected = data['depth_corrected'][()]
    all_deltae = data['depth_deltae'][()]

    # Load intrinsics
    K = np.loadtxt(args.intrinsics).reshape(3,3)
    Kinv = np.linalg.inv(K)

    h, w = all_depths_ir[(poses[0], temps[0])].shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x, y)
    xy = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))

    se_xyz_before = []
    se_xyz_after = []
    se_depth_before = []
    se_depth_after = []

    for p in poses:
        depth_target = all_depths_rgb[(p, temps[0])]            

        for t in temps:
            print('Processing pos {}, temperature {}'.format(p, t))            
            
            depth_ir = all_depths_ir[(p, t)] # Actual
            depth_c = all_corrected[(p, t)] # Corrected

            xyz_t = sensor_unproject(xy, depth_target.ravel(), Kinv)
            xyz_a = sensor_unproject(xy, depth_ir.ravel(), Kinv)
            xyz_c = sensor_unproject(xy, depth_c.ravel(), Kinv)
            
            # remove extreme outliers
            outliers = mask_outliers(np.abs(depth_ir - depth_target)).ravel()

            before_xyz = np.square((xyz_t - xyz_a)[~outliers])
            after_xyz = np.square((xyz_t - xyz_c)[~outliers])

            se_xyz_before.append(before_xyz)
            se_xyz_after.append(after_xyz)

            rmse_xyz_before = np.sqrt(np.mean(before_xyz, axis=0))
            rmse_xyz_after = np.sqrt(np.mean(after_xyz, axis=0))
            print('  RMSE before (x,y,z) {}'.format(rmse_xyz_before))
            print('  RMSE after (x,y,z)  {}'.format(rmse_xyz_after))

print('Overall')
rmse_xyz_before = np.sqrt(np.mean(np.concatenate(se_xyz_before), axis=0))
rmse_xyz_after = np.sqrt(np.mean(np.concatenate(se_xyz_after), axis=0))
print('  RMSE before (x,y,z) {}'.format(rmse_xyz_before))
print('  RMSE after (x,y,z)  {}'.format(rmse_xyz_after))
