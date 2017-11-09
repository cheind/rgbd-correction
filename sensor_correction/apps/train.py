__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from sensor_correction.utils import sensor_unproject
from sensor_correction.gp_cpu import GPRegressor

def select_data(temps, poses, all_depths_ir, all_depths_rgb, Kinv, xy, target='rgb'):
    
    sel_xyzt = []
    sel_deltas = []

    for p in poses:
        if target == 'rgb':
            depth_target = all_depths_rgb[(p, temps[0])]            
        elif target == 'ir':
            depth_target = all_depths_ir[(p, temps[0])]

        d_target = depth_target[xy[:,1], xy[:,0]]

        for t in temps:
            depth_ir = all_depths_ir[(p, t)] # Actual
            d_ir = depth_ir[xy[:,1], xy[:,0]]
            
            xyz = sensor_unproject(xy, d_ir, Kinv)          
            
            xyzt = np.empty((xyz.shape[0], 4), dtype=np.float32)
            xyzt[:, :3] = xyz
            xyzt[:, 3] = t

            delta = d_target - d_ir
            mask = d_ir > 0.

            """
            plt.imshow(depth_rgb - depth_ir)
            plt.plot(xy[:,0][mask], xy[:,1][mask], 'k+')
            plt.colorbar()
            plt.show()
            """
            
            sel_xyzt.append(xyzt[mask])
            sel_deltas.append(delta[mask])

    sel_xyzt = np.concatenate(sel_xyzt)
    sel_deltas = np.concatenate(sel_deltas)

    return sel_xyzt, sel_deltas

if __name__ == '__main__':

    np.random.seed(1)

    import argparse
    parser = argparse.ArgumentParser(description='Train Gaussian Process for depth correction.')
    parser.add_argument('depth', type=str, help='Preprocessed depth data')
    parser.add_argument('intrinsics', type=str, help='Camera intrinsics')
    parser.add_argument('--output', type=str, help='Result regressor filename', default='gpr.pkl')
    parser.add_argument('--target', type=str, help='Target depth to train for, RGB or IR.', default='rgb')
    args = parser.parse_args() 

    # Load depth data
    data = np.load(args.depth)    
    temps = data['temps']
    poses = data['poses']
    all_depths_ir = data['depth_ir'][()]
    all_depths_rgb = data['depth_rgb'][()]

    h, w = all_depths_ir[(poses[0], temps[0])].shape

    # Load intrinsics
    K = np.loadtxt(args.intrinsics).reshape(3,3)
    Kinv = np.linalg.inv(K)

    # Create train and test data
    x = np.linspace(0, w-1, 8, dtype=np.int32)
    y = np.linspace(0, h-1, 8, dtype=np.int32)
    xx, yy = np.meshgrid(x, y)    
    xy_train = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))    
    train_xyzt, train_deltae = select_data(
        temps[::2], 
        poses, 
        all_depths_ir, 
        all_depths_rgb, 
        Kinv, 
        xy_train,
        target=args.target.lower())

    xy_test = np.random.uniform(0, [w-1,h-1], size=(10,2)).astype(np.int32)    
    test_xyzt, test_deltae = select_data(
        temps[::2], 
        poses[::2],
        all_depths_ir, 
        all_depths_rgb, 
        Kinv, 
        xy_test,
        target=args.target.lower())

    r = GPRegressor()
    r.fit(train_xyzt, train_deltae, length_scale=[0.5, 0.5, 0.5, 10], signal_std=1., noise_std=0.002, optimize=True, normalize=True, repeat=2)
    ypred = r.predict(test_xyzt)

    d = ypred - test_deltae
    rmse = np.sqrt(np.mean(np.square(d)))

    print('RMSE {:e}'.format(rmse))
    print('Optimized length scale {}'.format(r.length_scale))
    print('Optimized signal std {}'.format(r.signal_std))
    print('Optimized noise std {}'.format(r.noise_std))

    r.save(args.output)