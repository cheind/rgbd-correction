__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

from sensor_correction.gp_cpu import GPRegressor

import seaborn as sbn
sbn.set_context('paper')
sbn.set(font_scale=2)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Plot transform values as function of temperature')    
    parser.add_argument('--depth', type=str, help='Preprocessed depth. If not specfied, fixed values will be used')   
    parser.add_argument('--axis', type=int, help='Axis position. Only valid when --depth is specified', default=1200)        
    parser.add_argument('--length-scale', type=float, help='Length scale')
    parser.add_argument('--signal-std', type=float, help='Signal sigma')
    parser.add_argument('--noise-std', type=float, help='Noise sigma')
    
    args = parser.parse_args() 

    if args.depth:
        print('Loading data')
        data = np.load(args.depth)
        temps = data['temps']
        depths_ir = data['depth_ir'][()]
        depths_rgb = data['depth_rgb'][()]
        h,w = depths_ir[(args.axis, temps[0])].shape

        d_rgb = []
        d_ir = []
        for t in temps:
            ir = np.median(depths_ir[(args.axis, t)][h//2-5:h//2+5, w//2-5:w//2+5])
            rgb = np.median(depths_rgb[(args.axis, t)][h//2-5:h//2+5, w//2-5:w//2+5])          
            d_ir.append(ir)
            d_rgb.append(rgb)

        deltad = d_rgb[0] - np.array(d_ir)
        deltad *= 1000
        
        X = temps[::1]
        Y = deltad[::1]
        print(Y.__repr__())
        print(X.__repr__())    
    else:
        print('Using demo data')
        X = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], dtype=np.float32)
        Y = np.array([ 11.20424271,  10.94418716,  11.39420319,  11.40421677,
        11.38424873,  11.74420166,  11.70426559,  12.38423538,
        12.38423538,  12.38423538,  11.40421677,  11.38424873,
        11.52420044,  12.13425446,  12.38423538,  12.38423538,
        12.38423538,  12.38423538,  12.62420464,  12.73423386,
        12.87418556,  13.38422298,  13.38422298,  13.38422298,
        13.38422298,  14.26422596], dtype=np.float32)

    r = GPRegressor()
    opt_params = []
    if args.length_scale is None:
        opt_params.append('length_scale')
        args.length_scale = 5.
    if args.noise_std is None:
        opt_params.append('noise_std')
        args.noise_std = 1e-5
    if args.signal_std is None:
        opt_params.append('signal_std')
        args.signal_std = 0.2

    print('Fitting data. Optimizing for params: {}'.format(opt_params))

    r.fit(X.reshape(-1, 1), Y, args.length_scale, args.signal_std, args.noise_std, normalize=True, optimize=opt_params, repeat=10)
    print('Length scale {}'.format(r.length_scale))
    print('Signal std {}'.format(r.signal_std))
    print('Noise std {}'.format(r.noise_std))

    xtest = np.linspace(X.min(), X.max(), 100)
    ytest, sigma = r.predict(xtest.reshape(-1, 1), return_std=True)

    
    
    plt.xlabel('Temperature (C°)')
    plt.ylabel('RGB/IR depth offset (mm)')
    plt.scatter(X, Y, label='Samples')
    plt.plot(xtest, ytest, label='Prediction')
    plt.fill(
        np.concatenate([xtest, xtest[::-1]]),
        np.concatenate([ytest - 1.9600 * sigma, (ytest + 1.9600 * sigma)[::-1]]),
        alpha=.2, fc='b', ec='b', label='95% confidence interval')
    
    plt.legend(loc='upper left')    
    plt.savefig('gp_ls{:.3f}_ss{:.3f}_ns{:.3f}.png'.format(r.length_scale, r.signal_std, r.noise_std), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()    

    """
    plt.plot(temps, np.repeat(d_rgb[0], 26))
    plt.plot(temps, d_ir)
    plt.plot(temps, d_ir + y/1000.)
    plt.show()
    """


    