__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

def crop(img, border):
    return img[border[1]:-border[1], border[0]:-border[0]]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Create mean depth images for RGB/IR by averaging over redundant captures.')
    parser.add_argument('index', type=str, help='Index CSV')   
    parser.add_argument('--crop', type=int, help='Crop images by this size', default=100)
    parser.add_argument('--unitscale', type=float, help='Scale depth by this value', default=0.001)
    parser.add_argument('--output', type=str, help='Result file', default='input_depths.npz')   
    args = parser.parse_args() 

    df = pd.DataFrame.from_csv(args.index, sep=' ')
    
    depth_ir = {}
    depth_rgb = {}
    temps = []
    poses = []

    groups = df.groupby(df.Temp)
    first = True
    for t, tgroup in groups:
        temps.append(t)
        print('Processing temperature {}'.format(t))        
        for p, pgroup in tgroup.groupby(tgroup.Axis):
            if first:
                poses.append(p)
            print('  Processing position {}'.format(p))
            # Read IR Depth
            d = []
            for name in pgroup[pgroup.Type == 'depth.png']['Name']:
                fname = os.path.join(os.path.dirname(args.index), name)
                dm = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                dm[d==0] = np.nan
                d.append(dm)
            d = np.stack(d, axis=0)
            d = np.mean(d, axis=0)
            depth_ir[(p, t)] = d * args.unitscale

            d = []
            for name in pgroup[pgroup.Type == 'sdepth.exr']['Name']:
                fname = os.path.join(os.path.dirname(args.index), name)
                dm = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                dm[d==0] = np.nan
                d.append(dm)
            d = np.stack(d, axis=0)
            d = np.mean(d, axis=0)
            depth_rgb[(p, t)] = d * args.unitscale
        first = False

    depth_ir = {k: crop(img, (args.crop, args.crop)) for k, img in depth_ir.items()}
    depth_rgb = {k: crop(img, (args.crop, args.crop)) for k, img in depth_rgb.items()}

    np.savez(args.output, depth_ir=depth_ir, depth_rgb=depth_rgb, temps=temps, poses=poses)



    