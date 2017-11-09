
__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import glob
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sensor_correction.utils import mask_outliers

import seaborn as sbn
sbn.set_context('paper')
sbn.set(font_scale=3)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Gaussian Process')
    parser.add_argument('depth', type=str, help='Preprocessed depth')       
    parser.add_argument('corrected', type=str, help='Corrected depth')
    parser.add_argument('--no-show', action='store_true', help='Do not display results, just save image')
    parser.add_argument('--temps', nargs='*', type=int)
    parser.add_argument('--poses', nargs='*', type=int)
    args = parser.parse_args() 

    #matplotlib.rcParams.update({'font.size': 20})

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


    for p in poses:
        depth_target = all_depths_rgb[(p, temps[0])]            

        for t in temps:
            print('Processing pos {}, temperature {}'.format(p, t))            
            
            depth_ir = all_depths_ir[(p, t)] # Actual
            depth_c = all_corrected[(p, t)] # Corrected
            depth_delta = all_deltae[(p, t)] # Corrected
            
            errbefore = np.abs(depth_ir - depth_target)
            errafter = np.abs(depth_c - depth_target)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8), dpi=60)
            #fig.suptitle('Gaussian Process Regression', fontsize=24, fontweight='bold')
            
            ax1.set_title('RGB/IR Depth Error Before')
            ax1.axis('off')
            img1 = ax1.imshow(errbefore, cmap='cubehelix', interpolation='nearest', aspect='equal', vmin=0.0, vmax=0.02)                
            d1 = make_axes_locatable(ax1)            
            cax1 = d1.append_axes("right", size="7%", pad=0.05)
            
            ax2.set_title('Depth Correction by GP')
            ax2.axis('off')
            img2 = ax2.imshow(depth_delta, cmap='cubehelix', interpolation='nearest', aspect='equal', vmin=0.0, vmax=0.02)                
            d2 = make_axes_locatable(ax2)            
            cax2 = d2.append_axes("right", size="7%", pad=0.05)           

            ax3.set_title('RGB/IR Depth Error After')
            ax3.axis('off')
            img3 = ax3.imshow(errafter, cmap='cubehelix', aspect='equal', interpolation='nearest', vmin=0.0, vmax=0.02)                
            d3 = make_axes_locatable(ax3)            
            cax3 = d3.append_axes("right", size="7%", pad=0.05)

            cbar1 = fig.colorbar(img1, cax = cax1)
            fig.delaxes(fig.axes[3])

            cbar2 = fig.colorbar(img2, cax = cax2)
            fig.delaxes(fig.axes[3])

            cbar3 = fig.colorbar(img3, cax = cax3)
            cbar3.set_label('Error/Correction (m)', labelpad=3.)

            fig.tight_layout(pad=0.05)
            fig.subplots_adjust(wspace=0.0)
            plt.savefig('correction_t{}_p{:04d}.png'.format(t, p), bbox_inches='tight')
            plt.savefig('correction_t{}_p{:04d}.pdf'.format(t, p), bbox_inches='tight', transparent=True, dpi=300)
            if not args.no_show:
                plt.show()

