__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from collections import defaultdict

import seaborn as sbn
sbn.set_context('paper')
sbn.set(font_scale=2)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Plot transform values as function of temperature')
    parser.add_argument('depth', type=str, help='Preprocessed depth')   
    parser.add_argument('--axis', type=int, help='Axis position', default=1000)        
    parser.add_argument('--corrected-depth', type=str, help='Corrected depth')
    args = parser.parse_args() 

    data = np.load(args.depth)
    temps = data['temps']
    depths_ir = data['depth_ir'][()]
    depths_rgb = data['depth_rgb'][()]
    h,w = depths_ir[(args.axis, temps[0])].shape

    wnd = 20
    crops = {
        'Top-Left Crop': [np.s_[0:wnd], np.s_[0:wnd]],
        'Top-Right Crop' : [np.s_[0:wnd], np.s_[w-wnd:-1]],
        'Bottom-Left Crop' : [np.s_[h-wnd:-1], np.s_[0:wnd]],
        'Bottom-Right Crop' : [np.s_[h-wnd:-1], np.s_[w-wnd:-1]],
        'Center Crop' : [np.s_[h//2-wnd//2:h//2+wnd//2], np.s_[w//2-wnd//2:w//2+wnd//2]] # center crop
    }

    depths_corrected = None
    if args.corrected_depth:
        data = np.load(args.corrected_depth)
        depths_corrected = data['depth_corrected'][()]
    

    d_rgb = defaultdict(list)
    d_ir = defaultdict(list)
    d_corr = defaultdict(list)
    for t in temps:
        for k, c in crops.items():
          ir = np.mean(depths_ir[(args.axis, t)][c[0], c[1]])
          rgb = np.mean(depths_rgb[(args.axis, t)][c[0], c[1]])  
          
          d_ir[k].append(ir)
          d_rgb[k].append(rgb)

          if depths_corrected:        
            c = np.mean(depths_corrected[(args.axis, t)][c[0], c[1]])
            d_corr[k].append(c)
    
    display_crops = ['Bottom-Right Crop']
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')

    for cr, c in zip(display_crops, colors):
        plt.plot(temps, d_rgb[cr], c=c, label='Depth from RGB')
    for cr, c in zip(display_crops, colors):
        plt.plot(temps, d_ir[cr], c=c, linestyle='--', label='Depth from IR')

    plt.legend(loc='best')    
    plt.tight_layout()
    plt.savefig('depth_vs_temp_p{}.png'.format(args.axis), dpi=300,  bbox_inches='tight')
    plt.savefig('depth_vs_temp_p{}.pdf'.format(args.axis), dpi=300, bbox_inches='tight', transparent=True)
    ylim = plt.gca().get_ylim()
    plt.show()    

    if depths_corrected:
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Depth (m)')

        for cr, c in zip(display_crops, colors):
            plt.plot(temps, np.repeat(d_rgb[cr][0], temps.shape[0]), c=c, label='Depth from RGB')
        #for c in display_crops:
        #    plt.plot(temps, d_ir[c], linestyle='--', label='Depth from IR / {}'.format(c))
        for cr, c in zip(display_crops, colors):
            plt.plot(temps, d_corr[cr], linestyle='-.', c=c, label='Corrected Depth')
            
        plt.ylim(ylim)
        plt.legend(loc='bottom left')
        plt.tight_layout()        
        plt.savefig('depth_vs_temp_p{}_corrected.pdf'.format(args.axis), dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig('depth_vs_temp_p{}_corrected.png'.format(args.axis), dpi=300, bbox_inches='tight')                
        plt.show()
    
