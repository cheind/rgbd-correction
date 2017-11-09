__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

'''
Converts raw caputered 4D data (temperature, axis-position, color, depth) from @gebenh capture tool to
a list of unique filenames matching the following pattern

    <id>_t<temperature>_p<axisposition>_<color|depth>.png
'''

import glob
import os
import re
import numpy as np
from shutil import copyfile

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Convert raw image files to pandas csv with unique file naming.')
    parser.add_argument('indir', type=str, help='Directory containing caputured files')    
    parser.add_argument('outdir', type=str, help='Result directory. Will be created if not found.')
    args = parser.parse_args() 

    regex = re.compile(r'experiment_(?P<temp>\d+)[/\\](?P<id>\d+)_(?P<type>depth|color).png$', flags=re.I)

    os.makedirs(args.outdir)

    axis = None
    for idx, fname in enumerate(sorted(glob.iglob(os.path.join(args.indir, '**', '*.png'), recursive=True))):
        if axis is None:
            # Assume same axis movement for all temperature directories.
            axis = np.loadtxt(os.path.join(os.path.dirname(fname), 'axis.txt'), dtype=int, skiprows=1)[:, 1] 

        r = regex.search(fname)
        if r is None:
            print('Warning file \'{}\' does not match format'.format(fname))
        else:
            newfname = '{:06d}_t{:02d}_p{:04d}_{}.png'.format(idx//2, int(r['temp']), axis[int(r['id'])], r['type'])
            copyfile(fname, os.path.join(args.outdir, newfname))