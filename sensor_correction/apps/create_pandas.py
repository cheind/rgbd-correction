__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import pandas as pd
import numpy as np
import glob
import re
import os

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Convert raw image files to pandas csv with unique file naming.')
    parser.add_argument('indir', type=str, help='Directory containing caputured files')   
    args = parser.parse_args() 

    
    regex = re.compile(r'(?P<id>\d+)_t(?P<temp>\d+)_p(?P<axis>\d+)_(?P<type>.*)', flags=re.I)

    items = []
    for fname in glob.glob(os.path.join(args.indir, '*')):
        print(fname)
        r = regex.search(fname)
        if r is not None:
            items.append((
                int(r['id']),
                int(r['temp']),
                int(r['axis']),
                r['type'].lower(),
                os.path.basename(fname)
            ))
        else:
            print('File \'{}\' does not match pattern.'.format(fname))
    
    df = pd.DataFrame.from_records(items, columns=['Id', 'Temp', 'Axis', 'Type', 'Name'])
    df.to_csv('index.csv', header=True, sep=' ', index=False)

        

