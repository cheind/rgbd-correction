__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import glob
import os
import cv2
import numpy as np
import re

def model_points(pattern):
    corners = np.zeros((pattern[0]*pattern[1], 3), dtype=np.float32)
    for i in range(pattern[1]):
        for j in range(pattern[0]):
            corners[i * pattern[0] + j] = [j * pattern[2], i * pattern[2], 0.0]
    return corners

def camera_rays(K, w, h):
    Kinv = np.linalg.inv(K)

    rays = np.zeros((w*h, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            rays[y * w + x] = Kinv.dot([x, y, 1.])
    
    return rays

if __name__ == '__main__':

    def size(s):
        return tuple(map(int, s.split('x')))

    import argparse
    parser = argparse.ArgumentParser(description='Dense depth from pattern under planar assumption')
    parser.add_argument('intrinsics', type=str, help='File containing color camera matrix. Single row 3x3 matrix stored in row-major layout.')
    parser.add_argument('distortions', type=str, help='File containing distortion parameters.Single row 1x4 matrix stored in row-major layout.')
    parser.add_argument('indir', type=str, help='Source directory containing images')
    parser.add_argument('-outdir', type=str, nargs='?', help='Target directory. If not specified indir is used.')  
    parser.add_argument('-pattern', type=size, metavar='WIDTHxHEIGHTxSIZE', help='Pattern size.', default=(10,7,34))
    args = parser.parse_args() 

    if args.outdir is None:
        args.outdir = args.indir

    os.makedirs(args.outdir, exist_ok=True)

    K = np.loadtxt(args.intrinsics).reshape(3,3)
    D = np.loadtxt(args.distortions)

    images = sorted(glob.glob(os.path.join(args.indir, '*_color.png')))
    modelpoints = model_points(args.pattern)
    rays = None

    axis = np.float32([[0,0,0],[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)
    axis *= args.pattern[2] * 3

    regex = re.compile(r'(?P<id>\d+)_t(?P<temp>\d+)_p(?P<axis>\d+)_(?P<type>depth|color).png$', flags=re.I)

    for fname in images:
        print('Processing {}'.format(fname))
        r = regex.search(fname)
        if r is None:
            print('Warning file format does not match for \'{}\'.'.format(fname))
            continue

        img = cv2.imread(fname)
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,w = imggray.shape[0:2]

        (found, corners) = cv2.findChessboardCorners(imggray, args.pattern[0:2])
        if found:

            # Estimate position of chessboard w.r.t camera

            cv2.cornerSubPix(imggray, corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.01))        
            (rv, rvec, tvec) = cv2.solvePnP(modelpoints, corners, K, D,  None, None, False, cv2.SOLVEPNP_ITERATIVE)           

            (R,jacobian) = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec.reshape(3)

            # Generate virtual depth map
            pn = T[0:3, 2] # Plane normal
            p0 = T[0:3, 3] # Plane origin
            d = p0.dot(pn)

            if rays is None:
                rays = camera_rays(K, w, h)
            
            t = d / (rays.dot(pn.reshape(3,1)))
            isects = np.multiply(rays, t)

            depths = isects[:,2].reshape(h, w, 1)

            fnamenew = '{:06d}_t{:02d}_p{:04d}_sdepth'.format(int(r['id']), int(r['temp']), int(r['axis']))

            cv2.imwrite(os.path.join(args.outdir, fnamenew + '.png'), depths.astype(np.ushort))  # ushort
            cv2.imwrite(os.path.join(args.outdir, fnamenew + '.exr'), depths.astype(np.float32)) # floats
            np.savetxt(os.path.join(args.outdir, fnamenew + '.txt'), T.reshape(1,-1))

            pts, _ = cv2.projectPoints(axis, rvec, tvec, K, D)
            orig = tuple(pts[0].ravel())
            cv2.line(img, orig, tuple(pts[1].ravel()), (0,0,255), 2)
            cv2.line(img, orig, tuple(pts[2].ravel()), (0,255,0), 2)
            cv2.line(img, orig, tuple(pts[3].ravel()), (255,0,0), 2)

        else:
            print('No pattern found in {}'.format(os.path.basename(fname)))

        
        cv2.drawChessboardCorners(img, args.pattern[0:2], corners, found)
        cv2.imshow('x', img)
        if cv2.waitKey(50) == ord('x'):
            break







