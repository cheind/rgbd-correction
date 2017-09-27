# Dense RGB-D depth correction in the spatio-thermal domain

This repository contains code and data accompanying our work on spatio-thermal depth correction of RGB-D sensors based on Gaussian Processes in real-time.

![](etc/correction_t17_p1200.png)

## Capture setup

Our capture setup consists of a RGB-D sensor looking towards a known planar object. The sensor is coupled with an electronic linear axis to adjust distance. We captured data at distances [40cm, 90cm, 10cm steps] in the temperate range of [25°C, 35°C, 1°C steps]. At each temperature/distance tuple we grabbed 50 images from both RGB and IR (aligned with RGB) sensors. We then created an artificial depth map for all RGB images utilizing the known calibration target in sight.

## Dataset

We provide two versions of our dataset
- [rgbd-correction-mini.zip](https://s3.eu-central-1.amazonaws.com/rgbd-correction/rgbd-correction-mini.zip) ~80 MBytes | Contains mostly pre-processed mean depth images as described in our paper.
- [rgbd-correction-raw.zip](https://s3.eu-central-1.amazonaws.com/rgbd-correction/rgbd-correction-raw.zip) ~6 GBytes | Contains the entire raw capture data, plus artificial depth maps.

Depending on your needs, you might choose one over the other. As a rule of thumb use `rgbd-correction-mini.zip` if you like to reproduce our results or play around with our code basis. Otherwise go with `rgbd-correction-raw.zip`.

### The `rgbd-correction-mini` version

The `.zip` has the following file structure
```
root
│   ReadMe.md                       Notes on the dataset
│   intrinsics.txt                  Intrinsic camera parameters
│   distortions.txt                 Lens distortion parameters
|   preprocessed_depth.npz          Pre-processed data in numpy format
```

The following snippet shows how to load the extracted data.

```python
import numpy as np
import sys

# Read npz file
data = np.load(sys.argv[1])    

# All temps and positions enumerated
temps = data['temps']
poses = data['poses']    

print('Temperatures {}'.format(temps))
print('Positions {}'.format(poses))

# Access pre-processed mean depth maps
all_depths_ir = data['depth_ir'][()]
all_depths_rgb = data['depth_rgb'][()]

# Index an individual depth map by position-temperature tuple
depth_ir = all_depths_ir[(poses[0], temps[0])]
print(depth_ir.shape)
```

### The `rgbd-correction-raw` version

The `.zip` has the following file structure
```
root
│   ReadMe.md                       Notes on the dataset
│   intrinsics.txt                  Intrinsic camera parameters
│   distortions.txt                 Lens distortion parameters
|   index.csv                       CSV index of all files
|   000000_t10_p0700_color.png      RGB image at t=10°C, pos=700
|   000000_t10_p0700_depth.png      IR depth image in mm
|   000000_t10_p0700_sdepth.exr     Artificial RGB float32 depth map in mm
|   000000_t10_p0700_sdepth.exr     Artificial RGB ushort16 depth map in mm
|   000000_t10_p0700_sdepth.txt     4x4 pose of calibration object w.r.t RGB
|   ...
|   007799_t35_p1200_sdepth.txt
```

The following snippet makes use of index.csv to enumerate the data quickly.

```python
import numpy as np
import pandas as pd
import sys

# Read index.csv from CLI arguments
df = pd.DataFrame.from_csv(sys.argv[1], sep=' ')

# Loop over all IR depth maps grouped by temperature
# This will give 6x50 filenames per group
groups = df[df.Type == 'depth.png'].groupby('Temp')

for t, group in groups:
    print('Processing temperature {}'.format(t))
    for n in group['Name']:
        print('  Found {}'.format(n))
```

Then use your favorite libraries to process the images.

## Code
The code performing our regression will be uploaded in the coming days.

## Acknowledgements
This research is funded by the projects Lern4MRK (Austrian Ministry for Transport, Innovation and Technology), and AssistMe (FFG, 848653), as well as the European Union in cooperation with the State of Upper Austria within the project Investition in Wachstum und Besch\"aftigung (IWB).

Code and dataset created by members of [PROFACTOR Gmbh](http://www.profactor.at).



