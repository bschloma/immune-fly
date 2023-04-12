import numpy as np
import napari
from PIL import Image
from glob import glob
from pathlib import Path


experiment_path = ''
larvae_dirs = glob(experiment_path + '/larvae*')
for i, larvae_dir in enumerate(larvae_dirs):
    green_mip = np.array(Image.open(larvae_dir + '/mips/mip_green_0.tif'))
    red_mip = np.array(Image.open(larvae_dir + '/mips/mip_red_0.tif'))
    if i == 0:
        green_mips = np.zeros((len(larvae_dirs),) + green_mip.shape)
        green_mips[i] = green_mip
        red_mips = np.zeros((len(larvae_dirs),) + red_mip.shape)
        red_mips[i] = red_mip