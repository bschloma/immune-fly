from readwrite import get_mips
import zarr
import dask.array as da
from glob import glob
from pathlib import Path
from dask.diagnostics import ProgressBar
from PIL import Image
from zarr.storage import DirectoryStore



experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2025_04_17-PGRP-LC-GFP_pilot',
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2025_06_10_gfp-rel_ecoli-hs-dtom_6hrs']


zarr_paths = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larva*')
    for larvae_dir in larvae_dirs:
        zarr_paths.append(Path(larvae_dir) / 'im.ome.zarr/0')
        #zarr_paths.append(Path(larvae_dir) / 'bacteria.segmentation.ome.zarr/0')

for path in zarr_paths:
    # if not path.is_dir():
    #     continue
    try:
        # save mips to tiff using PIL
        mip_dir = Path(path).parent.parent / "mips_0"
        mip_dir.mkdir(exist_ok=True)
        #mip_dir.mkdir(exist_ok=False)
        mip_dir = mip_dir.__str__()

        data = da.from_zarr(DirectoryStore(path.__str__()))

        # green
        with ProgressBar():
            green_mip = get_mips(data, channel=0)

        for i in range(green_mip.shape[0]):
            this_green_mip = green_mip[i]
            Image.fromarray(this_green_mip).save(mip_dir + '/mip_green' + '_' + str(i) + '.tif')

        # red
        if data.shape[1] == 1:
            continue
        with ProgressBar():
            red_mip = get_mips(data, channel=1)

        for i in range(green_mip.shape[0]):
            this_red_mip = red_mip[i]
            Image.fromarray(this_red_mip).save(mip_dir + '/mip_red' + '_' + str(i) + '.tif')


    except Exception as e:
        print(f'Exception: {e}, skipping!')
        continue

