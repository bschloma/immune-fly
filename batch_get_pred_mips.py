from readwrite import get_mips
import zarr
import dask.array as da
from glob import glob
from pathlib import Path
from dask.diagnostics import ProgressBar
from PIL import Image


experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_24-dpt-gfp_r4-gal4_uas-mcd8-mcherry_mock_inj_early_mid_24hrs',
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_05_24-dpt-gfp_r4-gal4_uas-mcd8-mcherry_no_inj_early_mid_24hrs']
zarr_paths = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larvae*')
    for larvae_dir in larvae_dirs:
        zarr_paths.append(Path(larvae_dir) / 'prediction.zarr')     # assume ZYX regular zarr structure

for path in zarr_paths:
    if not path.is_dir():
        continue
    try:
        # save mips to tiff using PIL
        mip_dir = Path(path).parent.__str__()

        data = da.from_zarr(path)

        with ProgressBar():
            pred_mip = da.max(data, axis=0).compute()

        Image.fromarray(pred_mip).save(mip_dir + '/prediction_mip' + '.tif')

    except Exception as e:
        print(f'Exception: {e}, skipping!')
        continue

