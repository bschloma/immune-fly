from readwrite import convert_czi_views_to_fuse_reg_ome_zarr
import pandas as pd

path_to_czi_dir = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_03_07-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_modified_earlyL3_timeseries/larvae_2/timeseries_1'
path_to_new_zarr = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_03_07-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_modified_earlyL3_timeseries/larvae_2/blah.zarr'
core_file_name = r'scan'
suffix = r'.czi'
namestr = '*.czi'
chunk_sizes = (1, 1, 100, 1920, 1920)#(1, 64, 256, 256)
num_time_points = 30
num_views = 9
num_sheets = 2
pyramid_scales = 5
reversed_y = False
num_channels = 2

#stage_positions = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_03_07-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_modified_earlyL3_timeseries/larvae_2/stage_positions.pkl')
convert_czi_views_to_fuse_reg_ome_zarr(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr, core_file_name, suffix, chunk_sizes, pyramid_scales, reversed_y, stage_positions=None)



# pasted mip code here temporarily
from readwrite import get_mips
import zarr
import dask.array as da
from glob import glob
from pathlib import Path
from dask.diagnostics import ProgressBar
from PIL import Image

path = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_03_07-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_modified_earlyL3_timeseries/larvae_2/tmp.zarr/tmp_big_stack'
mip_dir = Path(path).parent.parent / "mips"
mip_dir.mkdir(exist_ok=True)
mip_dir = mip_dir.__str__()

data = da.from_zarr(path)

# green
with ProgressBar():
    green_mip = get_mips(data, channel=0)

for i in range(green_mip.shape[0]):
    this_green_mip = green_mip[i]
    Image.fromarray(this_green_mip).save(mip_dir + '/mip_green' + '_' + str(i) + '.tif')

# red
with ProgressBar():
    red_mip = get_mips(data, channel=1)

for i in range(green_mip.shape[0]):
    this_red_mip = red_mip[i]
    Image.fromarray(this_red_mip).save(mip_dir + '/mip_red' + '_' + str(i) + '.tif')