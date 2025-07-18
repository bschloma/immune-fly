from readwrite import create_pyramid_from_zarr
from pathlib import Path
from glob import glob
from tqdm import tqdm


experiment_paths = [
    r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2023_06_07-dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_early-mid_24hrs_high_dose',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_07_dpt-gfp_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_earlyL3_24hrs',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_10_dpt-gfp_4r-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_24hrs',
    r'/media/brandon/Data1/Brandon/fly_immune/Lightsheet_Z1/2023_02_16_dpt-gfp_4r-gal4_uas-mcd8-mcherry_ecoli-hs-dtom_24hrs_mid']

zarr_paths = []
for path in experiment_paths:
    larvae_dirs = glob(path + '/larvae*')
    for larvae_dir in larvae_dirs:
        zarr_paths.append(Path(larvae_dir) / 'tmp.zarr/tmp_big_stack')

for path in tqdm(zarr_paths):
    if not path.is_dir():
        continue
    try:
        create_pyramid_from_zarr(path_to_plain_zarr=path, path_to_ome_zarr=path.parent.parent / 'im.ome.zarr')

    except Exception as e:
        print(f'Exception: {e}, skipping!')
        continue

