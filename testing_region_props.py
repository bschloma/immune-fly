import zarr
import napari
from cucim.skimage.measure import regionprops_table
import cupy as cp
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd


path_to_labels_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_08_24_dpt-gfp_silverman_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom/larvae_1/ds_h5/scan_0/labels_pred.zarr'
path_to_culled_labels_zarr = r'/media/brandon/Data1/Brandon/fly_immune/Serenity/2022_08_24_dpt-gfp_silverman_r4-gal4_uas-mcd8-mcherry_ecoli-hs-dtom/larvae_1/ds_h5/scan_0/labels_pred.culled.zarr'

labels = zarr.open(path_to_labels_zarr, 'r')
culled_labels = zarr.open(path_to_culled_labels_zarr, 'a')

min_area = 4000
max_area = 18000
min_eccentricity = 0.0
max_eccentricity = 0.9

# make a dataframe with object properties
props = pd.DataFrame(columns=('label', 'z', 'area', 'eccentricity'))
if len(labels.shape) > 2:
    for z in range(labels.shape[0]):
        this_slice_cu = cp.asarray(labels[z])
        these_props = regionprops_table(this_slice_cu, properties=('label', 'area', 'eccentricity'))
        this_df = pd.DataFrame(np.array([cp.asnumpy(these_props.get('label')).T, cp.asnumpy(these_props.get('area')).T, cp.asnumpy(these_props.get('eccentricity')).T]).T, columns=('label',  'area', 'eccentricity'))
        this_df['z'] = z * np.ones((len(this_df), 1))
        props = pd.concat([props, this_df], ignore_index=True)
elif len(labels.shape) == 2:
    z = 0
    this_slice_cu = cp.asarray(labels)
    these_props = regionprops_table(this_slice_cu, properties=('label', 'area', 'eccentricity'))
    this_df = pd.DataFrame(np.array([cp.asnumpy(these_props.get('label')).T, cp.asnumpy(these_props.get('area')).T,
                                     cp.asnumpy(these_props.get('eccentricity')).T]).T,
                           columns=('label', 'area', 'eccentricity'))
    this_df['z'] = z * np.ones((len(this_df), 1))
    props = pd.concat([props, this_df], ignore_index=True)

bad_df = props[((props.area < min_area) + (props.area > max_area) + (props.eccentricity < min_eccentricity) + (props.eccentricity > max_eccentricity)) > 0]
bad_ids = np.array(bad_df.label, dtype=np.uint32)
z_ids = np.array(bad_df.z, dtype=np.uint32)

for i in range(len(bad_ids)):
    print(str(i) + ' of ' + str(len(bad_ids)))
    z = z_ids[i]
    if len(labels.shape) > 2:
        this_slice = np.array(culled_labels[z])
        this_mask = this_slice != bad_ids[i]
        culled_labels[z] = this_mask * this_slice
    else:
        this_slice = np.array(culled_labels)
        this_mask = this_slice != bad_ids[i]
        culled_labels = this_mask * this_slice

viewer = napari.view_labels(culled_labels)


