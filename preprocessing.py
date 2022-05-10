import numpy as np
from dexp.datasets import ZDataset
import zarr
import glob
import dask.array as da
from readwrite import extract_coords_from_czi, extract_pixel_sizes_from_czi
import numcodecs


def simple_mean_fusion(raw_zarr, path_to_fused_zarr, channel_names):
    # assume structure: root/time/views/sheets/channels. Might change later.
    fused = zarr.open(path_to_fused_zarr, mode='w-')
    for t, _iter_t in raw_zarr.groups():
        assert 'Time' in t
        time = fused.create_group(t)
        for v, _iter_v in raw_zarr.get(t).groups():
            assert 'View' in v
            print(_iter_v)
            view = time.create_group(v)
            print(raw_zarr.get(t))
            print(raw_zarr.get(t).get(v))

            sheet_0 = da.from_array(raw_zarr.get(t).get(v).get('Sheet0/multi_channel'))
            sheet_0 = sheet_0.astype(np.uint32)
            sheet_1 = da.from_array(raw_zarr.get(t).get(v).get('Sheet1/multi_channel'))
            sheet_1 = sheet_1.astype(np.uint32)
            mean_sheet = (sheet_0 + sheet_1)*0.5
            mean_sheet = mean_sheet.astype(np.uint16)
            for c in range(mean_sheet.shape[0]):
                arr = view.create(channel_names[c], shape=mean_sheet[c].shape)
                da.to_zarr(mean_sheet[c], arr)

    return fused



def register_by_stage_coords(fused, first_czi, path_to_registered_zarr):
    """one time point for now. same shape per view for now."""
    x, y, z = extract_coords_from_czi(first_czi)
    num_views = len(x)
    first_scan = fused.Time0.View0.get('488nm')
    reg = zarr.open(path_to_registered_zarr, 'w-')
    scan_shape = first_scan.shape
    big_shape = (5*scan_shape[0], num_views*scan_shape[1], 5*scan_shape[2])
    c0 = reg.zeros('Ch0-488nm', shape=big_shape)
    c1 = reg.zeros('Ch1-561nm', shape=big_shape)

    dx, dy, dz = extract_pixel_sizes_from_czi(first_czi)
    # convert positions to microns, then to pixels, and center on first scan
    x = np.int16(1e6*(x - x[0])/dx + scan_shape[2]); y = np.int16(1e6*(y-y[0])/dy); z = np.int16((z-z[0])/dz + scan_shape[0])

    time = fused.Time0
    counter = 0
    for view, _iter in time.groups():
        print(counter)
        this_c0 = fused.Time0.get(view).get('488nm')
        this_c1 = fused.Time0.get(view).get('561nm')

        sz, sy, sx = this_c0.shape
        c0[z[counter]:z[counter]+sz, y[counter]:y[counter]+sy, x[counter]:x[counter]+sx] = this_c0

        sz, sy, sx = this_c1.shape
        c1[z[counter]:z[counter]+sz, y[counter]:y[counter]+sy, x[counter]:x[counter]+sx] = this_c1

        counter += 1
    ########################## get rid of padding ############################
    print('removing padding')
    # z
    proj = da.sum(da.sum(da.asarray(c0), axis=1), axis=1)
    ids_z = da.where(proj > 0)

    # y
    proj = da.sum(da.sum(da.asarray(c0), axis=0), axis=1)
    ids_y = da.where(proj > 0)

    # x
    proj = da.sum(da.sum(da.asarray(c0), axis=0), axis=0)
    ids_x = da.where(proj > 0)

    # overwrite existing arrays
    reg.create_dataset('Ch0-488nm', data=c0[ids_z[0], ids_y[0], ids_x[0]], object_codec=numcodecs.JSON(), overwrite=True)
    reg.create_dataset('Ch1-561nm', data=c1[ids_z[0], ids_y[0], ids_x[0]], object_codec=numcodecs.JSON(), overwrite=True)

    return reg



