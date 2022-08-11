"""functions for reading and writing image data"""

import numpy as np
from aicsimageio import AICSImage, readers
import zarr
import glob
import dask.array as da
from matplotlib.pyplot import imread
from dexp.datasets import ZDataset
#from tqdm import tqdm
from pathlib import Path
import os
import h5py
from itertools import product
from typing import Iterator, Tuple
import cupy as cp
from cucim.skimage.transform import pyramid_reduce
from cucim.skimage.transform.pyramids import _check_factor
import math


def extract_coords(im_bio):
    """extract xyz stage coordinates from a Bioformats Reader object"""
    num_views = len(im_bio.metadata.images)
    x = np.zeros(num_views)
    y = np.zeros(num_views)
    z = np.zeros(num_views)

    for view in range(num_views):
        x[view] = im_bio.metadata.images[view].stage_label.x
        y[view] = im_bio.metadata.images[view].stage_label.y
        z[view] = im_bio.metadata.images[view].stage_label.z

    return x, y, z


def extract_coords_from_czi(first_czi):
    img = AICSImage(first_czi, reader=readers.bioformats_reader.BioformatsReader)
    x, y, z = extract_coords(img)

    return x, y, z


def extract_pixel_sizes_from_czi(first_czi):
    im_bio = AICSImage(first_czi, reader=readers.bioformats_reader.BioformatsReader)
    dx = im_bio.metadata.images[0].pixels.physical_size_x
    dy = im_bio.metadata.images[0].pixels.physical_size_y
    dz = im_bio.metadata.images[0].pixels.physical_size_z

    return dx, dy, dz


def zeiss_filename(core_name, suffix, iteration):
    if iteration == 0:
        out_name = core_name + suffix
    else:
        out_name = core_name + '(' + str(iteration) + ')' + suffix

    return out_name


def convert_czi_views_to_zarr(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr,
                              core_name, suffix=r'.czi', chunk_sizes=(1, 64, 256, 256), channel_names=None):
    """function to read in a folder of czi files then save as zarr with prescribed chunking. Assumes file structure:
    each czi file can contain multiple channels, but different sheets, views, and timepoints are separate files.

    Notes:   """

    # get file names in czi dir
    filenames = glob.glob(path_to_czi_dir + '/' + namestr)
    num_files = len(filenames)

    # if no channel names are provided, get them from AICSImage
    if channel_names is None:
        # get channel names from first file
        first_file = zeiss_filename(core_name, suffix, 0)
        img = AICSImage(path_to_czi_dir + '/' + first_file, reader=readers.bioformats_reader.BioformatsReader)
        channel_names = img.channel_names

    # create file_id_tree used to pick the right image files
    assert num_files == num_time_points * num_views * num_sheets
    id_tree = create_file_id_hierarchy(num_time_points, num_views, num_sheets)

    # create zarr
    root = zarr.open(path_to_new_zarr, mode='w-')
    for t in range(num_time_points):
        time = root.create_group('Time' + str(t))
        for v in range(num_views):
            view = time.create_group('View' + str(v))
            for s in range(num_sheets):
                sheet = view.create_group('Sheet' + str(s))
                file_number = get_file_number_from_tvs(id_tree, t, v, s)
                this_file_name = zeiss_filename(core_name, suffix, file_number)
                img = AICSImage(path_to_czi_dir + '/' + this_file_name)
                this_data = img.get_image_dask_data("CZYX", T=0)
                this_data.rechunk(chunk_sizes)
                arr = sheet.create(name='multi_channel', shape=this_data.shape)
                da.to_zarr(this_data, arr)
                # for c in range(len(channel_names)):
                #     this_data = img.get_image_dask_data("ZYX", C=c, T=0)
                #     this_data.rechunk(chunk_sizes)
                #     arr = sheet.create(name=channel_names[c], shape=this_data.shape)
                #     da.to_zarr(this_data, arr)

    return root


def convert_tiffs_to_zarr(im_dir, path_to_new_zarr, chunk_sizes=(1, 64, 256, 256)):
    """convert folder of tiffs to zarr. just for testing registration for now, not ideal as long term system. assumes a dir structure"""
    # create zarr
    root = zarr.open(path_to_new_zarr, mode='w-')

    # get regions
    region_dirs = glob.glob(im_dir + '/region*')

    # loop
    for r in range(len(region_dirs)):
        print("collecting view " + str(r) + ' of ' + str(len(region_dirs) - 1))

        # create group in zarr
        view = root.create_group("View" + str(r))

        # get sheets (illuminations)
        this_region_dir = region_dirs[r]
        sheet_dirs = glob.glob(this_region_dir + '/sheet*')

        for s in range(len(sheet_dirs)):
            # create group in zarr
            sheet = view.create_group("Sheet" + str(s))

            # get channels
            this_sheet_dir = sheet_dirs[s]
            channel_dirs = glob.glob(this_sheet_dir + '/c*')

            for c in range(len(channel_dirs)):

                # get z slices
                this_channel_dir = channel_dirs[c]
                slices = glob.glob(this_channel_dir + '/*.tif')

                for z in range(len(slices)):
                    # read tiff
                    this_im = imread(slices[z])
                    if z == 0:
                        # create group in zarr
                        # channel = sheet.create("Channel" + str(c), shape=(len(slices), this_im.shape[0], this_im.shape[1]))
                        im_stack = np.zeros((len(slices), this_im.shape[0], this_im.shape[1]))
                    im_stack[z] = this_im

                if len(slices) > 0:
                    im_stack_da = da.array(im_stack)
                    im_stack_da.rechunk(chunk_sizes)
                    channel = sheet.create(name="Channel" + str(c), shape=im_stack_da.shape)
                    da.to_zarr(im_stack_da, channel)

    return root


def create_file_id_hierarchy(num_time_points, num_views, num_sheets):
    id_tree = zarr.group()
    counter = 0
    for t in range(num_time_points):
        time = id_tree.create_group('Time' + str(t))
        for v in range(num_views):
            view = time.create_group('View' + str(v))
            for s in range(num_sheets):
                sheet = view.create_group('Sheet' + str(s))
                sheet.create('file_id', shape=(1,), dtype=np.uint16)
                sheet.file_id[0] = np.array(counter)
                counter += 1

    return id_tree


def get_file_number_from_tvs(id_tree, t, v, s):
    file_number = id_tree.get('Time' + str(t)).get('View' + str(v)).get('Sheet' + str(s)).file_id[0]

    return file_number


def convert_czi_views_to_fuse_reg_ZD(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr,
                                     core_name, suffix=r'.czi', chunk_sizes=(1, 64, 256, 256), channel_names=None,
                                     big_shape=None):
    """function to read in a folder of czi files, compute fusion + registration, then save as ZDataset with prescribed chunking. Assumes file structure:
    each czi file can contain multiple channels, but different sheets, views, and timepoints are separate files.

    Notes:   """

    # get file names in czi dir
    filenames = glob.glob(path_to_czi_dir + '/' + namestr)
    num_files = len(filenames)

    # get info from first file
    first_file = zeiss_filename(core_name, suffix, 0)
    first_img = AICSImage(path_to_czi_dir + '/' + first_file, reader=readers.bioformats_reader.BioformatsReader)

    # will create a big shape for registration first, then crop later
    if big_shape is None:
        big_shape = (3 * first_img.dims.Z, num_views * first_img.dims.Y, 3 * first_img.dims.X)

    # coordinates of each view
    x, y, z = extract_coords_from_czi(path_to_czi_dir + '/' + first_file)

    # pixel sizes
    dx, dy, dz = extract_pixel_sizes_from_czi(path_to_czi_dir + '/' + first_file)

    # convert positions to microns, then to pixels, and center on first scan. assumes first view is smallest y.
    x = np.int16(1e6 * (x - x[0]) / dx + first_img.dims.X)
    y = np.int16(1e6 * (y - y[0]) / dy)
    z = np.int16((z - z[0]) / dz + first_img.dims.Z)

    # check that the provided big shape is compatible with the data. Raise error if there's a problem
    check_big_shape(big_shape, x, y, z, first_img.shape[2:])

    # if no channel names are provided, get them from AICSImage
    if channel_names is None:
        channel_names = first_img.channel_names

    # create file_id_tree used to pick the right image files
    assert num_files == num_time_points * num_views * num_sheets
    id_tree = create_file_id_hierarchy(num_time_points, num_views, num_sheets)

    # create ZDataset
    root = ZDataset(path_to_new_zarr, mode="w-")

    for t in range(num_time_points):
        # create channels using first timepoint info. from this microscope, both channels have same shape
        if t == 0:
            for c in range(len(channel_names)):
                root.add_channel(name=channel_names[c], shape=(num_time_points,) + big_shape, dtype=first_img.dtype,
                                 chunks=chunk_sizes)

        # create temporary zarr on disk for registering big stack that we will overwrite and then delete
        tmp_zarr_path = Path(path_to_new_zarr).parent / 'tmp.zarr'
        tmp_zarr = zarr.open(tmp_zarr_path.__str__(), 'w')
        tmp_big_stack = tmp_zarr.create('tmp_big_stack', shape=(len(channel_names),) + big_shape, dtype=first_img.dtype,
                                        chunks=chunk_sizes)

        # note: apparent issue with tqdm and napari
        #for v in tqdm(range(num_views), 'converting time point ' + str(t) + ' of ' + str(num_time_points - 1)):
        for v in range(num_views):

            # sheet_0
            s = 0
            file_number = get_file_number_from_tvs(id_tree, t, v, s)
            this_file_name = zeiss_filename(core_name, suffix, file_number)
            img = AICSImage(path_to_czi_dir + '/' + this_file_name)
            sheet_0 = img.get_image_dask_data("CZYX", T=0)
            sheet_0.rechunk(chunk_sizes)
            sheet_0 = sheet_0.astype(np.uint32)

            # sheet_1
            s = 1
            file_number = get_file_number_from_tvs(id_tree, t, v, s)
            this_file_name = zeiss_filename(core_name, suffix, file_number)
            img = AICSImage(path_to_czi_dir + '/' + this_file_name)
            sheet_1 = img.get_image_dask_data("CZYX", T=0)
            sheet_1.rechunk(chunk_sizes)
            sheet_1 = sheet_1.astype(np.uint32)

            # fuse
            mean_sheet = (sheet_0 + sheet_1) * 0.5
            mean_sheet = mean_sheet.astype(np.uint16)

            # assemble each view into tmp_big_stack
            for c in range(len(channel_names)):
                sz, sy, sx = mean_sheet[c].shape
                print("writing c " + str(c))
                tmp_big_stack[c, z[v]:z[v] + sz, y[v]:y[v] + sy, x[v]:x[v] + sx] = mean_sheet[c]

        for c in range(len(channel_names)):
            grp = channel_names[c]
            root.write_stack(grp, t, da.from_zarr(tmp_zarr.tmp_big_stack)[c])

    return root


def convert_czi_views_to_fuse_reg_ome_zarr(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr,
                                     core_name, suffix=r'.czi', chunk_sizes=(1, 64, 256, 256), channel_names=None,
                                     big_shape=None, pyramid_scales=5):
    """function to read in a folder of czi files, compute fusion + registration, then save as ome-zarr with prescribed chunking and pyramiding. Assumes file structure:
    each czi file can contain multiple channels, but different sheets, views, and timepoints are separate files. Pyramid computation based on code by Jordao Bragantini.

    Notes:   NEED TO TEST"""

    # get file names in czi dir
    filenames = glob.glob(path_to_czi_dir + '/' + namestr)
    num_files = len(filenames)

    # get info from first file
    first_file = zeiss_filename(core_name, suffix, 0)
    first_img = AICSImage(path_to_czi_dir + '/' + first_file, reader=readers.bioformats_reader.BioformatsReader)

    # will create a big shape for registration first, then crop later
    if big_shape is None:
        big_shape = (3 * first_img.dims.Z, num_views * first_img.dims.Y, 3 * first_img.dims.X)

    # coordinates of each view
    x, y, z = extract_coords_from_czi(path_to_czi_dir + '/' + first_file)

    # pixel sizes
    dx, dy, dz = extract_pixel_sizes_from_czi(path_to_czi_dir + '/' + first_file)

    # convert positions to microns, then to pixels, and center on first scan. assumes first view is smallest y.
    x = np.int16(1e6 * (x - x[0]) / dx + first_img.dims.X)
    y = np.int16(1e6 * (y - y[0]) / dy)
    z = np.int16((z - z[0]) / dz + first_img.dims.Z)

    # check that the provided big shape is compatible with the data. Raise error if there's a problem
    check_big_shape(big_shape, x, y, z, first_img.shape[2:])

    # if no channel names are provided, get them from AICSImage
    if channel_names is None:
        channel_names = first_img.channel_names

    # create file_id_tree used to pick the right image files
    assert num_files == num_time_points * num_views * num_sheets
    id_tree = create_file_id_hierarchy(num_time_points, num_views, num_sheets)

    # create temporary zarr on disk for registering big stack that we will overwrite and then delete
    tmp_zarr_path = Path(path_to_new_zarr).parent / 'tmp.zarr'
    tmp_zarr = zarr.open(tmp_zarr_path.__str__(), 'w-')
    tmp_big_stack = tmp_zarr.create('tmp_big_stack', shape=(num_time_points, len(channel_names),) + big_shape, dtype=first_img.dtype,
                                    chunks=chunk_sizes)

    print("computing fusion and registration")
    for t in range(num_time_points):
        for v in range(num_views):
            # sheet_0
            s = 0
            file_number = get_file_number_from_tvs(id_tree, t, v, s)
            this_file_name = zeiss_filename(core_name, suffix, file_number)
            img = AICSImage(path_to_czi_dir + '/' + this_file_name)
            sheet_0 = img.get_image_dask_data("CZYX", T=0)
            sheet_0.rechunk(chunk_sizes[1:])
            sheet_0 = sheet_0.astype(np.uint32)

            # sheet_1
            s = 1
            file_number = get_file_number_from_tvs(id_tree, t, v, s)
            this_file_name = zeiss_filename(core_name, suffix, file_number)
            img = AICSImage(path_to_czi_dir + '/' + this_file_name)
            sheet_1 = img.get_image_dask_data("CZYX", T=0)
            sheet_1.rechunk(chunk_sizes[1:])
            sheet_1 = sheet_1.astype(np.uint32)

            # fuse
            mean_sheet = (sheet_0 + sheet_1) * 0.5
            mean_sheet = mean_sheet.astype(np.uint16)

            # assemble each view into tmp_big_stack
            for c in range(len(channel_names)):
                sz, sy, sx = mean_sheet[c].shape
                print("writing c " + str(c))
                tmp_big_stack[t, c, z[v]:z[v] + sz, y[v]:y[v] + sy, x[v]:x[v] + sx] = mean_sheet[c]

            print("time point " + str(t) +": completed view number " + str(v))

    # crop out black
    crop_padding(str(tmp_zarr_path))

    #
    #
    # # compute pyramid structure
    # print("creating pyramid structure")
    # # create output ome-zarr
    # root = create_pyramid_from_zarr(path_to_plain_zarr, 'tmp_big_stack', path_to_new_zarr, pyramid_scales, chunk_sizes):
    # print("done!")

    return tmp_zarr
    #return root


def crop_padding(tmp_zarr_path, slicing=None):
    tmp_root = zarr.open(tmp_zarr_path, 'r')
    tmp_big_stack = tmp_root.get('tmp_big_stack')
    if slicing is None:
        slicing = get_nonzero_slicing_range_ome(tmp_zarr_path, 'tmp_big_stack')
    cropped_size_z = slicing[0][1] - slicing[0][0]
    cropped_size_y = slicing[1][1] - slicing[1][0]
    cropped_size_x = slicing[2][1] - slicing[2][0]
    cropped_shape = (tmp_big_stack.shape[0], tmp_big_stack.shape[1],) + (cropped_size_z, cropped_size_y, cropped_size_x)
    print(cropped_shape)
    crop_store = zarr.DirectoryStore(Path(tmp_zarr_path).parent / 'tmp.crop.zarr')
    crop_root = zarr.group(store=crop_store, overwrite=False)
    crop_root.create_dataset('tmp_crop', shape=cropped_shape, dtype=tmp_big_stack.dtype, chunks=(1, 1, 1, cropped_shape[3], cropped_shape[4]))
    for t in range(cropped_shape[0]):
        for c in range(cropped_shape[1]):
            crop_root.tmp_crop[t, c] = tmp_big_stack[t, c, slicing[0][0]:slicing[0][1], slicing[1][0]:slicing[1][1], slicing[2][0]:slicing[2][1] ]
    # da_big = da.from_zarr(tmp_big_stack)
    # da_big.rechunk(chunks=(1, 1, cropped_shape[2], cropped_shape[3], cropped_shape[4]))
    # print(da_big.chunks)
    # da_cropped = da_big[:, :, slicing[0][0]:slicing[0][1], slicing[1][0]:slicing[1][1], slicing[2][0]:slicing[2][1]]
    # print(da_cropped.chunks)
    # da_cropped.to_zarr(crop_store, store=crop_root.get('tmp_crop').store, chunks=(1, 1, cropped_shape[2], cropped_shape[3], cropped_shape[4]))

    return


def create_pyramid_from_zarr(path_to_plain_zarr, group_name, path_to_new_zarr, pyramid_scales, chunk_sizes=None):
    """ assumes TCZYX array structure"""
    store = zarr.DirectoryStore(path_to_new_zarr)
    root: zarr.Group = zarr.group(store=store, overwrite=False)

    plain = zarr.open(path_to_plain_zarr, 'r')
    arr = plain.get(group_name)
    arr_shape = arr.shape
    num_time_points = arr_shape[0]
    num_channels = arr_shape[1]
    num_slices = arr_shape[2]
    if chunk_sizes is None:
        chunk_sizes = arr.chunks

    for t in range(num_time_points):
        for c in range(num_channels):
            for zslice in range(num_slices):
                print('t: ' + str(t) + 'c: ' + str(c) + 'making pyramid for z = ' + str(zslice))
                region = arr[t, c, zslice]
                pyramid = tiled_pyramid_gaussian(
                    region,
                    max_layer=pyramid_scales - 1,
                    downscale=2,
                    preserve_range=True,
                    tile_size=(32768, 32768,),
                )

                for i, im in enumerate(pyramid):
                    if zslice == 0 and t == 0 and c == 0:
                        root.create_dataset(
                            str(i),
                            shape=(num_time_points, num_channels, num_slices, *im.shape),
                            dtype=region.dtype,
                            chunks=chunk_sizes,
                        )

                    root[str(i)][t, c, zslice] = im

    return root


def compute_output_slicing(
    original_start: Tuple[int], downscale: int, tile_size: Tuple[int]
) -> Tuple[slice]:

    """Computes downscaled slicing size."""
    original_start = np.asarray(original_start)
    resized_start = tuple(np.round(original_start / downscale).astype(int))
    resized_end = tuple(np.ceil((original_start + tile_size) / downscale).astype(int))
    output_slicing = tuple(slice(s, e) for s, e in zip(resized_start, resized_end))

    return output_slicing


def tiled_pyramid_gaussian(
    image: np.ndarray,
    max_layer: int,
    downscale: int,
    tile_size: Tuple[int],
    **kwargs,
) -> Iterator[np.ndarray]:

    """
    Tiled cupy pyramid gaussian implementation.
    Copied and modified from skimage.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    downscale : int
        Downscale factor.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    tile_size : Tuple[int]
        Tile size for individual processing.
    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.
    """

    assert len(image.shape) == len(tile_size)
    _check_factor(downscale)
    layer = 0
    dtype = image.dtype
    current_shape = image.shape
    prev_layer_image = image

    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1
        tile_size = tuple(min(s, t) for s, t in zip(current_shape, tile_size))
        out_shape = tuple(math.ceil(d / downscale) for d in current_shape)
        layer_image = np.empty(out_shape, dtype=dtype)

        # getting start of slicing from n-dim tiles
        for start in product(
            *[range(0, d, t) for d, t in zip(current_shape, tile_size)]

        ):
            input_slicing = tuple(slice(s, s + t) for s, t in zip(start, tile_size))
            output_slicing = compute_output_slicing(start, downscale, tile_size)
            layer_image[output_slicing] = pyramid_reduce(
                cp.asarray(prev_layer_image[input_slicing]),
                downscale,
                **kwargs,
            ).get()

        prev_shape = current_shape
        prev_layer_image = layer_image
        current_shape = layer_image.shape

        # no change to previous pyramid layer
        if current_shape == prev_shape:
            break

        yield layer_image


def get_nonzero_slicing_range(root, channel_name):
    # x and y
    proj0 = root.get_projection_array(channel_name, axis=0)[0]
    # x
    ids_x = np.where(np.sum(proj0, axis=0) > 0)
    start_x = ids_x[0][0]
    end_x = ids_x[0][-1]

    # y
    ids_y = np.where(np.sum(proj0, axis=1) > 0)
    start_y = ids_y[0][0]
    end_y = ids_y[0][-1]

    # z
    proj1 = root.get_projection_array(channel_name, axis=1)[0]
    ids_z = np.where(np.sum(proj1, axis=1) > 0)
    start_z = ids_z[0][0]
    end_z = ids_z[0][-1]

    print('x = ' + str(start_x) + ':' + str(end_x))
    print('y = ' + str(start_y) + ':' + str(end_y))
    print('z = ' + str(start_z) + ':' + str(end_z))

    slicing = ((start_z, end_z), (start_y, end_y), (start_x, end_x))

    return slicing


def get_nonzero_slicing_range_ome(path_to_zarr, group_name):
    root = zarr.open(path_to_zarr, 'r')
    arr = root.get(group_name)
    d = da.from_zarr(arr)
    # x and y
    proj0 = da.max(d[0, 0], axis=0).compute()
    # x
    ids_x = np.where(np.sum(proj0, axis=0) > 0)
    start_x = ids_x[0][0]
    end_x = ids_x[0][-1]

    # y
    ids_y = np.where(np.sum(proj0, axis=1) > 0)
    start_y = ids_y[0][0]
    end_y = ids_y[0][-1]

    # z
    proj1 = da.max(d[0, 0], axis=1).compute()
    ids_z = np.where(np.sum(proj1, axis=1) > 0)
    start_z = ids_z[0][0]
    end_z = ids_z[0][-1]

    print('x = ' + str(start_x) + ':' + str(end_x))
    print('y = ' + str(start_y) + ':' + str(end_y))
    print('z = ' + str(start_z) + ':' + str(end_z))

    slicing = ((start_z, end_z), (start_y, end_y), (start_x, end_x))

    return slicing


def check_big_shape(big_shape, x, y, z, view_size):
    # x
    these_coords = x
    this_dim = 2
    this_scan_size = view_size[this_dim]
    max_needed_size = np.max(these_coords) + this_scan_size - np.min(these_coords)
    assert max_needed_size + these_coords[0] <= big_shape[this_dim], f'x mismatch in big_size'

    # y
    these_coords = y
    this_dim = 1
    this_scan_size = view_size[this_dim]
    max_needed_size = np.max(these_coords) + this_scan_size - np.min(these_coords)
    assert max_needed_size + these_coords[0] <= big_shape[this_dim], f'y mismatch in big_size'

    # z
    these_coords = z
    this_dim = 0
    this_scan_size = view_size[this_dim]
    max_needed_size = np.max(these_coords) + this_scan_size - np.min(these_coords)
    assert max_needed_size + these_coords[0] <= big_shape[this_dim], f'z mismatch in big_size'

    return


def ZDataset_to_hdf5(path_to_zarr, path_to_new_hdf5, channels):
    ds = ZDataset(path_to_zarr, mode="r-")
    for channel in channels:
        data = da.array(ds.get_array(channel))
        da.to_hdf5(path_to_new_hdf5, '/' + channel, data)

    return


def ZDataset_to_individual_hdf5s(path_to_zarr, path_to_new_hdf5_dir, channels, timepoints=None, z_slices=None):
    """convert part of a ZDataset to a dir of h5 files, one for each z slice."""
    ds = ZDataset(path_to_zarr, mode="r")
    for channel in channels:
        data = da.array(ds.get_array(channel))
        if timepoints is None:
            timepoints = range(data.shape[0])
        if z_slices is None:
            z_slices = range(data.shape[1])
        for t in timepoints:
            print(t)
            os.mkdir(path_to_new_hdf5_dir + '/scan_' + str(t))

            # note: issue with tqdm and napari
            #for z in tqdm(z_slices, 'converting slices for time: ' + str(t)):
            for counter, z in enumerate(z_slices):
                print(str(counter) + ' of ' + str(len(z_slices)))
                this_slice = data[t, z]
                da.to_hdf5(path_to_new_hdf5_dir + '/scan_' + str(t) + '/z_' + str(z) + '.h5', '/' + channel, this_slice)

    return


def individual_hdf5s_to_ZDataset(path_to_hdf5_dir, path_to_new_zarr, z_slices, prefix, suffix, dtype):
    """one timepoint for now"""
    file_names = glob.glob(path_to_hdf5_dir + '/*.h5')
    if z_slices is None:
        z_slices = range(len(file_names))

    ds = ZDataset(path_to_new_zarr, mode='w-')

    # get channel and shape info from first file
    channels = h5py.File(file_names[0]).keys()
    for channel in channels:
        slice_shape = np.squeeze(h5py.File(file_names[0]).get(channel)).shape
        stack_shape = (1, len(z_slices)) + slice_shape
        ds.add_channel(channel, shape=stack_shape, dtype=dtype, chunks=(1, 1) + slice_shape)
        ds.write_array(channel, zarr.zeros(stack_shape))

    # loop over slices and write to ZDataset
    for channel in channels:
        this_stack = ds.get_array(channel)
        for z, z_slice in enumerate(z_slices):
            print(str(z) + ' of ' + str(len(z_slices)))
            file_name = path_to_hdf5_dir + '/' + prefix + str(z_slice) + suffix
            f = h5py.File(file_name)
            this_stack[0, z] = zarr.array(np.squeeze(f.get(channel)[0]))

    ds.close()

    return

