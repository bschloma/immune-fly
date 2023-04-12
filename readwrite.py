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
from cucim.skimage.transform import pyramid_reduce, pyramid_gaussian
from cucim.skimage.transform.pyramids import _check_factor
import math
from scipy.optimize import least_squares
import yaml
from pytorch3dunet.unet3d.model import get_model
import torch
from dask.diagnostics import ProgressBar

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


def convert_czi_views_to_fuse_reg_ome_zarr(path_to_czi_dir, path_to_new_zarr, num_time_points, num_views, num_sheets, namestr,
                                     core_name, suffix=r'.czi', chunk_sizes=(1, 64, 256, 256),
                                    pyramid_scales=5, reversed_y = False, stage_positions=None, sx=1920, sy=1920,
                                           dtype='uint16', num_channels=2):
    """function to read in a folder of czi files, compute fusion + registration, then save as ome-zarr with prescribed chunking and pyramiding. Assumes file structure:
    each czi file can contain multiple channels, but different sheets, views, and timepoints are separate files. Pyramid computation based on code by Jordao Bragantini.

    Notes:   NEED TO TEST"""

    # get file names in czi dir
    filenames = glob.glob(path_to_czi_dir + '/' + namestr)
    num_files = len(filenames)

    # can pass in a dataframe, stage_positions, with x, y, z postions of views in microns.
    if stage_positions is None:
        # get info from first file
        first_file = zeiss_filename(core_name, suffix, 0)
        #first_img = AICSImage(path_to_czi_dir + '/' + first_file, reader=readers.bioformats_reader.BioformatsReader)
        # trying a new way where I manually copy the 0th .czi file to parent directory, to see if this avoids memory issue
        first_img = AICSImage(Path(path_to_czi_dir).parent.__str__() + '/' + first_file, reader=readers.bioformats_reader.BioformatsReader)

        # coordinates of each view --- doing maually now to avoid remaking first_img
        """extract xyz stage coordinates from a Bioformats Reader object"""
        x = np.zeros(num_views)
        y = np.zeros(num_views)
        z = np.zeros(num_views)

        for view in range(num_views):
            x[view] = first_img.metadata.images[view].stage_label.x
            y[view] = first_img.metadata.images[view].stage_label.y
            z[view] = first_img.metadata.images[view].stage_label.z

        # pixel sizes --- doing manually now to avoid remaking first_img
        dx = first_img.metadata.images[0].pixels.physical_size_x
        dy = first_img.metadata.images[0].pixels.physical_size_y
        dz = first_img.metadata.images[0].pixels.physical_size_z

        # convert positions to microns, then to pixels, and center on first scan.
        # try ordering the stacks one way. If that leads to negative indices, order them the other way.
        x = np.int16(1e6 * (x - np.min(x)) / dx)
        y = np.int16(1e6 * (y - np.min(y)) / dy)
        z = np.int16((z - np.min(z)) / dz)

        sx = first_img.dims.X
        sy = first_img.dims.Y
        sz = first_img.dims.Z

        # size of the big array
        lx = np.max(x) + sx
        ly = np.max(y) + sy
        lz = np.max(z) + sz
        big_shape = (lz, ly, lx)

        num_channels = int(first_img.dims.C / num_sheets)

    else:
        # x, y, z already in microns
        positions = stage_positions.loc[:, ['x', 'y', 'z']].values
        x = positions[:, 0].tolist()
        y = positions[:, 1].tolist()
        z = positions[:, 2].tolist()

        x = np.int16((x - np.min(x)))
        y = np.int16((y - np.min(y)))
        z = np.int16((z - np.min(z)))

        sz = stage_positions.loc[:, ['z_slices']].values.tolist()
        sz = [z[0] for z in sz]

        lx = np.max(x) + sx
        ly = np.max(y) + sy
        lz = np.max(z) + np.max(sz)
        big_shape = (lz, ly, lx)

    # create file_id_tree used to pick the right image files
    assert num_files == num_time_points * num_views * num_sheets
    id_tree = create_file_id_hierarchy(num_time_points, num_views, num_sheets)

    # create temporary zarr on disk for registering big stack that we will overwrite and then delete
    tmp_zarr_path = Path(path_to_new_zarr).parent / 'tmp.zarr'
    tmp_zarr = zarr.open(tmp_zarr_path.__str__(), 'w')      # overwritttttttte!
    # note: with bioformats reader reading the first image, dims.C = num_channels * num_sheets. Not true w/ czi reader.
    tmp_big_stack = tmp_zarr.create('tmp_big_stack', shape=(num_time_points, num_channels,) + big_shape, dtype=dtype,
                                    chunks=chunk_sizes)
    # create the array that will correct each image for non-uniform sheet intensity
    # red
    I0, xc, yc, xR, sigma_y = load_best_fit_sheet_params(color='red')
    ygrid, xgrid = np.indices((sy, sx))
    sheet_correction = sheet_intensity(xgrid, ygrid, I0, xc, yc, xR, sigma_y)
    sheet_correction_red = np.expand_dims(sheet_correction / np.max(sheet_correction), axis=0)

    # green
    I0, xc, yc, xR, sigma_y = load_best_fit_sheet_params(color='green')
    ygrid, xgrid = np.indices((sy, sx))
    sheet_correction = sheet_intensity(xgrid, ygrid, I0, xc, yc, xR, sigma_y)
    sheet_correction_green = np.expand_dims(sheet_correction / np.max(sheet_correction), axis=0)

    print("computing fusion and registration")
    for t in range(num_time_points):
        #try:
        y_start = 0
        for v in range(num_views):
            # sheet_0
            s = 0
            file_number = get_file_number_from_tvs(id_tree, t, v, s)
            this_file_name = zeiss_filename(core_name, suffix, file_number)
            img_0 = AICSImage(path_to_czi_dir + '/' + this_file_name)
            #sheet_0 = img.get_image_data("CZYX", T=0).astype(np.uint32)

            # sheet_1. assumes same dims as sheet_0.
            s = 1
            file_number = get_file_number_from_tvs(id_tree, t, v, s)
            this_file_name = zeiss_filename(core_name, suffix, file_number)
            img_1 = AICSImage(path_to_czi_dir + '/' + this_file_name)
            #sheet_1 = img.get_image_data("CZYX", T=0).astype(np.uint32)

            # fuse
            mean_sheet = (img_0.get_image_data("CZYX", T=0).astype(np.uint32) +
                          img_1.get_image_data("CZYX", T=0).astype(np.uint32)) * 0.5

            # rescale red channel
            sheet_correction_stack = np.repeat(sheet_correction_red, mean_sheet[1].shape[0], axis=0)
            mean_sheet[1] = mean_sheet[1] / sheet_correction_stack
            mean_sheet = mean_sheet.astype(np.uint16)

            # rescale green channel
            sheet_correction_stack = np.repeat(sheet_correction_green, mean_sheet[1].shape[0], axis=0)
            mean_sheet[0] = mean_sheet[0] / sheet_correction_stack
            mean_sheet = mean_sheet.astype(np.uint16)

            # remove top part of image
            if v > 0:
                y_start = sy + y[v - 1] - y[v]
            mean_sheet = mean_sheet[:, :, y_start:, :]
            _num_channels, sz, cropped_sy, sx = mean_sheet.shape
            tmp_big_stack[t, :,  z[v]:z[v] + sz, (y[v] + y_start):y[v] + y_start + cropped_sy, x[v]:x[v] + sx] = mean_sheet
            # # # assemble each view into tmp_big_stack
            # for c in range(num_channels):
            #     sz, sy, sx = mean_sheet[c].shape
            #     tmp_big_stack[t, c, z[v]:z[v] + sz, y[v]:y[v] + sy, x[v]:x[v] + sx] = mean_sheet[c]
            #
            #     # trying out blending of two views. average them only if the pixel value is > 0. No errors but need to revisit dtype during mean. images come out weird.
                # current_data_in_stack = da.from_array(tmp_big_stack[t, c, z[v]:z[v] + sz, y[v]:y[v] + sy, x[v]:x[v] + sx])
                # current_data_gtr_0 = current_data_in_stack > 0
                #tmp_big_stack[t, c, z[v]:z[v] + sz, y[v]:y[v] + sy, x[v]:x[v] + sx] = (current_data_in_stack +  mean_sheet[c])/  (1 + current_data_gtr_0)

            print("time point " + str(t) +": completed view number " + str(v))
        #except ValueError:
            #print('error with time point' + str(t) + ', skipping')
            #continue


    # crop out black
    #slicing = get_nonzero_slicing_range_ome(tmp_zarr_path.__str__(), 'tmp_big_stack')
    #crop_padding(tmp_zarr_path.__str__(), slicing=slicing)

    # compute pyramid structure
    print("creating pyramid structure")
    # create output ome-zarr
    #path_to_crop_zarr = tmp_zarr_path.parent / 'tmp.crop.zarr'
    #root = create_pyramid_from_zarr(tmp_zarr_path.__str__(), 'tmp_big_stack', path_to_new_zarr, pyramid_scales, chunk_sizes)
    print("done!")

    return #root


def crop_padding(tmp_zarr_path, slicing=None):
    tmp_root = zarr.open(tmp_zarr_path, 'r')
    tmp_big_stack = tmp_root.get('tmp_big_stack')
    if slicing is None:
        slicing = get_nonzero_slicing_range_ome(tmp_zarr_path, 'tmp_big_stack')
    print(slicing)
    cropped_size_z = slicing[0][1] - slicing[0][0]
    cropped_size_y = slicing[1][1] - slicing[1][0]
    cropped_size_x = slicing[2][1] - slicing[2][0]
    cropped_shape = (tmp_big_stack.shape[0], tmp_big_stack.shape[1],) + (cropped_size_z, cropped_size_y, cropped_size_x)
    print(cropped_shape)
    crop_store = zarr.DirectoryStore((Path(tmp_zarr_path).parent / 'tmp.crop.zarr').__str__())
    crop_root = zarr.group(store=crop_store, overwrite=False)
    crop_root.create_dataset('tmp_crop', shape=cropped_shape, dtype=tmp_big_stack.dtype, chunks=(1, 1, 1,
                                                                                                 int(cropped_shape[3]),
                                                                                                 int(cropped_shape[4])))
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


# def create_pyramid_from_zarr(path_to_plain_zarr, group_name, path_to_new_zarr, pyramid_scales, chunk_sizes=None):
#     """ assumes TCZYX array structure"""
#     store = zarr.DirectoryStore(path_to_new_zarr)
#     root: zarr.Group = zarr.group(store=store, overwrite=False)
#
#     plain = zarr.open(path_to_plain_zarr, 'r')
#     arr = plain.get(group_name)
#     arr_shape = arr.shape
#     num_time_points = arr_shape[0]
#     num_channels = arr_shape[1]
#     num_slices = arr_shape[2]
#     if chunk_sizes is None:
#         chunk_sizes = arr.chunks
#
#     for t in range(num_time_points):
#         for c in range(num_channels):
#             for zslice in range(num_slices):
#                 print('t: ' + str(t) + 'c: ' + str(c) + 'making pyramid for z = ' + str(zslice))
#                 region = arr[t, c, zslice]
#                 pyramid = tiled_pyramid_gaussian(
#                     region,
#                     max_layer=pyramid_scales - 1,
#                     downscale=2,
#                     preserve_range=True,
#                     tile_size=(32768, 32768,),
#                 )
#
#                 for i, im in enumerate(pyramid):
#                     if zslice == 0 and t == 0 and c == 0:
#                         root.create_dataset(
#                             str(i),
#                             shape=(num_time_points, num_channels, num_slices, *im.shape),
#                             dtype=region.dtype,
#                             chunks=chunk_sizes,
#                         )
#
#                     root[str(i)][t, c, zslice] = im
#
#     return root


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
                pyramid = pyramid_gaussian(
                    cp.asarray(region),
                    max_layer=pyramid_scales - 1,
                    downscale=2,
                    preserve_range=True,
                )

                for i, im in enumerate(pyramid):
                    if zslice == 0 and t == 0 and c == 0:
                        root.create_dataset(
                            str(i),
                            shape=(num_time_points, num_channels, num_slices, *im.shape),
                            dtype=region.dtype,
                            chunks=chunk_sizes,
                        )

                    root[str(i)][t, c, zslice] = im.get()

    return root


# def compute_output_slicing(
#     original_start: Tuple[int], downscale: int, tile_size: Tuple[int]
# ) -> Tuple[slice]:
#
#     """Computes downscaled slicing size."""
#     original_start = np.asarray(original_start)
#     resized_start = tuple(np.round(original_start / downscale).astype(int))
#     resized_end = tuple(np.ceil((original_start + tile_size) / downscale).astype(int))
#     output_slicing = tuple(slice(s, e) for s, e in zip(resized_start, resized_end))
#
#     return output_slicing
#
#
# def tiled_pyramid_gaussian(
#     image: np.ndarray,
#     max_layer: int,
#     downscale: int,
#     tile_size: Tuple[int],
#     **kwargs,
# ) -> Iterator[np.ndarray]:
#
#     """
#     Tiled cupy pyramid gaussian implementation.
#     Copied and modified from skimage.
#
#     Parameters
#     ----------
#     image : np.ndarray
#         Input image.
#     downscale : int
#         Downscale factor.
#     max_layer : int
#         Number of layers for the pyramid. 0th layer is the original image.
#         Default is -1 which builds all possible layers.
#     tile_size : Tuple[int]
#         Tile size for individual processing.
#     Returns
#     -------
#     pyramid : generator
#         Generator yielding pyramid layers as float images.
#     """
#
#     #assert len(image.shape) == len(tile_size)
#     assert len(image.shape[1:]) == len(tile_size)
#     _check_factor(downscale)
#     layer = 0
#     dtype = image.dtype
#     current_shape = image.shape
#     current_shape = image.shape[1:]
#     prev_layer_image = image
#
#     yield image
#
#     # build downsampled images until max_layer is reached or downscale process
#     # does not change image size
#     while layer != max_layer:
#         layer += 1
#         tile_size = tuple(min(s, t) for s, t in zip(current_shape, tile_size))
#         out_shape = tuple(math.ceil(d / downscale) for d in current_shape)
#         layer_image = np.empty(out_shape, dtype=dtype)
#
#         # getting start of slicing from n-dim tiles
#         for start in product(
#             *[range(0, d, t) for d, t in zip(current_shape, tile_size)]
#
#         ):
#             input_slicing = tuple(slice(s, s + t) for s, t in zip(start, tile_size))
#             output_slicing = compute_output_slicing(start, downscale, tile_size)
#             layer_image[output_slicing] = pyramid_reduce(
#                 cp.asarray(prev_layer_image[input_slicing]),
#                 downscale,
#                 **kwargs,
#             ).get()
#
#         prev_shape = current_shape
#         prev_layer_image = layer_image
#         current_shape = layer_image.shape
#
#         # no change to previous pyramid layer
#         if current_shape == prev_shape:
#             break
#
#         yield layer_image


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


def check_big_shape(big_shape, x, y, z, view_size, reversed_y=False):
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
    if reversed_y:
        y_id = -1
    else:
        y_id = 0
    assert max_needed_size + these_coords[y_id] <= big_shape[this_dim], f'y mismatch in big_size'

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


def ome_zarr_to_individual_hdf5s(path_to_zarr, path_to_new_hdf5_dir, channel_numbers, channel_names, timepoints=None, z_slices=None):
    """convert highest res of ome-zarr to a dir of h5 files, one for each z slice. channel_numbers = list of channel
    indices in zarr array. channel_names = list, each element the name of the corresponding channel number in
    channel_numbers."""
    data = da.from_zarr(path_to_zarr + '/0')
    if timepoints is None:
        timepoints = range(data.shape[0])
    if z_slices is None:
        z_slices = range(data.shape[2])
    for t in timepoints:
        print(t)
        os.mkdir(path_to_new_hdf5_dir + '/scan_' + str(t))
        for counter_c, c in enumerate(channel_numbers):
            for counter_z, z in enumerate(z_slices):
                print(str(counter_z) + ' of ' + str(len(z_slices)))
                this_slice = data[t, c, z]
                da.to_hdf5(path_to_new_hdf5_dir + '/scan_' + str(t) + '/z_' + str(z) + '.h5', '/' + channel_names[counter_c], this_slice)

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


def individual_hdf5s_to_zarr(path_to_hdf5_dir, path_to_new_zarr, z_slices, prefix, suffix, dtype):
    """one timepoint for now. assumes 1 channel"""
    file_names = glob.glob(path_to_hdf5_dir + '/*.h5')
    if z_slices is None:
        z_slices = range(len(file_names))

    # get channel and shape info from first file
    channels = h5py.File(file_names[0]).keys()
    slice_shape = np.squeeze(h5py.File(file_names[0]).get(list(channels)[0])).shape   # assumes one channel
    stack_shape = (1, 1, len(z_slices)) + slice_shape
    pred_zarr = zarr.open(path_to_new_zarr, 'w-')
    pred = pred_zarr.create('pred', shape=stack_shape, dtype=dtype, chunks=(1, 1, 1) + slice_shape)


    # loop over slices and write to ZDataset
    #this_stack = pred[0, 0]
    for z, z_slice in enumerate(z_slices):
        print(str(z) + ' of ' + str(len(z_slices)))
        file_name = path_to_hdf5_dir + '/' + prefix + str(z_slice) + suffix
        f = h5py.File(file_name)
        print(f.get(list(channels)[0]).shape)
        #print(this_stack.shape)
        pred[0, 0, z] = f.get(list(channels)[0])[0, 0]

    return


def individual_gasp_hdf5s_to_zarr(path_to_hdf5_dir, path_to_new_zarr, z_slices, prefix, suffix, dtype):
    """one timepoint for now. assumes 1 channel"""
    file_names = glob.glob(path_to_hdf5_dir + '/z*.h5')
    if z_slices is None:
        z_slices = range(len(file_names))

    # get channel and shape info from first file
    channels = h5py.File(file_names[0]).keys()
    slice_shape = np.squeeze(h5py.File(file_names[0]).get(list(channels)[0])).shape   # assumes one channel
    stack_shape = (1, 1, len(z_slices)) + slice_shape
    gasp_zarr = zarr.open(path_to_new_zarr, 'w-')
    gasp = gasp_zarr.create('gasp', shape=stack_shape, dtype=dtype, chunks=(1, 1, 1) + slice_shape)


    # loop over slices and write to ZDataset
    #this_stack = pred[0, 0]
    for z, z_slice in enumerate(z_slices):
        print(str(z) + ' of ' + str(len(z_slices)))
        file_name = path_to_hdf5_dir + '/' + prefix + str(z_slice) + suffix
        f = h5py.File(file_name)
        print(f.get(list(channels)[0]).shape)
        #print(this_stack.shape)
        gasp[0, 0, z] = f.get(list(channels)[0])[0]

    return


def fit_sheet(data):
    """Returns params found by a fit"""
    params = get_initial_sheet_param_estimates(data)

    def error_function(p): return np.ravel(
        sheet_intensity(np.indices(data.shape)[1], np.indices(data.shape)[0], p[0], p[1],
                        p[2], p[3], p[4]) - data)

    result = least_squares(error_function, params, bounds=(0, np.inf), method='trf')

    return result


def sheet_intensity(x, y, I0, xc, yc, xR, sigma_y):
    I = I0 * (1 / (1 + ((x - xc) / xR) ** 2)) * np.exp(-0.5 * ((y - yc) / sigma_y) ** 2)

    return I


def get_initial_sheet_param_estimates(data):
    """assume 2d input"""
    I0 = np.max(data)
    xc = data.shape[1] / 2
    yc = data.shape[0] / 2
    xR = 0.75 * data.shape[1]
    sigma_y = 0.75 * data.shape[0]

    return I0, xc, yc, xR, sigma_y


def load_best_fit_sheet_params(color):
    """I0, xc, yc, xR, sigma_y"""
    if color == 'red':
        return [6227.21838549834, 931.5715074820621, 1040.3644589367918, 1290.8884641580382, 760.6163552682698]
    elif color == 'green':
        return [53015.42534478,   958.02432563,   960.56444003,  1281.66615808, 749.00575153]



def load_plantseg_model(model_path=r'/home/brandon/Documents/Code/immune-fly/2dunet_bce_dice_ds3x'):
    config_path = model_path + '/config_train.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_config = config.pop('model')
    model = get_model(model_config)

    return model


def plantseg_predict(chunk, model):
    chunk = chunk[0]
    #chunk = normalize_std(normalize_0_255(chunk))
    if chunk.shape[0] > 1:
        t = torch.tensor(chunk.astype('float32'), device='cuda')
        t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(t, dim=0), dim=0), dim=0)
        # call model in test model
        with torch.no_grad():
            t = model(t)

        t = np.array(t.cpu()).squeeze()
        t = np.expand_dims(t, axis=0)
    else:
        t = np.expand_dims(chunk, axis=0)

    return t


def run_plantseg_predict(path_to_zarr, channel, path_to_pred):
    im = zarr.open(path_to_zarr, 'r')[:, channel]
    model = load_plantseg_model()
    model = model.to('cuda')

    for t in range(im.shape[0]):
        print(f't = {t}')
        im_da = da.from_array(im[t, 100:110], chunks=(1, 256, 256))
        prediction = da.map_blocks(plantseg_predict, im_da, model, dtype='float32')
        #prediction = da.map_overlap(plantseg_predict, im_da, depth=(1, 80, 80), model=model, dtype='float32')
        with ProgressBar():
            prediction.to_zarr(path_to_pred)

    return


def normalize_01(data: np.array) -> np.array:
    """
    copied from plantseg. normalize a numpy array between 0 and 1
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12).astype('float32')


def normalize_m11(data: np.array) -> np.array:
    """
    copied from plantseg. normalize a numpy array between -1 and 1
    """
    return 2 * normalize_01(data) - 1


def normalize_0_255(data: np.array) -> np.array:
    """
    copied from plantseg. normalize a numpy array between -1 and 1
    """
    return (255 * normalize_01(data)).astype('uint8')


def normalize_std(data):
    data = np.float32(data)
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / np.clip(std, a_min=1e-12, a_max=None)


def get_mips(dask_array, channel):
    this_data = dask_array[:, channel, :]
    mip = da.max(this_data, axis=1)

    return mip.compute()
