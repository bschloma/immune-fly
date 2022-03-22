"""functions for reading and writing image data"""


import numpy as np
from aicsimageio import AICSImage, readers
#from dexp.datasets import ZDataset
import zarr
import glob
import dask.array as da
from matplotlib.pyplot import imread

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


def zeiss_filename(core_name, suffix, iteration):
    if iteration == 0:
        out_name = core_name + suffix
    else:
        out_name = core_name + '(' + str(iteration) + ')' + suffix

    return out_name


def convert_czi_views_to_zarr(path_to_czi_dir, path_to_new_zarr, namestr, core_name, suffix=r'.czi', chunk_sizes=(1, 64, 256, 256), channel_names=None):
    """function to read in a folder of czi files, each being one view/tile, with aicsimageio, extract numpy array, then save as zarr with prescribed chunking.
    Each view/channel pair becomes its own channel in ZDataset. Uses file names specific to z1 output. Assumes 2 light sheets

    Note:   code is good, works for view0, but issue with view1 and higher. for some reason the bioformats metadata has dimC=16 for those files ???
            Need to just save a separate czi for each view and illumination"""

    # get file names in czi dir
    filenames = glob.glob(path_to_czi_dir + '/' + namestr)
    numfiles = len(filenames)

    # create zarr
    root = zarr.open(path_to_new_zarr, mode='w-')

    # assume 2 sheets
    num_sheets = 2

    if channel_names is None:
        # get channel names from first file
        first_file = zeiss_filename(core_name, suffix, 0)
        img = AICSImage(path_to_czi_dir + '/' + first_file, reader=readers.bioformats_reader.BioformatsReader)
        channel_names = img.channel_names

    # create channels Reader. assume all channels have same shape
    for f in range(numfiles):
        print("collecting view " + str(f) + ' of ' + str(numfiles-1))
        this_file_name = zeiss_filename(core_name, suffix, f)
        #img = AICSImage(path_to_czi_dir + '/' + this_file_name, reader=readers.bioformats_reader.BioformatsReader)
        # use default czi reader
        img = AICSImage(path_to_czi_dir + '/' + this_file_name)
        # update sheet number (successive files are different sheets, until all sheets are reached)
        s = np.mod(f, num_sheets)
        if s == 0:
            view = root.create_group("View" + str(f/num_sheets))
        sheet = view.create_group("Sheet" + str(s))
        for c in range(len(channel_names)):
            channel = sheet.create_group(channel_names[c])
            for t in range(img.shape[0]):
                this_data = img.get_image_dask_data("ZYX", C=c, T=t)
                this_data.rechunk(chunk_sizes)
                arr = channel.create(name='T' + str(t), shape=this_data.shape)
                da.to_zarr(this_data, arr)
        del img


def convert_tiffs_to_zarr(im_dir, path_to_new_zarr, chunk_sizes=(1, 64, 256, 256)):
    """convert folder of tiffs to zarr. just for testing registration for now, not ideal as long term system. assumes a dir structure"""
    # create zarr
    root = zarr.open(path_to_new_zarr, mode='w-')

    # get regions
    region_dirs = glob.glob(im_dir + '/region*')

    # loop
    for r in range(len(region_dirs)):
        print("collecting view " + str(r) + ' of ' + str(len(region_dirs)-1))

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
                        #channel = sheet.create("Channel" + str(c), shape=(len(slices), this_im.shape[0], this_im.shape[1]))
                        im_stack = np.zeros((len(slices), this_im.shape[0], this_im.shape[1]))
                    im_stack[z] = this_im

                if len(slices) > 0:
                    im_stack_da = da.array(im_stack)
                    im_stack_da.rechunk(chunk_sizes)
                    channel = sheet.create(name="Channel" + str(c), shape=im_stack_da.shape)
                    da.to_zarr(im_stack_da, channel)

    return root




