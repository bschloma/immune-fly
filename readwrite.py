"""functions for reading and writing image data"""

import numpy as np
from aicsimageio import AICSImage, readers
import zarr
import glob
import dask.array as da
from matplotlib.pyplot import imread
from dexp.datasets import ZDataset
from pathlib import Path
from tqdm import tqdm


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

    # convert positions to microns, then to pixels, and center on first scan
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
        for v in tqdm(range(num_views), 'converting time point ' + str(t) + ' of ' + str(num_time_points - 1)):
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
            sheet_1 = sheet_0.astype(np.uint32)

            # fuse
            mean_sheet = (sheet_0 + sheet_1) * 0.5
            mean_sheet = mean_sheet.astype(np.uint16)

            # assemble each view into tmp_big_stack
            for c in range(len(channel_names)):
                sz, sy, sx = mean_sheet[c].shape
                tmp_big_stack[c, z[v]:z[v] + sz, y[v]:y[v] + sy, x[v]:x[v] + sx] = mean_sheet[c]

        for c in range(len(channel_names)):
            grp = channel_names[c]
            root.write_stack(grp, t, da.from_zarr(tmp_zarr.tmp_big_stack)[c])

    return root


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
