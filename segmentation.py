import numpy as np
import pandas as pd
import cupy as cp
import dask.array as da
from dask.diagnostics import ProgressBar
from dask_image.ndfilters import gaussian_filter
from dask_image.ndmeasure import area, labeled_comprehension
from dask_image.ndmeasure import label as da_label
from cucim.skimage.filters import gaussian, frangi, difference_of_gaussians
from cucim.skimage.measure import label
from cucim.skimage.morphology import binary_opening, binary_erosion, disk, white_tophat, binary_closing, \
    remove_small_holes
# from dexp.datasets import ZDataset
from skimage.segmentation import watershed
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt  # , label
import zarr
from tqdm import tqdm
from os.path import exists
from zarr.storage import DirectoryStore
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from glob import glob
from skimage.io import imread


def init_frangi_params():
    params = pd.DataFrame()
    params.sigma_blur = 3.0
    params.sigmas = 8.0
    params.alpha = 0.5
    params.beta = 0.8
    params.gamma = 0.8

    return params


def cp_make_boundaries(arr, method="frangi", params=init_frangi_params()):
    """ arr is a ZYX cupy array """
    # unpack this zslice
    arr = arr[0]

    if method == "frangi":
        # apply some filters
        arr = gaussian(arr, sigma=params.sigma_blur)
        arr = frangi(arr, sigmas=params.sigmas, alpha=params.alpha, beta=params.beta, gamma=params.gamma)

        # scale 0 to 1
        arr = (arr - cp.min(arr)) / (cp.max(arr) - cp.min(arr))
    else:
        raise NotImplementedError

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def cp_make_seeds(arr, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """gpu based method for seeds. inverse frangi mask"""
    # unpack this zslice
    arr = arr[0]

    if method == "frangi":
        og_arr = arr
        # apply some filters
        arr = gaussian(arr, sigma=params.sigma_blur)
        arr = frangi(arr, sigmas=params.sigmas, alpha=params.alpha, beta=params.beta, gamma=params.gamma)

        # create and apply a mask
        arr = arr > thresh
        arr = arr * (og_arr > 3000)
        arr = binary_opening(arr, disk(9))

        # arr = arr * og_arr
        arr = label(arr)

    else:
        raise NotImplementedError

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def make_seeds_distance(arr, seed_inten_thresh=0.4, seed_distance_thresh=0.5):
    """use dask around scipy.ndimage function. np backed arrays"""
    # arr = gaussian_filter(arr, sigma=2.0)
    arr = (arr - da.min(arr)) / (da.max(arr) - da.min(arr))
    arr = arr < seed_inten_thresh
    arr = distance_transform_edt(arr)
    # arr = (arr - da.min(arr)) / (da.max(arr) - da.min(arr))
    # arr = arr > seed_distance_thresh
    # arr = label(arr)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def run_2D_watershed(arr, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """main watershed fcn. arr is a ZYX dask array"""
    # boundaries: use gpu
    # arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    # boundaries = da.map_blocks(cp_make_boundaries, arr_cu, method, params, dtype=np.float32)
    # boundaries = boundaries.map_blocks(cp.asnumpy, meta=boundaries, dtype=np.float32)
    # seeds = da.map_blocks(cp_make_seeds, arr_cu, method, params, thresh, dtype=np.float32)
    # seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)
    # labels = da.map_blocks(watershed, boundaries, seeds)

    arr_cu = arr.map_overlap(to_gpu, dtype=np.float32, depth={0: 0, 1: 1, 2: 0})
    boundaries = da.map_overlap(cp_make_boundaries, arr_cu, dtype=np.float32, depth={0: 0, 1: 1, 2: 0}, method=method,
                                params=params)
    boundaries = boundaries.map_overlap(cp.asnumpy, meta=boundaries, dtype=np.float32, depth={0: 0, 1: 1, 2: 0})
    seeds = da.map_overlap(cp_make_seeds, arr_cu, dtype=np.float32, depth={0: 0, 1: 1, 2: 1}, method=method,
                           params=params, thresh=thresh)
    seeds = seeds.map_overlap(cp.asnumpy, meta=seeds, dtype=np.float32, depth={0: 0, 1: 1, 2: 0})
    labels = da.map_overlap(watershed, boundaries, seeds, depth={0: 0, 1: 1, 2: 0})

    return labels


def filter_segments_by_size_2D(labels, min_area, max_area):
    """dask array input"""
    ids = da.unique(labels)
    areas = area(labels, labels, index=np.uint16(ids))
    bad_ids = ids[[areas < min_area] or [areas > max_area]]
    for bad_id in bad_ids:
        labels[labels == bad_id] = 0

    return labels


def make_boundaries(arr, method="frangi", params=init_frangi_params()):
    """run just the boundary prediction. inputs and outputs are dask arrays"""
    arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    boundaries = da.map_blocks(cp_make_boundaries, arr_cu, method, params, dtype=np.float32)
    boundaries = boundaries.map_blocks(cp.asnumpy, meta=boundaries, dtype=np.float32)

    return boundaries


def make_seeds(arr, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """run just seeds. inputs and outputs are dask arrays"""
    arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    seeds = da.map_blocks(cp_make_seeds, arr_cu, method, params, thresh, dtype=np.float32)
    seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)

    return seeds


def run_ws_from_boundaries(arr, boundaries, method="frangi", params=init_frangi_params(), thresh=1E-6):
    """pass boundaries as input. inputs and outputs are dask arrays"""
    arr_cu = arr.map_blocks(to_gpu, dtype=np.float32)
    seeds = da.map_blocks(cp_make_seeds, arr_cu, method, params, thresh, dtype=np.float32)
    seeds = seeds.map_blocks(cp.asnumpy, meta=seeds, dtype=np.float32)
    labels = da.map_blocks(watershed, boundaries, seeds)

    return labels


def simple_make_seeds_from_boundaries(boundaries):
    seeds = sk_label(boundaries < 0.1)

    return seeds


def make_seeds_arr_from_points(points_df, path_to_seeds_zarr, shape):
    seeds = zarr.zeros(shape, store=path_to_seeds_zarr)
    for i in range(len(points_df)):
        seeds[np.uint16(points_df[0][i]), np.uint16(points_df[1][i]), np.uint16(points_df[2][i])] = i + 1

    return


def create_bacteria_labels(arr, path_to_bacteria_labels_zarr):
    raise NotImplementedError

    return


def make_mem_mask(dask_arr, sigma_blur, beta, gamma, sigma_frangi, thresh):
    arr_cu = dask_arr.map_blocks(to_gpu, dtype=np.float32)
    mask = da.map_blocks(cp_make_mem_mask, arr_cu, sigma_blur, beta, gamma, sigma_frangi, thresh, dtype=np.float32)
    mask = mask.map_blocks(cp.asnumpy, meta=mask, dtype=np.uint8)

    return mask


def cp_make_mem_mask(arr, sigma_blur, beta, gamma, sigma_frangi, thresh):
    # unpack this zslice
    arr = arr[0]

    # drop background
    arr[arr < 0.005] = 0

    # # apply some filters
    arr = gaussian(arr, sigma=sigma_blur)
    # if sigma_frangi > 0:
    #    arr = frangi(arr, sigmas=sigma_frangi, alpha=0.5, beta=beta, gamma=gamma)

    # create and apply a mask
    arr = arr > thresh
    arr = binary_closing(arr, disk(39))
    arr = remove_small_holes(arr, 15000)

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def create_mip_mask_by_mem(path_to_data_zarr, sigma_blur, beta, gamma, sigma_frangi, thresh, time_point=0, channel=0,
                           mem=None):
    """optionally pass mem dask array. otherwise mem is taken to be channel 1 of data zarr."""
    stack = da.from_zarr(path_to_data_zarr)[time_point]
    im = stack[channel]
    if mem is None:
        mem = stack[1]
    mask = make_mem_mask(mem, sigma_blur, beta, gamma, sigma_frangi, thresh)
    masked_im = im * mask
    with ProgressBar():
        masked_mip = da.max(masked_im, axis=0).compute()
    # im = zarr.open(path_to_data_zarr, 'r')[time_point, channel]
    # mem = zarr.open(path_to_mem_zarr, 'r')[0, 0]
    # tmp_masked_im = np.zeros(im.shape)
    # for i in range(im.shape[0]):
    #     mem_cu = cp.asarray(mem[i])
    #     mem_cu = cp_make_mem_mask(mem_cu, sigma_blur, beta, gamma, sigma_frangi, thresh)
    #     this_mem = cp.asnumpy(mem_cu)
    #     tmp_masked_im[i] = this_mem * im[i]
    #
    # masked_mip = np.max(tmp_masked_im, axis=0)

    return masked_mip


def mask_ecr(path_to_ome_zarr, path_to_output_labels_zarr, path_to_masked_zarr, sigma_low, sigma_high, thresh,
             maximum_size):
    if not exists(path_to_output_labels_zarr):
        segment_ecr(path_to_ome_zarr, path_to_output_labels_zarr, sigma_low, sigma_high, thresh, maximum_size)
    mask_da = da.from_zarr(path_to_output_labels_zarr)
    im_da = da.from_zarr(path_to_ome_zarr + '/0')[0, 0]

    masked_im_da = im_da * mask_da
    masked_im_da.to_zarr(path_to_masked_zarr)
    # masked_zarr = zarr.create(store=path_to_masked_zarr, shape=im_da.shape,
    #                           chunks=(1, im_da.shape[1], im_da.shape[2]), dtype='uint16')
    # masked_zarr[:] = im_da * mask_da

    return


def segment_ecr(path_to_ome_zarr, path_to_output_labels_zarr, sigma_low, sigma_high, thresh, maximum_size):
    # TEMP crop for testing.
    im_da = da.from_zarr(path_to_ome_zarr + '/0')[0, 0]
    with ProgressBar():
        mask = gpu_process_ecr(im_da, sigma_low, sigma_high, thresh).compute()

    """label matrix is needing too much memory...need to redo whole approach using 3D chunks. For now, just save mask."""
    # print('creating label matrix')
    # labels = sk_label(mask)
    # print("done wiht label matrix")
    #
    # # filter by object size via regionprops
    # regions = regionprops(labels)
    # for i in tqdm(range(len(regions))):
    #     if regions[i].area > maximum_size:
    #         labels[labels == i] = 0
    labels = mask
    segmentation = zarr.create(store=path_to_output_labels_zarr, shape=labels.shape,
                               chunks=(1, labels.shape[1], labels.shape[2]), dtype='uint16')
    segmentation[:] = labels

    return


def ecr_filter(arr, sigma_low, sigma_high, thresh):
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    arr = difference_of_gaussians(arr, sigma_low, sigma_high)
    # create and apply a mask
    arr = arr > thresh

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def gpu_process_ecr(darr, sigma_low, sigma_high, thresh):
    # lazy move to gpu
    mem_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(ecr_filter, mem_cu, sigma_low, sigma_high, thresh, dtype=np.float32)

    # actual compute step for visualization
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint16)

    return filt_np


def segment_nuclei(path_to_ome_zarr, path_to_output_labels_zarr, sigma, thresh, time_points=None, maximum_size=None,
                   channel=1, sigma_low=None, opening_size=None, dask_label=False):
    im_da = da.from_zarr(DirectoryStore(path_to_ome_zarr + '/0'))[:, channel]
    if time_points is not None:
        im_da = im_da[time_points]
    else:
        time_points = np.arange(im_da.shape[0])

    if dask_label:
        segmentation = zarr.create(store=path_to_output_labels_zarr, shape=im_da.shape,
                                   chunks=(1, 64, 512, 512), dtype='uint16')
    else:
        segmentation = zarr.create(store=path_to_output_labels_zarr, shape=im_da.shape,
                                   chunks=(1, 1, im_da.shape[1], im_da.shape[2]), dtype='uint16')
    for t in time_points:
        with ProgressBar():
            mask = gpu_process_nuclei(im_da[t], sigma, thresh, sigma_low=sigma_low, opening_size=opening_size).compute()

        # label matrix
        if dask_label:
            mask_da = da.from_array(mask, chunks=(64, 512, 512))
            labels, n_objects = da_label(mask_da)
            with ProgressBar():
                labels = labels.compute()

        else:
            labels = sk_label(mask)

            # filter by object size via regionprops. only works with sk_label for now.
            if maximum_size is not None:
                regions = regionprops(labels)
                for i in tqdm(range(len(regions))):
                    if regions[i].area > maximum_size:
                        labels[labels == i] = 0

        # path_to_tmp_labels = Path(path_to_output_labels_zarr).parent / 'tmp_labels.zarr'
        # labels.to_zarr(path_to_tmp_labels.__str__())
        #
        # labels = zarr.open(DirectoryStore(path_to_tmp_labels.__str__()))
        segmentation[t] = labels

    return


def nuclei_filter(arr, sigma, thresh, sigma_low=None, opening_size=None):
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    arr = gaussian(arr, sigma)

    if sigma_low is not None:
        arr = difference_of_gaussians(arr, low_sigma=sigma_low, high_sigma=3*sigma_low)

    # create and apply a mask
    arr = arr > thresh

    # optional opening
    if opening_size is not None:
        arr = binary_opening(arr, disk(opening_size))

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def gpu_process_nuclei(darr, sigma, thresh, sigma_low=None, opening_size=None):
    # lazy move to gpu
    mem_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(nuclei_filter, mem_cu, sigma, thresh, sigma_low, opening_size, dtype=np.float32)

    # actual compute step
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint8)

    return filt_np


def segment_bacteria(path_to_zarr, path_to_output_labels_zarr, sigma_blur=1.0, bright_thresh=0.04, sigma_low=1.0, sigma_high=2.0, disk_size=3, bacteria_thresh=0.0005, time_points=None, maximum_size=None,
                   channel=1, dask_label=False, sigma_blur_agg=4, disk_size_agg=5, bacteria_thresh_agg=10**-1.8):
    im_da = da.from_zarr(DirectoryStore(path_to_zarr))[:, channel]
    if time_points is not None:
        im_da = im_da[time_points]
    else:
        time_points = np.arange(im_da.shape[0])

    if dask_label:
        segmentation = zarr.create(store=path_to_output_labels_zarr, shape=im_da.shape,
                                   chunks=(1, 64, 512, 512), dtype='uint16')
    else:
        segmentation = zarr.create(store=path_to_output_labels_zarr, shape=im_da.shape,
                                   chunks=(1, 1, im_da.shape[1], im_da.shape[2]), dtype='uint16')
    for t in time_points:
        with ProgressBar():
            single_bacteria_mask = gpu_process_bacteria(im_da[t], sigma_blur, bright_thresh, sigma_low, sigma_high, disk_size, bacteria_thresh).compute()

        with ProgressBar():
            agg_mask = gpu_process_bacterial_aggregates(im_da[t], sigma_blur_agg, disk_size_agg, bacteria_thresh_agg).compute()

        mask = (single_bacteria_mask + agg_mask) > 0
        del single_bacteria_mask, agg_mask

        # label matrix
        if dask_label:
            path_to_tmp_label_zarr = Path(path_to_output_labels_zarr).parent / 'tmp_labels.zarr'
            mask_da = da.from_array(mask, chunks=(64, 512, 512))
            labels, n_objects = da_label(mask_da)
            with ProgressBar():
                #labels = labels.compute()
                labels.to_zarr(DirectoryStore(path_to_tmp_label_zarr.__str__()))

            labels = zarr.open(DirectoryStore(path_to_tmp_label_zarr.__str__()), 'r')

        else:
            labels = sk_label(mask)

            # filter by object size via regionprops. only works with sk_label for now.
            if maximum_size is not None:
                regions = regionprops(labels)
                for i in tqdm(range(len(regions))):
                    if regions[i].area > maximum_size:
                        labels[labels == i] = 0

        del mask
        segmentation[t] = labels
        del labels

    return


def gpu_process_bacteria(darr, sigma_blur, bright_thresh, sigma_low, sigma_high, disk_size, bacteria_thresh):
    # lazy move to gpu
    arr_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(bacteria_mask, arr_cu, sigma_blur, bright_thresh, sigma_low, sigma_high, disk_size, bacteria_thresh, dtype=np.uint8)

    # actual compute step
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint8)

    return filt_np


def bacteria_mask(arr, sigma_blur, bright_thresh, sigma_low, sigma_high, disk_size, bacteria_thresh):
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    arr = gaussian(arr, sigma=sigma_blur)
    arr[arr > bright_thresh] = 0
    arr = difference_of_gaussians(arr, sigma_low, sigma_high)

    # if disk_size > 0:
    #     arr = white_tophat(arr, disk(disk_size))

    arr = arr > bacteria_thresh

    arr = binary_opening(arr, disk(disk_size))

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def gpu_process_bacterial_aggregates(darr, sigma_blur, disk_size, bacteria_thresh):
    # lazy move to gpu
    arr_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(bacteria_aggregate_mask, arr_cu, sigma_blur, disk_size, bacteria_thresh, dtype=np.uint8)

    # actual compute step
    filt_np = filt.map_blocks(cp.asnumpy, meta=filt, dtype=np.uint8)

    return filt_np


def bacteria_aggregate_mask(arr, sigma_blur, disk_size, bacteria_thresh):
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    arr = gaussian(arr, sigma=sigma_blur)

    arr = arr > bacteria_thresh

    arr = binary_opening(arr, disk(disk_size))

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def initialize_nuclei_dataframe(path_to_segments, dxy, dz, path_to_im=None, voxel_size=(30, 100, 100)):
    seg = zarr.open(DirectoryStore(path_to_segments), 'r')
    if path_to_im is not None:
        im = zarr.open(DirectoryStore(path_to_im), 'r')

    tmp_dfs = []
    for t in range(len(seg)):
        #rprops = regionprops(seg[t])
        rprops = regionprops(seg[t, 0])

        for rprop in tqdm(rprops, desc=f'time {t} of {len(seg) - 1}'):
            locations = np.int16(rprop['centroid'])
            this_dict = dict()
            if path_to_im is not None:
                z0, y0, x0 = locations
                z, y, x = voxel_size
                if check_voxel_boundary(im[t, 0].shape, locations, voxel_size):
                    continue
                else:
                    for channel in range(im.shape[1]):
                        this_voxel = im[t, channel, (z0 - (z - 1) // 2):(z0 + (z - 1) // 2 + 1), (y0 - (y - 1) // 2):(y0 + (y - 1) // 2) + 1,
                                     (x0 - (x - 1) // 2):(x0 + (x - 1) // 2 + 1)]

                        this_dict[f'ch{channel}'] = [this_voxel]

                    this_voxel = seg[t, 0, (z0 - (z - 1) // 2):(z0 + (z - 1) // 2 + 1), (y0 - (y - 1) // 2):(y0 + (y - 1) // 2) + 1,
                                     (x0 - (x - 1) // 2):(x0 + (x - 1) // 2 + 1)]

                    this_dict['segments'] = [this_voxel]

            this_dict['seg_id'] = [rprop['label']]
            this_dict['t'] = [t]
            this_dict['z'] = [locations[0]]
            this_dict['y'] = [locations[1]]
            this_dict['x'] = [locations[2]]

            this_dict['z_um'] = [locations[0] * dz]
            this_dict['y_um'] = [locations[1] * dxy]
            this_dict['x_um'] = [locations[2] * dxy]

            this_df = pd.DataFrame.from_dict(this_dict)
            tmp_dfs.append(this_df)

    df = pd.concat(tmp_dfs, axis=0)

    return df


def check_voxel_boundary(im_shape, location, voxel_size):
    """check whether the spot's voxel lives entirely within the image."""
    z0, y0, x0 = location
    z, y, x = voxel_size
    hit_boundary = ((z0 - (z - 1) // 2) <= 0
                    or (z0 + (z - 1) // 2 + 1) >= im_shape[0]
                    or (y0 - (y - 1) // 2) <= 0
                    or (y0 + (y - 1) // 2 + 1) >= im_shape[1]
                    or (x0 - (x - 1) // 2) <= 0
                    or (x0 + (x - 1) // 2 + 1) >= im_shape[2])

    return hit_boundary


def create_tracked_segments_zarr(df, path_to_segments):
    segments = zarr.open(DirectoryStore(path_to_segments), 'r')
    # segments = da.from_zarr(DirectoryStore(path_to_segments))
    tracked_zarr_path = Path(path_to_segments).parent / 'segmentation.tracked.zarr'
    tracked = zarr.create(store=tracked_zarr_path.__str__(), shape=segments.shape,
                          chunks=segments.chunks, dtype='uint16')
    # tracked = zarr.create(store=tracked_zarr_path.__str__(), shape=segments.shape, dtype='uint16')
    # tracked = da.from_zarr(DirectoryStore(tracked_zarr_path))
    #partial_func = partial(replace_nucleus_id_with_tracked_id, segments=seg_cp, tracked=tracked)
    #sub_df.progress_apply(partial_func, axis=1)

    for i in tqdm(range(segments.shape[0])):
        seg_cp = cp.asarray(segments[i])
        sub_df = df[df.t == i]
        seg_ids = sub_df.seg_id.values
        particles = sub_df.particle.values
        new_tracks = cp.zeros_like(seg_cp)
        for j in range(len(seg_ids)):
            new_tracks[seg_cp == seg_ids[j]] = particles[j]

        tracked[i] = new_tracks.get()

    # tqdm.pandas(desc="my bar!")
    # #df.apply(partial_func, axis=1)
    #df.progress_apply(partial_func, axis=1)

    return


def replace_nucleus_id_with_tracked_id(row, segments, tracked):
    old_label = row.seg_id
    new_label = row.particle
    these_segments = segments[int(row.t)]
    new_segments = np.zeros_like(these_segments)
    new_segments[these_segments == old_label] = new_label
    tracked[int(row.t)] = tracked[int(row.t)] + new_segments

    return


def quantify(df, path_to_particles, path_to_im_zarr, channel, fun, fun_name):
    """apply fun to pixels in im by segment. return a new column to dataframe. maybe move this to a new module later.
    fun must return a scalar"""
    particles = zarr.open(DirectoryStore(path_to_particles), 'r')
    im = zarr.open(DirectoryStore(path_to_im_zarr), 'r')

    df = df.sort_values(by='t')
    fun_vals = np.zeros(len(df))
    counter = 0
    for i in tqdm(range(im.shape[0])):
        particles_cp = cp.asarray(particles[i])
        im_cp = cp.asarray(im[i, channel])
        sub_df = df[df.t == i]
        particle_ids = sub_df.particle.unique()
        for j in range(len(particle_ids)):
            pixels = im_cp[particles_cp == particle_ids[j]]
            fun_out = fun(pixels)
            fun_vals[counter] = fun_out
            counter += 1

    new_col = pd.DataFrame(fun_vals, columns=[fun_name])
    df = df.reset_index(drop=True)
    new_col = new_col.reset_index(drop=True)
    df = pd.concat((df, new_col), axis=1)

    return df


def quantify_mips(df, path_to_mips, fun, fun_name, radius=6):
    """apply fun to pixels in im by segment. return a new column to dataframe. maybe move this to a new module later.
    fun must return a scalar"""
    files = sorted(glob(path_to_mips + '/*green*.tif'))
    prefix = 'mip_green_'
    suffix = '.tif'
    n_files_to_read = len(files)
    for i in range(n_files_to_read):
        im = imread(path_to_mips + '/' + prefix + str(i) + suffix)
        if i == 0:
            im_stack = np.zeros((n_files_to_read,) + im.shape)
            im_filt_stack = np.zeros((n_files_to_read,) + im.shape)
        im_stack[i] = im

    im_stack = cp.asarray(im_stack)
    y_grid, x_grid = cp.indices(im_stack[0].shape)
    df = df.sort_values(by='t')
    fun_vals = np.zeros(len(df))
    locations = df.get(['t', 'y', 'x']).values
    for i in tqdm(range(len(locations))):
        t, y, x = locations[i]
        this_slice = im_stack[int(t)]
        pixels = this_slice[cp.sqrt((y_grid - y) ** 2 + (x_grid - x) ** 2) <= radius]
        fun_vals[i] = fun(pixels)

    new_col = pd.DataFrame(fun_vals, columns=[fun_name])
    df = df.reset_index(drop=True)
    new_col = new_col.reset_index(drop=True)
    df = pd.concat((df, new_col), axis=1)

    return df


def assemble_manual_dfs(path_to_dfs):
    dfs = glob(path_to_dfs + '/cell*.pkl')
    df_list = []
    n_files = len(dfs)
    prefix = 'cell_'
    suffix = '.pkl'
    for i in range(n_files):
        this_df = pd.read_pickle(path_to_dfs + '/' + prefix + str(i) + suffix)
        this_df['particle'] = i
        df_list.append(this_df)

    df = pd.concat(df_list, axis=0)

    return df


def quantify_nuclear_intensities(path_to_zarr, path_to_labels, channel=0):
    im = da.from_zarr(DirectoryStore(path_to_zarr))[:, channel]
    labels = da.from_zarr(DirectoryStore(path_to_labels))[:, 0]

    with ProgressBar():
        index = da.unique(labels).compute()
    index = index[index > 0]
    with ProgressBar():
        intensities = labeled_comprehension(im, labels, index=index, func=da.mean, out_dtype=np.float, default=np.nan).compute()

    return intensities


def initialize_bacteria_dataframe(path_to_segments, path_to_im, dxy, dz, channel=1):
    seg = zarr.open(DirectoryStore(path_to_segments), 'r')
    im = zarr.open(DirectoryStore(path_to_im), 'r')

    tmp_dfs = []
    for t in range(len(seg)):
        rprops = regionprops(seg[t, 0], intensity_image=im[t, channel])

        for rprop in tqdm(rprops, desc=f'time {t} of {len(seg) - 1}'):
            this_dict = dict()

            this_dict['data'] = [rprop['intensity_image']]
            this_dict['seg_id'] = [rprop['label']]
            locations = np.int16(rprop['centroid'])
            this_dict['t'] = [t]
            this_dict['z'] = [locations[0]]
            this_dict['y'] = [locations[1]]
            this_dict['x'] = [locations[2]]

            this_dict['z_um'] = [locations[0] * dz]
            this_dict['y_um'] = [locations[1] * dxy]
            this_dict['x_um'] = [locations[2] * dxy]

            this_df = pd.DataFrame.from_dict(this_dict)
            tmp_dfs.append(this_df)

    df = pd.concat(tmp_dfs, axis=0)

    return df


# class WS:
#     """creating this weird class to use as superpixel generator in elf gasp"""
#     def __init__(self, seeds, method="frangi", params=init_frangi_params(), thresh=1E-6):
#
#         self.method = method
#         self.params = params
#         self.thresh = thresh
#         self.seeds = seeds
#
#     def __call__(self, affinities, foreground_mask=None):
#         boundaries = 1 - affinities[0]
#         #labels = da.map_blocks(watershed, boundaries, self.seeds)
#         seeds = simple_make_seeds_from_boundaries(boundaries)
#         labels = watershed(boundaries, seeds)
#
#         return labels
#
