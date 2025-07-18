from segmentation import filter_bacteria_labels
import pandas as pd
import numpy as np


path_to_labels = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_1/bacteria.segmentation.ome.zarr/0'
path_to_filtered_labels = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_1/bacteria.segmentation.nmax2.zarr'
df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_07_02-NP1029-Gal4_ecoli-hs-dtom_2hr/larva_1/bacteria.pkl')

# filter for single bacteria
method = 'bkg_sub_sum_data'
intens = df.get(method)
single_cell_inten = np.median(intens)
df['n_bacteria'] = np.clip(intens / single_cell_inten, a_min=1, a_max=np.inf).astype('int')

bad_df = df[df.n_bacteria >= 3]
bad_ids = bad_df.seg_id.unique()

filter_bacteria_labels(path_to_labels, path_to_filtered_labels, bad_ids)