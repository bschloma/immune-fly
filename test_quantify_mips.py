import pandas as pd
from segmentation import quantify_mips
import numpy as np


df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_14_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/manual_analysis/all_cells_20240710.pkl')
path_to_mips = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_14_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/mips'
fun = np.mean
fun_name = 'mean_dpt'

df = quantify_mips(df, path_to_mips, fun, fun_name)