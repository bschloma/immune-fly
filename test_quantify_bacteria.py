from quantify import quantify_bacteria
import pandas as pd


df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_03_20_dpt-gfp_r4-gal4_ecoli-hs-dtom_input-output_pilot_4-6hrs/larva_5/bacteria.pkl')
df_quant = quantify_bacteria(df, quant_col='data')