import pandas as pd
from quantify import quantify_nuclei

df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_06_ca-Gal4_UAS-His-RFP_mNG-EcR-B1/larva_2/nuclei_3.pkl')
df = quantify_nuclei(df, quant_col='ch0')

