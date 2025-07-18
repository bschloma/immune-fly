import pandas as pd
import trackpy as tp
import zarr


df = pd.read_pickle(r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_02_20_dpt-gfp_ca-Gal4_UAS-His-RFP_halocarbon_5x_timeseries/larva_2/nuclei.pkl')

memory = 2
pos_columns = ['z_um', 'y_um', 'x_um']
t_column = 't'
#pred = tp.predict.NearestVelocityPredict()
#df2 = pred.link_df(df, 17, memory=memory,
        # pos_columns=pos_columns,
        # t_column=t_column)

df2 = tp.link_df(df, 17, memory=memory,
        pos_columns=pos_columns,
        t_column=t_column)