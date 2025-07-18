from PIL import Image
from aicsimageio import AICSImage
from glob import glob
from pathlib import Path
from tqdm import tqdm


movie_dir = r'/media/brandon/Data2/Brandon/fly_immune/Lightsheet_Z1/2024_05_30-dpt-gfp_r4-gal4_ecoli-hs-dtom_4hrs_flow_field/larva_1/movie_9_fps33pt4_dual'
out_dir = movie_dir + '/tifs'
Path(out_dir).mkdir(exist_ok=True)
files = sorted(glob(movie_dir + '/im*.czi'))
n_time_points = int(len(files) / 2)
counter = 0
for i in tqdm(range(n_time_points)):
    if counter == 0:
        left_image = AICSImage(movie_dir + f'/im.czi').get_image_data().squeeze()
    else:
        left_image = AICSImage(movie_dir + f'/im({counter}).czi').get_image_data().squeeze()
    counter += 1

    right_image = AICSImage(movie_dir + f'/im({counter}).czi').get_image_data().squeeze()
    counter += 1
    mean_image = (0.5 * (left_image.astype('float') + right_image.astype('float'))).astype('uint16')

    Image.fromarray(mean_image).save(out_dir + f'/im_{i}.tif')
