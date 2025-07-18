from ome_zarr.scale import Scaler
import dask.array as da
import numpy as np
from typing import List
from skimage.transform import downscale_local_mean


class DaskScaler(Scaler):

    def dask_local_mean(self,  base: np.ndarray) -> List[np.ndarray]:
        rv = [base]
        stack_dims = base.ndim - 2
        factors = (*(1,) * stack_dims, *(self.downscale, self.downscale))
        for i in range(self.max_layer):
            rv.append(dask_downscale_local_mean(rv[-1], factors=factors).astype(base.dtype))
        return rv


def dask_downscale_local_mean(data, factors):
    return da.map_blocks(downscale_local_mean, data, factors=factors)

