import dask
import dask.array as da
from itertools import product
from skimage import io

def get_tensor_for_row_column(row: int, 
                              col: int,
                              n_channels: int = 3,
                              channel_digits: int = 1,
                              n_fields: int = 4,
                              patch_size: int = 224,
                              x0_patches: list = [220, 640],
                              y0_patches: list = [220, 640],
                              plane: int = 1,
                              shape: tuple = (1080, 1080),
                              dtype: str = "int16",
                              img_dir: str = "./data/test",
                              microscope_pattern: str="sk1fk1fl1") -> da.Array:
    lazy_arrays = [dask.delayed(io.imread)(img_dir / f"r{row:02d}c{col:02d}f{field:02d}p{plane:02d}-ch{channel:0{channel_digits}d}{microscope_pattern}.tiff")
                        for field, channel in product(range(1, n_fields + 1), range(1, n_channels + 1))]
    lazy_arrays = [da.from_delayed(x, shape=shape, dtype=dtype) for x in lazy_arrays]

    lazy_images = da.stack(lazy_arrays, axis=0)
    lazy_images = lazy_images.reshape((n_fields, n_channels) + lazy_images.shape[1:])

    lazy_patches = da.stack([
        lazy_images[field, :, x0_patch:x0_patch + patch_size, y0_patch:y0_patch + patch_size]
        for field, x0_patch, y0_patch in product(range(n_fields), x0_patches, y0_patches)]
        )
    
    return lazy_patches

if __name__ == "__main__":
    well = get_tensor_for_row_column(1, 1)
    print(well)