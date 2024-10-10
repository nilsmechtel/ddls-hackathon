import dask
import dask.array as da
from itertools import product
from pathlib import Path
from pythae.models import AutoModel
from skimage import io
import torch

from bioimage_embed import infer

torch.set_default_device("cuda")

DATA_DIR = Path("./data")
IMG_DIR = DATA_DIR / "test"
MODEL_DIR = Path("./models/resnet50_vqvae-idr0093")

model = AutoModel.load_from_folder(MODEL_DIR)

row = 4
col = 1
field = 1
plane = 1
channel = 1

pattern = f"r{row:02d}c{col:02d}f{field:02d}p{plane:02d}-ch{channel:01d}sk1fk1fl1.tiff"
print(pattern)

image = io.imread(IMG_DIR / pattern)

n_rows = 16
n_cols = 24
n_fields = 4
plane = 1
n_channels = 3

lazy_arrays = [dask.delayed(io.imread)(IMG_DIR / f"r{row:02d}c{col:02d}f{field:02d}p{plane:02d}-ch{channel:01d}sk1fk1fl1.tiff")
                    for field, channel in product(range(1, n_fields + 1), range(1, n_channels + 1))]
lazy_arrays = [da.from_delayed(x, shape=image.shape, dtype=image.dtype)
               for x in lazy_arrays]

lazy_images = da.stack(lazy_arrays, axis=0)
lazy_images = lazy_images.reshape((n_fields, n_channels) + lazy_images.shape[1:])


patch_size = 224
x0_patches = [220, 640]
y0_patches = [220, 640]

lazy_patches = da.stack([
    lazy_images[field, :, x0_patch:x0_patch + patch_size, y0_patch:y0_patch + patch_size]
    for field, x0_patch, y0_patch in product(range(n_fields), x0_patches, y0_patches)]
    )

res = infer(model, torch.Tensor(lazy_patches.compute()))

# Results
print(res["recon_x"])
print(res["quantized_embed"])
print(res["quantized_indices"])