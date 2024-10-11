import dask
import dask.array as da
from itertools import product
from pathlib import Path
from pythae.models import AutoModel
from skimage import io
import torch

from bioimage_embed import infer
from bioimage_embed.io import get_tensor_for_row_column
from bioimage_embed.normalise import percentile_normalize


DATA_DIR = Path("./data")
IMG_DIR = DATA_DIR / "test"
MODEL_DIR = Path("./models/resnet50_vqvae-idr0093")
RESULTS_DIR = Path("./results")

# Load model
model = AutoModel.load_from_folder(MODEL_DIR)

# Make dask array of images
n_rows = 16
n_cols = 24
all_wells = []

for row, col in product(range(1, n_rows + 1), range(1, n_cols + 1)):
    this_well = get_tensor_for_row_column(row, col)  # shape: (16, 3, 224, 224)
    all_wells.append(this_well)

all_wells = da.stack(all_wells)
all_wells = all_wells.rechunk((1, 1, 3, 224, 224))

# Run inference on each well
def run_for_patch(patch):
    assert patch.shape == (1, 1, 3, 224, 224), f"There should be a single well in the input, it was {patch.shape}"
    with torch.no_grad():
        this_patch = percentile_normalize(torch.Tensor(patch[0, 0]))
        out = infer(model, this_patch[None, ...])["quantized_embed"].squeeze(-1).squeeze(-1).numpy()
    return out[None, ...]

# map all chunks to the function
out = da.map_blocks(run_for_patch, all_wells, dtype="uint16", drop_axis=[-3, -2, -1], new_axis=2, chunks=(1, 1, 12544))

# Compute and save the results to a Zarr file
out.to_zarr(RESULTS_DIR / "features.zarr", overwrite=True)