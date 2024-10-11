# import dask
import dask.array as da
from itertools import product
import pandas as pd
from pathlib import Path
from pythae.models import AutoModel
# from skimage import io
import torch
import time

# torch.set_default_device("cuda")

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
metadata_df = pd.read_csv(IMG_DIR / "PLATEMAP_conf3_labels.csv")
index_map = {(row, col): n for n, (row, col) in enumerate(product(range(1, n_rows + 1), range(1, n_cols + 1)))}
metadata_df["zarr_index"] = metadata_df.apply(lambda x: index_map[(x["row"], x["col"])], axis=1)

selected_wells = metadata_df.query("labels != 'samples'")[["row", "col"]].reset_index(drop=True)
selected_wells.to_csv(RESULTS_DIR / "selected_wells.csv")

all_wells = []
for row, col in selected_wells.values:
    this_well = get_tensor_for_row_column(row, col)  # shape: (16, 3, 224, 224)
    all_wells.append(this_well)

all_wells = da.stack(all_wells)
all_wells = all_wells.rechunk((1, 1, 3, 224, 224))

# Run inference on each well
def run_for_patch(patch):
    assert patch.shape == (1, 1, 3, 224, 224), f"There should be a single well in the input, it was {patch.shape}"
    with torch.no_grad():
        this_patch = percentile_normalize(torch.Tensor(patch[0, 0].astype("int16")))
        out = infer(model, this_patch[None, ...])["quantized_embed"].squeeze(-1).squeeze(-1).numpy()
    return out[None, ...]

start_time = time.perf_counter()
# map all chunks to the function
out = da.map_blocks(run_for_patch, all_wells, dtype="float32", drop_axis=[-3, -2, -1], new_axis=2, chunks=(1, 1, 12544))

# Compute and save the results to a Zarr file
out.to_zarr(RESULTS_DIR / "features.zarr", overwrite=True)
end_time = time.perf_counter()

print(f"Time taken: {end_time - start_time:.2f} seconds")