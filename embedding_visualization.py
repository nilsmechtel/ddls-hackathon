from pathlib import Path
import pandas as pd
from itertools import product
import zarr

from bioimage_embed.calculate_umap import calculate_umap
from bioimage_embed.plot_umap import plot_umap


RESULTS_DIR = Path("./results")
DATA_DIR = Path("./data/test")

# Load the features
features = zarr.open(RESULTS_DIR / "features.zarr", mode="r")

n_rows = 16
n_cols = 24

index_map = {(row, col): n for n, (row, col) in enumerate(product(range(1, n_rows + 1), range(1, n_cols + 1)))}

metadata_df = pd.read_csv(DATA_DIR / "PLATEMAP_conf3_labels.csv")
metadata_df["zarr_index"] = metadata_df.apply(lambda x: index_map[(x["row"], x["col"])], axis=1)

# Calculate UMAP
concatenated_features = features[metadata_df["zarr_index"].values].reshape((-1, 12544))
embedding_2d = calculate_umap(concatenated_features, pca_components=50)
labels = []
for val in metadata_df["labels"].values:
    labels.extend([val, ] * 16)
labels

# Plot the UMAP visualization
plot_umap(embedding_2d, labels, "umap_plot.png")
