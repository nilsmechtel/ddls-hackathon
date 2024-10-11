from pathlib import Path
import zarr

from bioimage_embed.calculate_umap import calculate_umap
from bioimage_embed.plot_umap import plot_umap


RESULTS_DIR = Path("./results")
DATA_DIR = Path("./data/test")
pca_components = 48

# Load the features
features = zarr.open(RESULTS_DIR / "features_sanity_check_2.zarr", mode="r")


# Calculate UMAP
concatenated_features = features[:].reshape((-1, 12544))
embedding_2d = calculate_umap(concatenated_features, pca_components=pca_components)
# labels = ['Control'] * 16 + ['Disease'] * 16
labels = ['Batch 1'] * 16 + ['Batch 2'] * 16 + ['Batch 3'] * 16

# Plot the UMAP visualization
plot_umap(embedding_2d, labels, f"morphology_same_well_umap_plot_ncomp_{pca_components}.png")