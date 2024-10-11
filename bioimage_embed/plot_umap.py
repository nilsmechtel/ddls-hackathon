import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_umap(embedding_2d: np.array, labels: list, outpath: str):
    # Get unique labels and create a color map
    unique_labels = list(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Map labels to numeric values
    numeric_labels = [label_map[label] for label in labels]  # Convert labels to numeric values

    # Generate a color palette with as many colors as there are unique labels
    colors = cm.get_cmap('Spectral', len(unique_labels))  # Get a colormap with a number of distinct colors
    color_labels = unique_labels

    # Plot the embeddings using matplotlib
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=numeric_labels, cmap='Spectral', s=50)  # Use numeric labels for coloring
    plt.title("UMAP Projection of PCA-reduced Quantized Embeddings")

    # Create a legend for the labels, positioned outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                           markerfacecolor=colors(i), markersize=10) for label, i in zip(color_labels, range(len(color_labels)))]
    plt.legend(handles=handles, title='Labels', loc='upper left', bbox_to_anchor=(1, 1))  # Position the legend outside

    # Save the plot to the specified output path
    plt.savefig(outpath, bbox_inches='tight')  # Use bbox_inches='tight' to fit the plot nicely
    print(f"UMAP plot saved to {outpath}")

if __name__ == "__main__":
    # Example usage
    embedding_2d = np.random.rand(16, 2)  # Random 2D embeddings for 16 samples
    labels = ["Control"] * 8 + ["Disease"] * 8  # Create test labels for the 16 samples
    plot_umap(embedding_2d, labels, "test_umap_plot.png")  # Plot the UMAP visualization and save it to a file
