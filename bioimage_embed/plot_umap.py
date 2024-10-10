import torch
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_embeddings(quantized_embed: torch.Tensor, labels: list, outpath: str, pca_components: int=50):
    # Ensure the embeddings are detached from the computation graph and moved to CPU
    if quantized_embed.dim() == 4:
        quantized_embed = quantized_embed.squeeze(-1).squeeze(-1)  # Remove the last two singleton dimensions
    quantized_embed = quantized_embed.detach().cpu().numpy()    # Convert to NumPy array with dimensions [B, embedding_dim]

    # Check the shape after reshaping
    print(f"Embeddings shape: {quantized_embed.shape}")
    
    # Perform PCA for dimensionality reduction
    pca_components = min(pca_components, min(quantized_embed.shape))
    print(f"Performing PCA to reduce from {quantized_embed.shape[1]} to {pca_components} components...")
    pca = PCA(n_components=pca_components, svd_solver="auto")  # Use svd_solver="auto"
    pca_result = pca.fit_transform(quantized_embed)
    print(f"Shape after PCA: {pca_result.shape}")
    
    # Fit UMAP on PCA-reduced embeddings
    print("Performing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(pca_result)
    print(f"Shape after UMAP: {embedding_2d.shape}")

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
    quantized_embed = torch.randn(16, 12544, 1, 1)  # Shape: [B, 12544, 1, 1]
    labels = ["Control"] * 8 + ["Disease"] * 8  # Create test labels for the 16 samples
    visualize_embeddings(quantized_embed, labels, "test_umap_plot.png")  # Pass labels to the function
