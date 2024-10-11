import torch
import numpy as np
from sklearn.decomposition import PCA
import umap


def calculate_umap(quantized_embed: np.array, pca_components: int=50):
    if isinstance(quantized_embed, torch.Tensor):
        # Convert to NumPy array
        quantized_embed = quantized_embed.detach().cpu().numpy()

    # Check the shape before reshaping
    print(f"Embeddings shape: {quantized_embed.shape}")

    # Check the dimensions of the input tensor
    if quantized_embed.ndim == 4:
        quantized_embed = quantized_embed.squeeze(-1).squeeze(-1)  # Remove the last two singleton dimensions
    assert quantized_embed.ndim == 2, f"Expected 2D tensor, got {quantized_embed.ndim}D"
    assert quantized_embed.shape[1] == 12544, f"Expected 12544 features, got {quantized_embed.shape[1]}"
    print(f"Embeddings shape after reshaping: {quantized_embed.shape}")
    
    # Perform PCA for dimensionality reduction
    if pca_components > min(quantized_embed.shape):
        print(f"Adjusting PCA components from {pca_components} to {min(quantized_embed.shape)}")
        pca_components = min(quantized_embed.shape)

    print(f"Performing PCA to reduce from {quantized_embed.shape[1]} to {pca_components} components...")
    pca = PCA(n_components=pca_components, svd_solver="auto")  # Use svd_solver="auto"
    pca_result = pca.fit_transform(quantized_embed)
    print(f"Shape after PCA: {pca_result.shape}")
    
    # Fit UMAP on PCA-reduced embeddings
    print("Performing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(pca_result)
    print(f"Shape after UMAP: {embedding_2d.shape}")

    return embedding_2d

if __name__ == "__main__":
    # Example usage
    quantized_embed = torch.randn(16, 12544, 1, 1)  # Random 4D tensor
    embedding_2d = calculate_umap(quantized_embed)
    print(embedding_2d)