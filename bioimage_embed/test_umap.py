import numpy as np
import torch
from calculate_umap import calculate_umap
from plot_umap import plot_umap


# Generate mock data
def generate_mock_data(num_samples: int = 300, num_features: int = 12544):
    # Create two clusters of data with some Gaussian noise
    control_samples = np.random.normal(loc=0.5, scale=0.01, size=(num_samples // 2, num_features))
    disease_samples = np.random.normal(loc=-0.5, scale=0.01, size=(num_samples // 2, num_features))

    # Stack the control and disease samples together
    quantized_embed = np.vstack([control_samples, disease_samples])

    # Create labels
    labels = ['Control'] * (num_samples // 2) + ['Disease'] * (num_samples // 2)

    return quantized_embed, labels

# Test function to check if calculate_umap and plot_umap work
def test_umap_functions():
    # Generate the mock data
    quantized_embed, labels = generate_mock_data()

    # Convert the embeddings to a tensor (if needed, otherwise keep as numpy array)
    quantized_embed_tensor = torch.tensor(quantized_embed, dtype=torch.float32)

    # Calculate UMAP
    embedding_2d = calculate_umap(quantized_embed_tensor)

    # Plot UMAP
    output_path = "test_umap_plot.png"
    plot_umap(embedding_2d, labels, output_path)

    # Check if the plot was saved successfully
    print("Test completed.")

# Run the test
test_umap_functions()
