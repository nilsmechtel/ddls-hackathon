import torch

def percentile_normalize(tensor, lower_percentile=2, upper_percentile=98):
    tensor = tensor.to(torch.float32)  # Convert to float32 for calculations

    # Flatten the tensor to compute percentiles
    flattened_tensor = tensor.flatten()

    # Calculate the lower and upper percentiles
    lower_bound = torch.quantile(flattened_tensor, lower_percentile / 100.0)
    upper_bound = torch.quantile(flattened_tensor, upper_percentile / 100.0)

    # Normalize the tensor
    normalized_tensor = (tensor - lower_bound) / (upper_bound - lower_bound)

    # Clip the values to be between 0 and 1
    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)

    return normalized_tensor

if __name__ == "__main__":
    # Example usage
    tensor_uint16 = torch.randint(0, 65536, (3, 4), dtype=torch.uint16)  # Random uint16 tensor
    normalized_tensor = percentile_normalize(tensor_uint16)

    print("Original Tensor:")
    print(tensor_uint16)
    print("\nNormalized Tensor:")
    print(normalized_tensor)