from pythae.models import AutoModel
from bioimage_embed import infer
import torch


model = AutoModel.load_from_folder("models/resnet50_vqvae-idr0093")

x = torch.randn(1, 3, 224, 224)

res = infer(model, x)

# Results
print(res["recon_x"].shape)
print(res["quantized_embed"].shape)
print(res["quantized_indices"])