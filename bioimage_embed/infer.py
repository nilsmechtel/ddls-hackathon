
def infer(model, x):
    # Encoder
    encoder_output = model.encoder(x)
    embeddings = encoder_output.embedding
    if len(embeddings.shape) == 2:
        embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
        reshape_for_decoding = True
    embeddings = embeddings.permute(0, 2, 3, 1)

    # Quantizer
    uses_ddp = False
    quantizer_output = model.quantizer(embeddings, uses_ddp=uses_ddp)
    quantized_embed = quantizer_output.quantized_vector
    quantized_indices = quantizer_output.quantized_indices

    # Decoder
    reshape_for_decoding = False
    if reshape_for_decoding:
        quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)
    recon_x = model.decoder(quantized_embed).reconstruction

    return {
        "recon_x": recon_x,
        "quantized_embed": quantized_embed,
        "quantized_indices": quantized_indices,
    }