import torch
from mgqa.transformer import MGQATransformer, Decoder

x = torch.randint(0, 20000, (1, 1024))

model = MGQATransformer(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8,
        attn_kv_heads = 2 # say you want 4 query heads to attend to 1 key / value head
    )
)

result = model(x)
print(result)