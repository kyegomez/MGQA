[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MGQA
The open source implementation of the multi grouped query attention by the paper "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"


[Paper Link](https://arxiv.org/abs/2305.13245)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install mgqa`

# Usage
```python
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
```


# License
MIT

# Citations
```biblitex
@misc{2305.13245,
Author = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebrón and Sumit Sanghai},
Title = {GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
Year = {2023},
Eprint = {arXiv:2305.13245},
}
```