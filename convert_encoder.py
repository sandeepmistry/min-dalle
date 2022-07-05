# attention_head_count 16
# embed_count 1024
# glu_embed_count 2730
# text_token_count 64
# text_vocab_count 50264
# layer_count 12

import torch

from min_dalle.models import DalleBartEncoder

encoder = DalleBartEncoder(
    attention_head_count = 16,
    embed_count = 1024,
    glu_embed_count = 2730,
    text_token_count = 64,
    text_vocab_count = 50264,
    layer_count = 12
)

params = torch.load('pretrained/dalle_bart_mini/encoder.pt')

encoder.load_state_dict(params, strict=False)
del params

text_tokens = torch.tensor([
    [   0,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1
    ],
    [
        0, 8925,  742,    2,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1
    ]
], dtype=torch.int64)

import coremltools as ct
import numpy as np

encoder.eval()

scripted_model = torch.jit.script(encoder, example_inputs=[text_tokens])

# traced_model = torch.jit.trace(encoder, text_tokens)
# out = traced_model(text_tokens)

model = ct.convert(
    scripted_model, # traced_model
    inputs=[
        ct.TensorType(name="text_tokens", shape=text_tokens.shape, dtype=np.int64),
    ],
    outputs=[
        ct.TensorType(name="encoder_state", dtype=np.float32),
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32
)

model.save('encoder.mlpackage')
