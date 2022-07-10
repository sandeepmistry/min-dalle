import torch

from min_dalle.models import DalleBartDecoder

decoder = DalleBartDecoder(
    sample_token_count = 256,
    image_token_count = 256,
    image_vocab_count = 16384,
    attention_head_count = 16,
    embed_count = 1024,
    glu_embed_count = 2730,
    layer_count = 12,
    start_token = 16384
)

params = torch.load('pretrained/dalle_bart_mini/decoder.pt')

decoder.load_state_dict(params, strict=False)
del params

text_tokens = torch.tensor([
    [   0,    2,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1
    ],
    [   0, 8925,  742,    2,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
        1,    1,    1,    1
    ]
], dtype=torch.int64)

encoder_state = torch.zeros([2, 64, 1024], dtype=torch.float32)

torch.manual_seed(1)

image_tokens = decoder.forward(
    text_tokens, 
    encoder_state
)

print(image_tokens)

import coremltools as ct
import numpy as np

decoder.eval()

scripted_model = torch.jit.script(decoder, example_inputs=[text_tokens, encoder_state])

model = ct.convert(
    scripted_model,
    inputs=[
        ct.TensorType(name="text_tokens", shape=text_tokens.shape, dtype=np.int64),
        ct.TensorType(name="encoder_state", shape=encoder_state.shape, dtype=np.float32)
    ],
    outputs=[
        ct.TensorType(name="image_tokens", dtype=np.int64)
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32
)

model.save('decoder.mlpackage')

