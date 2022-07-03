import torch

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
], dtype=torch.int32)

import coremltools as ct

encoder = ct.models.MLModel('encoder.mlpackage')

print(encoder)

encoder_state = encoder.predict({
    'text_tokens': text_tokens
})['encoder_state_57']

print(encoder_state, encoder_state.dtype)
