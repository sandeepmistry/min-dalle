import torch
import numpy as np

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

decoder = ct.models.MLModel('decoder.mlpackage')
print(decoder)

detokenizer = ct.models.MLModel('detokenizer.mlpackage')
print(detokenizer)

encoder_state = encoder.predict({
    'text_tokens': text_tokens
})['encoder_state']

print(encoder_state, encoder_state.shape, encoder_state.dtype)

torch.manual_seed(1984)
np.random.seed(1984)


prev_tokens = torch.tensor([16384], dtype=torch.int32)
attention_state = torch.zeros((12, 4, 256, 1024), dtype=torch.float32)
image_tokens = torch.zeros((1, 256), dtype=torch.int32)
for i in range(256):
    decoder_result = decoder.predict({
        'text_tokens': text_tokens,
        'encoder_state': encoder_state,
        'attention_state': attention_state,
        'prev_tokens': prev_tokens,
        'token_index': torch.tensor([i], dtype=torch.int32)
    })

    image_token = decoder_result['image_tokens']
    attention_state = decoder_result['attention_state_']

    print(i, image_token, image_token.shape, image_token.dtype)

    image_tokens[:, i] = image_token[0]
    prev_tokens = image_token

print(image_tokens, image_tokens.shape, image_tokens.dtype)

images = detokenizer.predict({
    'image_tokens': image_tokens
})['images']

print(images[0].shape, images[0].dtype)

from PIL import Image
import numpy as np

image = Image.fromarray(images[0].astype(np.uint8))

image.save('generated_ct.png')

