import torch

from min_dalle.models import VQGanDetokenizer

detokenizer = VQGanDetokenizer()
params = torch.load('pretrained/vqgan/detoker.pt')
detokenizer.load_state_dict(params)
del params

image_tokens = torch.zeros((1, 256), dtype=torch.int64)

detokenizer.forward(image_tokens)

import coremltools as ct
import numpy as np

detokenizer.eval()

traced_model = torch.jit.trace(detokenizer, image_tokens)
out = traced_model(image_tokens)

model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="image_tokens", shape=image_tokens.shape, dtype=np.int64)
    ],
    outputs=[
        ct.TensorType(name="images", dtype=np.float32),
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32
)

model.save('detokenizer.mlpackage')
