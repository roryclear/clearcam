from tinygrad import Tensor, nn
Tensor.manual_seed(42)
import numpy as np

class YOLOv9():
  def __init__(self): self.model =  nn.Conv2d(160, 64, 1, (1,1), (0,0), (1,1), 1, True)
  def __call__(self, frame): return self.model(frame).sum()

if __name__ == "__main__":
  jit_cache = {}
  img = Tensor.rand((1, 160, 68, 120))
  conv = nn.Conv2d(160, 64, 1, (1,1), (0,0), (1,1), 1, True)
  for _ in range(3): ret = conv(img).sum().numpy()

  expected = 17993.432
  np.testing.assert_allclose(ret, expected, rtol=1e-4)

  # sanity test, BEAM is flakey

