from tinygrad import Tensor, nn
Tensor.manual_seed(42)
import numpy as np

if __name__ == "__main__":
  jit_cache = {}
  img = Tensor.rand((1, 160, 32, 1))
  conv = nn.Conv2d(160, 64, 1, (1,1), (0,0), (1,1), 1, False)
  ret = conv(img).sum().numpy()
  expected = -8.297832
  np.testing.assert_allclose(ret, expected, rtol=1e-4)

  # sanity test, BEAM is flakey

