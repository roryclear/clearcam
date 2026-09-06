from llm.qwen3vl import Qwen3VL
from utils.helpers import jit_infer
import cv2
if __name__ == "__main__":
  qwen = Qwen3VL(size=f"2B", res=(640, 640)) # h, w. they need to be multiples of 32
  print("prewarming Qwen")
  qwen.prewarm()
  print("DONE")
  text = qwen.generate(prompt="what make, model, and color is this? one sentence only", image=cv2.cvtColor(cv2.imread("test/clip_images/f40.jpg"), cv2.COLOR_BGR2RGB), reset=True)
  print("text =",text)
  assert text == "This is a red Ferrari F40, a classic sports car from the 1980s."