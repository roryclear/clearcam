from detection.yolov9 import YOLOv9, SIZES, safe_load, load_state_dict, Sequential, Silence, Conv, RepNCSPELAN4, AConv,\
ADown, CBLinear, CBFuse, SPPELAN, Upsample, Concat, DDetect, postprocess, fetch, rescale_bounding_boxes, draw_bounding_boxes_and_save
import cv2
from tinygrad import Tensor
from tinygrad.dtype import dtypes
import numpy as np
from pathlib import Path

def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad: im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

if __name__ == "__main__":
  for size in ["t", "s", "m", "c", "e"]:
    weights = f'./yolov9-{size}-tiny.pkl'
    imgsz = (640,640)
    model = YOLOv9(*SIZES[size]) if size in SIZES else YOLOv9()
    state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/yolov9/resolve/main/yolov9-{size}.safetensors'))
    load_state_dict(model, state_dict)
    source = "test/videos/MOT16-03.mp4"
    cap = cv2.VideoCapture(source)
    ret, im0 = cap.read()
    success, buffer = cv2.imencode(".jpg", im0)
    if not success: raise RuntimeError("Failed to encode frame")
    source = buffer

    cap.release()
    im = letterbox(im0, new_shape=(1280, 1280), stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = Tensor(im).cast(dtypes.float32)
    im /= 255
    if len(im.shape) == 3: im = im[None]
    
    pred = model(im)
    pred = pred.numpy()[0]
    pred = pred[pred[:, 4] >= 0.25]
    class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
    pred = rescale_bounding_boxes(pred, from_size=(im.shape[2:][::-1]), to_size=im0.shape[:2][::-1])
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    draw_bounding_boxes_and_save(source, f"outputs/out_{size}.jpg", pred, class_labels)

