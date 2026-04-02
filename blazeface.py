from tinygrad import Tensor, nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

class BlazeBlock():
    def __init__(self, c=None, channel_pad=0):
        if c is not None:
            self.stride = c.stride
            self.channel_pad = c.channel_pad
            self.conv0_tiny = c.conv0_tiny
            self.conv1_tiny = c.conv1_tiny
            return
        
        self.channel_pad = channel_pad
    
    def __call__(self, x):
        if self.stride == 2:
            h = x.pad(((0, 0), (0, 0), (0, 2), (0, 2)))
            x = x.max_pool2d(self.stride, self.stride)
        else:
            h = x

        if self.channel_pad > 0:
            x = x.pad(((0, 0), (0, self.channel_pad), (0, 0), (0, 0)))


        h = self.conv0_tiny(h)
        h = self.conv1_tiny(h)
        x += h
        x = x.relu()
        return x

class Seq():
    def __init__(self, size=0):
        super().__init__()
        self.list = [None] * size
    def __len__(self): return len(self.list)
    def __setitem__(self, key, value): self.list[key] = value
    def __getitem__(self, idx): return self.list[idx]
    def __call__(self, x):
        for y in self.list: x = y(x)
        return x

class FinalBlazeBlock():
    def __init__(self, f=None):
        if f is not None:
            self.act = f.act
            self.convs = f.convs
            self.conv0_tiny = f.conv0_tiny
            self.conv1_tiny = f.conv1_tiny
            return
        
        self.conv0_tiny = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0, groups=96, bias=True)
        self.conv1_tiny = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, x):
        x = x.pad(((0, 0), (0, 0), (0, 2), (0, 2)))
        x = self.conv0_tiny(x)
        x = self.conv1_tiny(x)
        x = x.relu()
        return x

class BlazeFace():
    def __init__(self, m=None):
        if m is not None:
            self.backbone_tiny = m.backbone_tiny
            self.conv_tiny = m.conv_tiny
            self.classifier_8 = m.classifier_8
            self.classifier_16 = m.classifier_16
            self.regressor_8 = m.regressor_8
            self.regressor_16 = m.regressor_16
            self.anchors = m.anchors
            self.x_scale = m.x_scale
            self.y_scale = m.y_scale
            self.w_scale = m.w_scale
            self.h_scale = m.h_scale
            self.score_clipping_thresh = m.score_clipping_thresh
            self.min_score_thresh = m.min_score_thresh
            self.min_suppression_threshold = m.min_suppression_threshold
            self.final = m.final
            return
        
        self.conv_tiny = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True)
        self.classifier_8_tiny = nn.Conv2d(in_channels=96, out_channels=2, kernel_size=1, groups=1, bias=True)
        self.classifier_16_tiny = nn.Conv2d(in_channels=96, out_channels=6, kernel_size=1, groups=1, bias=True)
        self.regressor_8_tiny = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, groups=1, bias=True)
        self.regressor_16_tiny = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, groups=1, bias=True)

        self.final = FinalBlazeBlock()
        self.backbone_tiny = Seq(31)
        for i in range(7):
            self.backbone_tiny[i] = BlazeBlock()
            self.backbone_tiny[i].conv0_tiny = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24, bias=True)
            self.backbone_tiny[i].conv1_tiny = nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1
        self.backbone_tiny[7] = BlazeBlock()
        self.backbone_tiny[7].conv0_tiny = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=0, groups=24, bias=True)
        self.backbone_tiny[7].conv1_tiny = nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.backbone_tiny[7].stride = 2
        for i in range(8, 15):
            self.backbone_tiny[i] = BlazeBlock()
            self.backbone_tiny[i].conv0_tiny = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24, bias=True)
            self.backbone_tiny[i].conv1_tiny = nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1
        self.backbone_tiny[15] = BlazeBlock(channel_pad=24)
        self.backbone_tiny[15].conv0_tiny = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=0, groups=24, bias=True)
        self.backbone_tiny[15].conv1_tiny = nn.Conv2d(24, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.backbone_tiny[15].stride = 2
        for i in range(16, 23):
            self.backbone_tiny[i] = BlazeBlock()
            self.backbone_tiny[i].conv0_tiny = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48, bias=True)
            self.backbone_tiny[i].conv1_tiny = nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1
        self.backbone_tiny[23] = BlazeBlock(channel_pad=48)
        self.backbone_tiny[23].conv0_tiny = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=0, groups=48, bias=True)
        self.backbone_tiny[23].conv1_tiny = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.backbone_tiny[23].stride = 2
        for i in range(24, 31):
            self.backbone_tiny[i] = BlazeBlock()
            self.backbone_tiny[i].conv0_tiny = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=True)
            self.backbone_tiny[i].conv1_tiny = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
            self.backbone_tiny[i].stride = 1

        self.anchors = Tensor.empty(896, 4)
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0

        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0
        self.min_score_thresh = 0.85
        self.min_suppression_threshold = 0.3

        load_state_dict(self, safe_load("blazeface.safetensors"))


    def forward(self, x):
        x = x.pad(((0, 0), (0, 0), (1, 2), (1, 2)))
        b = x.shape[0]      # batch size, needed for reshaping later
        x = self.conv_tiny(x)
        x = x.relu()
        x = self.backbone_tiny(x)           # (b, 16, 16, 96)
        h = self.final(x)              # (b, 8, 8, 96)
        c1 = self.classifier_8_tiny(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16_tiny(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c = Tensor.cat(c1, c2, dim=1)
        r1 = self.regressor_8_tiny(x)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)      # (b, 512, 16)
        
        r2 = self.regressor_16_tiny(h)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)      # (b, 384, 16)

        r = Tensor.cat(r1, r2, dim=1)
        return [r, c]

    def __call__(self, img):
        h0, w0 = img.shape[:2]

        scale = min(256/w0, 256/h0)
        new_w, new_h = int(w0*scale), int(h0*scale)
        img = resize(img, [new_w, new_h])
        pad_top = (256 - new_h) // 2
        pad_bottom = 256 - new_h - pad_top
        pad_left = (256 - new_w) // 2
        pad_right = 256 - new_w - pad_left
        x = img.pad(((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), value=0)
        x = x.permute((2, 0, 1))
        x = x.unsqueeze(0)
        x = x / 127.5 - 1.0
        out = self.forward(x)

        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        detections = Tensor.cat(detections[:, :4], detections[:, 16:17], dim=1)
        
        detections = self.postprocess(detections)[0]

        detections = detections * 256
        detections[:, [0, 2]] -= pad_top   # ymin, ymax
        detections[:, [1, 3]] -= pad_left  # xmin, xmax
        detections /= scale
        return detections[:, :5]

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)  # (B, N, 16)
        thresh = self.score_clipping_thresh
        scores = raw_score_tensor.clip(-thresh, thresh).sigmoid().squeeze(-1)
        mask = scores >= self.min_score_thresh  # (B, N)
        scores = scores.unsqueeze(-1)  # (B, N, 1)
        detections = Tensor.cat(detection_boxes, scores, dim=-1)  # (B, N, 17)
        detections *= mask.unsqueeze(-1)
        return detections[0]
    
    def _decode_boxes(self, raw_boxes, anchors):
        boxes = Tensor.zeros_like(raw_boxes).contiguous()
        ax = anchors[:, 0]
        ay = anchors[:, 1]
        aw = anchors[:, 2]
        ah = anchors[:, 3]
        x_center = raw_boxes[..., 0] / self.x_scale * aw + ax
        y_center = raw_boxes[..., 1] / self.y_scale * ah + ay
        w = raw_boxes[..., 2] / self.w_scale * aw
        h = raw_boxes[..., 3] / self.h_scale * ah
        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax
        keypoints = raw_boxes[..., 4:].view(*raw_boxes.shape[:-1], 6, 2)
        kp_x = keypoints[..., 0] / self.x_scale * aw.unsqueeze(0).unsqueeze(-1) + ax.unsqueeze(0).unsqueeze(-1)
        kp_y = keypoints[..., 1] / self.y_scale * ah.unsqueeze(0).unsqueeze(-1) + ay.unsqueeze(0).unsqueeze(-1)
        keypoints_decoded = Tensor.stack((kp_x, kp_y), dim=-1)  # (B, N, 6, 2)
        boxes[..., 4:] = keypoints_decoded.view(*raw_boxes.shape[:-1], -1)
        return boxes

    def postprocess(self, boxes):
        boxes = boxes.unsqueeze(0)
        max_det = 896
        iou_threshold = 0.3
        probs = boxes[:, :, 4] 
        order_all = Tensor.topk(probs, min(max_det, probs.shape[1]))[1]
        batch_idx = Tensor.arange(order_all.shape[0]).reshape(-1, 1)
        boxes = boxes[batch_idx, order_all]
        ious = compute_iou_matrix(boxes[:, :, :4])
        ious = Tensor.triu(ious, diagonal=1)
        high_iou_mask = (ious > iou_threshold)
        no_overlap_mask = high_iou_mask.sum(axis=1) == 0
        conf_mask = (boxes[:, :, 4] >= self.min_score_thresh)
        final_mask = no_overlap_mask * conf_mask
        return boxes * final_mask.unsqueeze(-1)

def compute_iou_matrix(boxes):
  x1s = boxes[:, :, 0]
  y1s = boxes[:, :, 1]
  x2s = boxes[:, :, 2]
  y2s = boxes[:, :, 3]
  areas = (x2s - x1s) * (y2s - y1s)
  x1 = Tensor.maximum(x1s[:, :, None], x1s[:, None, :])
  y1 = Tensor.maximum(y1s[:, :, None], y1s[:, None, :])
  x2 = Tensor.minimum(x2s[:, :, None], x2s[:, None, :])
  y2 = Tensor.minimum(y2s[:, :, None], y2s[:, None, :])
  w = Tensor.maximum(Tensor(0), x2 - x1)
  h = Tensor.maximum(Tensor(0), y2 - y1)
  intersection = w * h
  union = areas[:, :, None] + areas[:, None, :] - intersection
  return intersection / union

def resize(img, new_size):
  img = img.permute(2,0,1)
  img = Tensor.interpolate(img, size=(new_size[1], new_size[0]), mode='linear', align_corners=False)
  img = img.permute(1, 2, 0)
  return img