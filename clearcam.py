from tinygrad.tensor import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import fetch
from detection.yolov9 import safe_load, load_state_dict, YOLOv9
from detection.rfdetr import RFDETR, detr_to_yolo
import numpy as np
from pathlib import Path
import cv2
from collections import defaultdict, deque
import time, sys
from tinygrad.nn.state import safe_load, load_state_dict
import json
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import shutil
from datetime import datetime, time as time_obj
import uuid
import urllib
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import struct
from urllib.parse import unquote, quote
import zlib
from utils.db import db
import multiprocessing
import re
import base64
from utils.helpers import send_notif, find_ffmpeg, export_clip, upload_file, encrypt_file, export_and_upload

# RTSP URL
# Video capture thread
import subprocess
import threading
import time
import numpy as np
import cv2
from datetime import datetime
import os
import threading
from utils.helpers import BASE_DIR

(BASE_DIR / "cameras").mkdir(parents=True, exist_ok=True)
models = {1: "t", 2: "s", 3: "m", 4: "c", 5: "e", 6: "nano", 7: "small", 8:"medium", 9:"large"}

class RollingClassCounter:
  def __init__(self, window_seconds=None, max=None, classes=None, sched=[[0,86399],True,True,True,True,True,True,True],cam_name=None, desc=None, threshold=0.28):
    self.window = window_seconds
    self.data = defaultdict(deque)
    self.max = max
    self.classes = classes
    self.last_det = 0
    self.sched = sched
    self.cam_name = cam_name
    self.is_on = True
    self.is_notif = True
    self.zone = True
    self.reset = False
    self.new = True
    self.desc = desc
    self.desc_emb = None
    self.threshold = threshold

  def add(self, class_id):
    if self.classes is not None and class_id not in self.classes: return
    now = time.time()
    self.data[class_id].append(now)
    self.cleanup(class_id, now)

  def cleanup(self, class_id, now):
    q = self.data[class_id]
    window = self.window if self.window else (60 if self.is_notif else 1)
    while window and q and now - q[0] > window:
      q.popleft()

  def reset_counts(self):
    for class_id, _ in self.data.items():
       self.data[class_id] = deque() # todo, use in reset endpoint?
    self.reset = True

  def get_counts(self):
    window = self.window if self.window else (60 if self.is_notif else 1)
    max_reached = False
    now = time.time()
    counts = {}
    for class_id, q in self.data.items():
      while window and q and now - q[0] > window:
        q.popleft()
      if q:
        counts[class_id] = len(q)
        if self.max and len(q) >= self.max: max_reached = True
    return counts, max_reached
  
  def is_active(self, offset=0):
    if not alerts_on: return False
    if not getattr(self, "is_on", False): return False
    if not self.sched: return True
    now = time.localtime()
    time_of_day = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if not self.sched[time.localtime().tm_wday + 1]: return False
    window = self.window if self.window else (60 if self.is_notif else 1)
    return time_of_day < self.sched[0][1] and time_of_day > ((self.sched[0][0] - window) + offset)

def write_png(filename, array):
    array = array[..., ::-1]  # BGR to RGB
    height, width, _ = array.shape
    png_signature = b"\x89PNG\r\n\x1a\n"
    def chunk(chunk_type, data):
        return (struct.pack("!I", len(data)) +
                chunk_type +
                data +
                struct.pack("!I", zlib.crc32(chunk_type + data) & 0xffffffff))
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_data = b"".join(b"\x00" + array[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw_data, 9)
    png_bytes = (
        png_signature +
        chunk(b"IHDR", ihdr) +
        chunk(b"IDAT", compressed) +
        chunk(b"IEND", b"")
    )
    with open(filename, "wb") as f:
        f.write(png_bytes)

import numpy as np

def draw_rectangle_numpy(img, pt1, pt2, color, thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    if thickness == -1:  # fill
        img[y1:y2+1, x1:x2+1] = color
    else:
        img[y1:y1+thickness, x1:x2+1] = color
        img[y2-thickness+1:y2+1, x1:x2+1] = color
        img[y1:y2+1, x1:x1+thickness] = color
        img[y1:y2+1, x2-thickness+1:x2+1] = color
    return img


def is_vod(cam_name): return Path("data/cameras", cam_name, "streams", "video").is_dir()

def _get_stream_resolution(src):
  ffmpeg_path = find_ffmpeg()
  command = [ffmpeg_path, "-i", src]
  try:
    result = subprocess.run(
        command,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        timeout=10
    )
    output = result.stderr
    match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", output)
    if match:
      width, height = map(int, match.groups())
      return width, height
    return 1920, 1080
  except Exception as e:
    return 1920, 1080

class VideoCapture:
  def __init__(self, src, cam_name="camera", vod=False):
    self.vod = vod
    self.output_dir_det = BASE_DIR / "cameras" / f'{cam_name}_det' / "streams"
    self.output_dir_raw = BASE_DIR / "cameras" / f'{cam_name}' / "streams"
    # objects in scene count
    self.counter = RollingClassCounter(cam_name=cam_name, window_seconds=float('inf'))
    self.cam_name = cam_name
    self.object_set = set()
    self.object_set_zone = set()

    self.src = src
    self.max_frame_rate = 10 # for vod only
    self.width, self.height = _get_stream_resolution(src)
    self.proc = None
    self.hls_proc = None
    self.running = True

    self.raw_frame = None
    self.annotated_frame = None
    self.last_preds = []
    self.last_frame = None
    self.frame_diff = None

    self.settings = None
    
    self.alert_counters = database.run_get("alerts",self.cam_name)
    if not self.alert_counters:
      self.alert_counters = dict()
      id, alert_counter = str(uuid.uuid4()), RollingClassCounter(window_seconds=None, max=1, classes={0,1,2,3,5,7},cam_name=cam_name)
      self.alert_counters[id] = alert_counter
      database.run_put("alerts", self.cam_name, alert_counter, id=id)

    self.lock = threading.Lock()

    if not self.vod or not self.output_dir_raw.exists():
      self._open_ffmpeg()
      threading.Thread(target=self.capture_loop, daemon=True).start()

  def _get_new_stream_dir(self):
      timestamp = "video" if self.vod else datetime.now().strftime("%Y-%m-%d")
      stream_dir_raw = self.output_dir_raw / timestamp
      stream_dir = self.output_dir_det / timestamp
      stream_dir.mkdir(parents=True, exist_ok=True)
      stream_dir_raw.mkdir(parents=True, exist_ok=True)
      return stream_dir_raw

  def _safe_kill_process(self, proc):
    if proc:
      try:
        proc.terminate()
        proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
      except Exception:
        pass

  def _open_ffmpeg(self):
    path = self._get_new_stream_dir()
       
    self._safe_kill_process(self.proc)
    self._safe_kill_process(self.hls_proc)

    ffmpeg_path = find_ffmpeg()
    
    is_rtsp = self.src.startswith("rtsp")
    if self.vod:
      command = [
        ffmpeg_path,
        "-i", self.src,
        "-c:v", "copy",
        "-an",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "0",
        "-hls_flags", "independent_segments",
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(path / "seg_%06d.m4s"),
        str(path / "stream.m3u8"),
      ]
      self.hls_proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      self.proc = None
        
    else:  # Live streams
      # Original live stream pipeline
      command = [
          ffmpeg_path,
          *(["-rtsp_transport", "tcp"] if is_rtsp else []),
                  "-headers", "Referer: https://www,earthcam.com\r\n",
        "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
          "-fflags", "+genpts",
          "-avoid_negative_ts", "make_zero",
          "-i", self.src,
          "-c", "copy",
          "-an",
          "-f", "hls",
          "-hls_time", "2",
          "-hls_list_size", "0",
          "-hls_playlist_type", "event",
          "-hls_flags", "append_list+independent_segments+temp_file",
          "-hls_segment_filename", str(path / "stream_%06d.ts"),
          str(path / "stream.m3u8")
      ]
      self.hls_proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      self.start_time = time.time()
      time.sleep(15)
      
      command = [
          ffmpeg_path,
          "-live_start_index", "-1",
          "-i", str(path / "stream.m3u8"),
          "-loglevel", "quiet",
          "-reconnect", "1",
          "-reconnect_streamed", "1",
          "-reconnect_delay_max", "2",
          "-an",
          "-f", "rawvideo",
          "-pix_fmt", "bgr24",
          "-vf", f"scale={self.width}:{self.height}",
          "-timeout", "5000000",
          "-rw_timeout", "15000000",
          "-vsync", "2",
          "-fflags", "+discardcorrupt+fastseek+flush_packets+nobuffer",
          "-avioflags", "direct",
          "-flags", "low_delay",
          "-max_delay", "100000",
          "-threads", "1",
          "-"
      ]
      self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

  def save_object(self, p, ts=0):
    p = np.array([p.tlwh[0],p.tlwh[1],(p.tlwh[0]+p.tlwh[2]),(p.tlwh[1]+p.tlwh[3]),p.score,p.class_id,p.track_id])
    timestamp = "video" if self.vod else datetime.now().strftime("%Y-%m-%d")
    filepath = BASE_DIR / "cameras" / f"{self.cam_name}/objects/{timestamp}"
    filepath.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "cameras" / f"{self.cam_name}/faces/{timestamp}").mkdir(parents=True, exist_ok=True)
    object_filename = filepath / f"{ts}_{int(p[6])}_{int(p[5])}.jpg"
    x1, y1, x2, y2 = map(int, (p[0], p[1], p[2], p[3]))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    hw = (x2 - x1) // 2
    hh = (y2 - y1) // 2
    hw *= 2
    hh *= 2
    x1_new = cx - hw
    x2_new = cx + hw
    y1_new = cy - hh
    y2_new = cy + hh
    H, W = self.last_frame.shape[:2]
    x1_new = max(0, min(x1_new, W))
    x2_new = max(0, min(x2_new, W))
    y1_new = max(0, min(y1_new, H))
    y2_new = max(0, min(y2_new, H))
    if (y2_new - y1_new) < 100 or (x2_new - x1_new) < 100: return # too small
    crop = self.last_frame[y1_new:y2_new, x1_new:x2_new]
    cv2.imwrite(str(object_filename), crop)
     

  def capture_loop(self):
    self.shape_seg = 0
    frame_size = self.width * self.height * 3
    fail_count = 0
    last_det = -1
    send_det = False
    last_live_check = time.time()
    last_live_seg = time.time()
    last_preview_time = None
    last_counter_update = time.time()
    self.pred_occs = {}
    self.last_link_check = time.time()
    count = 0
    self.det_path = BASE_DIR / "cameras" / self.cam_name / "dets" / datetime.now().strftime("%Y-%m-%d")
    Path(self.det_path).mkdir(parents=True, exist_ok=True)
    self.det_manifest = str(self.det_path / "det_manifest.txt")
    prev_time = time.time()

    if self.vod:
      self.cap = cv2.VideoCapture(self.src)
      self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
      self.frame_step = max(1, int(round(self.src_fps / self.max_frame_rate)))
    while self.running:
      if time.time() - last_live_check > 5:
        last_live_check = time.time()
        link = database.run_get("links", self.cam_name)
        if type(link) == list: link = link[0] # todo, flakey?
        if link != self.src:
          self.src = link
          self._open_ffmpeg()
      try:
        if not (BASE_DIR / "cameras" / self.cam_name).is_dir(): os._exit(1) # deleted cam
        if self.vod:
          for _ in range(self.frame_step - 1):
            self.cap.grab()  # skip for max fps
          ret, frame = self.cap.read()
          if not ret or self.cam_name not in database.run_get("links", None):
            self.running = False
            database.run_put("analysis_prog", cam_name, {"Tracking":100})
            os._exit(0)
          else:
            self.last_preds, _ = self.run_inference(frame)
            database.run_put("analysis_prog", cam_name, {"Tracking":self.cap.get(cv2.CAP_PROP_POS_FRAMES)/self.cap.get(cv2.CAP_PROP_FRAME_COUNT)*100})
        else:
          raw_bytes = self.proc.stdout.read(frame_size)
          if len(raw_bytes) != frame_size:
            fail_count += 1
            if fail_count > 5:
              print(f"{self.cam_name} FFmpeg frame read failed (count={fail_count}), restarting stream...{self.src}")
              self._open_ffmpeg()
              fail_count = 0
            time.sleep(0.5)
            continue
          else:
            fail_count = 0
          frame = np.frombuffer(raw_bytes, np.uint8).reshape((self.height, self.width, 3))
        filtered_preds = self.last_preds

        if count > 10:
          if last_preview_time is None or time.time() - last_preview_time >= 3600: # preview every hour
            last_preview_time = time.time()
            filename = BASE_DIR / "cameras" / f"{self.cam_name}/preview.png"
            write_png(filename, self.raw_frame)
          for _,alert in self.alert_counters.items():
              if alert.desc is not None: continue
              if not alert.is_active():
                alert.reset_counts()
                continue
              window = alert.window if alert.window else (60 if alert.is_notif else 1)
              if not alert.is_active(offset=4): alert.last_det = time.time() # don't send alert when just active
              if alert.get_counts()[1]:
                  if time.time() - alert.last_det >= window:
                    if alert.is_notif and alert.desc is None: send_det = True
                    timestamp = "video" if self.vod else datetime.now().strftime("%Y-%m-%d")
                    filepath = BASE_DIR / "cameras" / f"{self.cam_name}/event_images/{timestamp}"
                    filepath.mkdir(parents=True, exist_ok=True)
                    annotated_frame = draw_predictions(self.last_frame.copy(), filtered_preds, color_dict)
                    # todo alerts can be sent with the wrong thumbnail if two happen quickly, use map
                    ts = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.src_fps) - 5 if self.vod else int(time.time() - self.streamer.start_time - 5)
                    filename = filepath / f"{ts}_notif.jpg" if alert.is_notif else filepath / f"{ts}.jpg"
                    if not self.vod: cv2.imwrite(str(filename), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # we've 10MB limit for video file, raw png is 3MB!
                    if (plain := filepath / f"{ts}.jpg").exists() and (filepath / f"{ts}_notif.jpg").exists():
                      plain.unlink() # only one image per event
                      filename = filepath / f"{ts}_notif.jpg"
                    text = f"Event Detected ({self.cam_name})"
                    if userID is not None and not self.vod and alert.is_notif: threading.Thread(target=send_notif, args=(userID,text,), daemon=True).start()
                    last_det = time.time()
                    alert.last_det = time.time()
          if (send_det and userID is not None and not self.vod) and time.time() - last_det >= 6: #send 15ish second clip after
              export_and_upload(cam_name=self.cam_name, thumbnail=filename, userID=userID, key=key)
              send_det = False
          if userID and not self.vod and (time.time() - last_live_check) >= 5:
              last_live_check = time.time()
              threading.Thread(target=check_upload_link, args=(self.cam_name,), daemon=True).start()
          if (time.time() - last_counter_update) >= 5: #update counter every 5 secs
            last_counter_update = time.time()

            counters = database.run_get("counters", self.cam_name)
            if counters not in [None, {}]:
              if counters.reset:
                self.counter.reset_counts()
                self.counter.reset = False
            database.run_put("counters", cam_name, self.counter)
            
            alerts = database.run_get("alerts", self.cam_name)
            for id,a in alerts.items():
              if not a.new: continue
              a.new = False
              database.run_put("alerts", self.cam_name, a, id=id)
              if a is None:
                del self.alert_counters[id]
                continue
              self.alert_counters[id] = a
              for c in a.classes: classes.add(str(c))
            
            new_settings = database.run_get("settings", self.cam_name)
            if self.settings is not None and new_settings != self.settings and is_vod(self.cam_name):
              self.reset_vod()
              if "reset" in new_settings: del new_settings["reset"]
            self.settings = new_settings
          
          self.alert_counters = {i:a for i,a in self.alert_counters.items() if i in alerts}
              
          if userID and not self.vod and self.cam_name in live_link and live_link[self.cam_name] and (time.time() - last_live_seg) >= 4:
              last_live_seg = time.time()
              mp4_filename = f"segment.mp4"
              export_clip(self.streamer.current_stream_dir_raw, Path(mp4_filename), live=True)
              encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
              Path(mp4_filename).unlink()
              threading.Thread(target=upload_to_r2, args=(Path(f"""{mp4_filename}.aes"""), live_link[self.cam_name]), daemon=True).start()
        else:
            count+=1
        with self.lock:
            self.raw_frame = frame.copy()

        # prob same frame, just cap to an fps maybe
        if frame is not None and self.last_frame is not None and np.mean(np.abs(frame - self.last_frame)) < 5: continue

        # inference
        prev_time = time.time()
        if not any(counter.is_active() for _, counter in self.alert_counters.items()): # don't run inference when no active scheds
          time.sleep(1)
          with self.lock: self.last_preds = [] # to remove annotation when no alerts active
          continue
        with self.lock:
          frame = self.raw_frame.copy() if self.raw_frame is not None else None
        if frame is not None:
          preds, frame = self.run_inference(frame)
          with self.lock:
            self.last_preds = preds.copy()
            self.last_frame = frame.numpy().copy()

          curr_time = time.time()
          fps = 1 / (curr_time - prev_time)
          prev_time = curr_time
          print(f"\rFPS: {fps:.2f}", end="", flush=True)

      except Exception as e:
        print("Error in capture_loop:", e, self.cam_name)
        self._open_ffmpeg()
        time.sleep(1)
  
  def reset_vod(self):
    self.cap = cv2.VideoCapture(self.src) # reset video on settings change
    shutil.rmtree(BASE_DIR / "cameras" / self.cam_name / "objects", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "cameras" / self.cam_name / "faces", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "cameras" / self.cam_name / "event_images", ignore_errors=True)


  def run_inference(self, frame):
    frame = Tensor(frame)
    preds = model(frame).numpy()
    thresh = (self.settings.get("threshold") if self.settings else 0.5) or 0.5 #todo clean!
    online_targets = tracker.update(preds, thresh)
    online_targets = [p for p in online_targets if (classes is None or str(int(p.class_id)) in classes)]
    if type(model) == RFDETR: # RF-DETR has different class_ids
      for j in range(len(online_targets)): online_targets[j].class_id = detr_to_yolo[int(online_targets[j].class_id)]
    preds = []
    for x in online_targets:
      if x.tracklet_len < 1: continue # dont alert for 1 frame, too many false positives.  
      # add to objects, regarless of speed
      if x.track_id not in self.pred_occs: self.pred_occs[x.track_id] = [time.time()]
      if (len(self.pred_occs[x.track_id]) < 20 and (time.time() - self.pred_occs[x.track_id][-1]) > 1) or (time.time() - self.pred_occs[x.track_id][-1]) > 10:
        self. pred_occs[x.track_id].append(time.time())
        ts = round((self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.src_fps) - 5,1) if self.vod else round((time.time() - self.streamer.start_time - 5),1)
        self.save_object(x, ts)

      if x.speed < 2.5: continue #min speed, don't detect still objects, they jitter too. # TODO what's the best min value?
      outside = False
      if hasattr(self, "settings") and self.settings is not None and self.settings.get("coords"):
        scaled_coors = np.array(self.settings["coords"])
        scaled_coors[:,] *= [frame.shape[1], frame.shape[0]] # decimal to full
        outside = point_not_in_polygon([[x.tlwh[0], x.tlwh[1]],[(x.tlwh[0]+x.tlwh[2]), x.tlwh[1]],[(x.tlwh[0]), (x.tlwh[1]+x.tlwh[3])],[(x.tlwh[0]+x.tlwh[2]), (x.tlwh[1]+x.tlwh[3])]], scaled_coors)
        outside = outside ^ self.settings["outside"]
      non_zone_alert = False
      if outside: # check if any alerts don't use zone
        for _, alert in self.alert_counters.items():
          if not alert.zone:
            non_zone_alert = True
            break
        if not non_zone_alert and outside: continue
      preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.score,x.class_id,x.track_id]))
      if (classes is None or str(int(x.class_id)) in classes):
        new = int(x.track_id) not in self.object_set
        new_in_zone = int(x.track_id) not in self.object_set_zone and not outside
        if new:
          self.object_set.add(int(x.track_id))
          self.counter.add(int(x.class_id))
        if new_in_zone: self.object_set_zone.add(int(x.track_id))
        for _, alert in self.alert_counters.items():
          if not alert.get_counts()[1] and ((new and not alert.zone) or (new_in_zone and alert.zone)): alert.add(int(x.class_id))
  
    preds = np.array(preds)
    return preds, frame

  def get_frame(self):
      with self.lock:
          if self.annotated_frame is not None: return self.annotated_frame.copy()
      return None, None

  def release(self):
      self.running = False
      if self.proc:
          self.proc.kill()
      if self.hls_proc:
         self.hls_proc.kill()    

def is_bright_color(color):
  r, g, b = color
  brightness = (r * 299 + g * 587 + b * 114) / 1000
  return brightness > 127

def draw_predictions(frame, preds, color_dict):
  for x1, y1, x2, y2, conf, cls, _ in preds:
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    label = f"{class_labels[int(cls)]}:{conf:.2f}"
    color = color_dict[class_labels[int(cls)]]
    frame = draw_rectangle_numpy(frame, (x1, y1), (x2, y2), color, 3)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    frame = draw_rectangle_numpy(frame, (x1, y1 - text_height - 10), (x1 + text_width + 2, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)
  return frame

class HLSStreamer:
    def __init__(self, video_capture, output_dir="streams", segment_time=4, cam_name="camera", vod=False):
        self.cam_name = cam_name
        self.cam = video_capture
        self.output_dir_det = BASE_DIR / "cameras" / (f"{self.cam_name}_det") / output_dir
        self.output_dir_raw = BASE_DIR / "cameras" / self.cam_name / output_dir
        self.segment_time = segment_time
        self.running = False
        self.start_time = time.time()
        self.feeding_frames = False
        self._stop_event = threading.Event()
        self.output_dir_det.mkdir(parents=True, exist_ok=True)
        self.output_dir_raw.mkdir(parents=True, exist_ok=True)
        self.vod = vod
    
    def _get_new_stream_dir(self):
        timestamp = "video" if self.vod else datetime.now().strftime("%Y-%m-%d")
        stream_dir_det = self.output_dir_det / timestamp
        stream_dir_raw = self.output_dir_raw / timestamp
        stream_dir_det.mkdir(exist_ok=True)
        stream_dir_raw.mkdir(exist_ok=True)
        return stream_dir_det, stream_dir_raw
        
    def start(self):
        self.running = True
        self.current_stream_dir_det, self.current_stream_dir_raw = self._get_new_stream_dir()
        self.start_time = time.time()

    def _safe_restart(self):
        try:
            self.stop()
        except Exception as e:
            print(f"Error during stop: {e}")
        time.sleep(2)
        while True:  # 3 second timeout
            if self.cam.get_frame() is not None:
                break
            time.sleep(1)    
        try:
            self.start()
            print("HLS streamer restarted successfully")
        except Exception as e:
            print(f"Failed to restart HLS: {e}")

    
    def stop(self):
        self.running = False


def point_not_in_polygon(coords, poly):
    n = len(poly)
    for j in range(len(coords)):
      inside = False
      p1x, p1y = poly[0]
      for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if coords[j][1] > min(p1y, p2y):
          if coords[j][1] <= max(p1y, p2y):
            if coords[j][0] <= max(p1x, p2x):
              if p1y != p2y:
                x_intersect = (coords[j][1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
              else:
                x_intersect = p1x
              if p1x == p2x or coords[j][0] <= x_intersect: inside = not inside
        p1x, p1y = p2x, p2y
      if inside:
        return False
    return True

def run_encode_text(return_q, clip, text):
  res = clip.model._encode_text(text, realize=True)
  return_q.put(res)
  return res

def run_search(return_q, clip, image_text, top_k, cam_name, selected_dir):
  res = clip.search(image_text, top_k, cam_name, selected_dir)
  return_q.put(res)

def run_clip(return_q, clip, im, top_k, cam_name, selected_dir, is_face):
  im = clip.preprocess_face(im) if is_face else clip.preprocess_clip(im)
  if im is not None:
    embedding = clip.precompute_face_embedding_bs1_np(im) if is_face else clip.precompute_embedding_bs1_np(im)
    res = clip.search(None, top_k, cam_name, selected_dir, embedding, is_face)
  else:
    res = []
  return_q.put(res)

class HLSRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.object_finder = kwargs.pop('clip_instance', None)
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args): pass # don't print stuff

    def send_results(self, results, start=0, count=100):
      image_data = []
      for path_str, score in results:
        if score < 0.21: break
        img_path = Path(path_str).resolve()
        ts = event_img_info(img_path.stem)["ts"]
        parts = img_path.parts
        cam_index = parts.index("cameras") + 1
        cam = parts[cam_index]
        rel = img_path.relative_to(BASE_DIR / "cameras")
        image_url = f"/{rel}"
        image_data.append({
          "url": image_url,
          "timestamp": ts,
          "filename": img_path.name,
          "cam_name": cam,
          "folder": img_path.parts[-2],
          "score": score,
        })
      image_data = image_data[start:start+count]
      response_data = {
        "images": image_data,
        "count": len(image_data),
      }
      self.send_200(response_data)

    def send_200(self, body=None):
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(json.dumps(body).encode("utf-8"))

    def get_camera_path(self, cam_name=None):
        if cam_name:
            return BASE_DIR / "cameras" / cam_name / "streams"
        return BASE_DIR / "cameras"
    
    def do_GET(self):
        parsed_path = urlparse(unquote(self.path))
        query = parse_qs(parsed_path.query)
        cam_name = query.get("cam", [None])[0]

        if parsed_path.path == "/set_max_storage":
          max_gb = float(query.get("max", [None])[0])
          self.server.max_gb = max_gb
          database.run_put("max_storage", "all", max_gb)
          self.send_200()
          return
        
        if parsed_path.path == "/get_features":
          self.send_200({"object_search":use_clip, "face_search":use_face})
          return
        if parsed_path.path == "/get_max_storage":
          self.send_200(body={"max_gb":self.server.max_gb})
          return

        if parsed_path.path == "/list_cameras":
          cams = database.run_get("links", None)
          progs = database.run_get("analysis_prog", None)
          cam_progress = {cam_name: progs.get(cam_name, None) for cam_name in cams}
          self.send_200(cam_progress)
          return

        if parsed_path.path == "/list_days":          
          base_path = "data/cameras"
          days = set()
          date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
          if os.path.exists(base_path):
            for cam_name in os.listdir(base_path):
              cam_path = os.path.join(base_path, cam_name, "streams")    
              if os.path.exists(cam_path):
                for date_folder in os.listdir(cam_path):
                  date_folder_path = os.path.join(cam_path, date_folder)
                  if os.path.isdir(date_folder_path) and date_pattern.match(date_folder): days.add(date_folder)
          days_list = sorted(list(days), reverse=True, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
          self.send_200(days_list)
          return

        if parsed_path.path == '/add_camera':
            cam_name = query.get("cam_name", [None])[0]
            src = query.get("src", [None])[0]
            
            if not cam_name or not src:
                self.send_error(400, "Missing cam_name or src")
                return
            
            start_cam(rtsp=src,cam_name=cam_name,model_variant=model_variant)
            database.run_put("links", cam_name, src)
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return
        
        if parsed_path.path == "/edit_settings":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return
            zone = database.run_get("settings", cam_name)
            if zone is None: zone = {}
            coords_json = query.get("coords", [None])[0]
            if coords_json is not None:
              coords = json.loads(coords_json)
              if isinstance(coords, list):
                if len(coords) >= 3:
                  zone["coords"] = [[float(x), float(y)] for x, y in coords]
                else:
                  if "coords" in zone: del zone["coords"]
            zone["is_notif"] = (str(is_notif).lower() == "true") if (is_notif := query.get("is_notif", [None])[0]) is not None else zone.get("is_notif")
            zone["outside"] = (str(outside).lower() == "true") if (outside := query.get("outside", [None])[0]) is not None else zone.get("outside")
            query.get("threshold", [None])[0] is not None and zone.update({"threshold": float(query.get("threshold", [None])[0])}) #need the val  
            database.run_put("settings", cam_name, zone) # todo, key for each
            if (url := query.get("url")) is not None: database.run_put("links", cam_name, url)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/edit_alert":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return

            raw_alerts = database.run_get("alerts", cam_name)
            alert = None
            alert_id = query.get("id", [None])[0]
            is_on = query.get("is_on", [None])[0]
            zone = query.get("zone", [None])[0]
            is_notif = query.get("is_notif", [None])[0]
            desc = query.get("desc", [None])[0]
            threshold = query.get("threshold", [None])[0]
            if threshold is not None: threshold = float(threshold) / 100
            if alert_id is None: # no id, add alert
                window = query.get("window", [None])[0]
                max_count = query.get("max", [None])[0]
                class_ids = query.get("class_ids", [None])[0]
                sched = json.loads(query.get("sched", ["[[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400]]"])[0]) # todo, weekly
                if window: window = int(window)
                max_count = int(max_count)
                classes = [int(c.strip()) for c in class_ids.split(",")]
                alert_id = str(uuid.uuid4())
                alert = RollingClassCounter(
                        window_seconds=window,
                        max=max_count,
                        classes=classes,
                        sched=sched,
                        cam_name=cam_name,
                        desc=desc,
                        threshold=threshold
                    )
                raw_alerts[alert_id] = alert
            else:
              if is_on is not None or is_notif is not None or zone is not None:
                if is_on is not None: raw_alerts[alert_id].is_on = str(is_on).lower() == "true"
                if is_notif is not None: raw_alerts[alert_id].is_notif = str(is_notif).lower() == "true"
                if zone is not None: raw_alerts[alert_id].zone = str(zone).lower() == "true"
                if desc is not None: raw_alerts[alert_id].desc = desc
                if threshold is not None: raw_alerts[alert_id].threshold = threshold
                alert = raw_alerts[alert_id]
                alert.new = True
              else:
                del raw_alerts[alert_id]
            if alert is not None:
              database.run_put("alerts", cam_name, alert, alert_id)
            else:
              database.run_delete("alerts", cam_name, alert_id)
            
            # make vod reset
            settings = database.run_get("settings", cam_name)
            if settings is None: settings = {}
            settings["reset"] = True
            database.run_put("settings", cam_name, settings)

 
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/get_settings":
            zone = database.run_get("settings",cam_name)
            if zone is not None:
              if cam_name in zone and "settings" in zone[cam_name]: zone = zone[cam_name]["settings"]
            else:
              zone = {}
            
            self.send_200(zone)
            return

        if parsed_path.path == "/get_alerts":
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return

            raw_alerts = database.run_get("alerts",cam_name)
            alert_info = []
            for key,alert in raw_alerts.items():
                sched = alert.sched if alert.sched else [[0,86399],True,True,True,True,True,True,True]
                alert_info.append({
                    "window": alert.window,
                    "max": alert.max,
                    "classes": list(alert.classes),
                    "id": str(key),
                    "sched": sched,
                    "is_on": alert.is_on,
                    "is_notif": alert.is_notif,
                    "zone": alert.zone,
                    "desc": alert.desc,
                    "threshold": alert.threshold,
                })
            self.send_200(alert_info)
            return

        if parsed_path.path == '/delete_camera':
            cam_name = query.get("cam_name", [None])[0]
            if not cam_name:
                self.send_error(400, "Missing cam_name parameter")
                return
            
            try:
              shutil.rmtree(BASE_DIR / "cameras" / (cam_name + "_det"), ignore_errors=True)
              shutil.rmtree(BASE_DIR / "cameras" / cam_name, ignore_errors=True)
              if os.path.isfile(database.run_get("links", None)[cam_name]): os.remove(database.run_get("links", None)[cam_name])
              # todo clean
              alerts = database.run_get("alerts", cam_name)
              for id, _ in alerts.items():
                database.run_delete("alerts", cam_name, id=id)
              database.run_delete("links", cam_name)
              database.run_delete("analysis_prog", cam_name)
              database.run_delete("settings", cam_name)
              database.run_delete("counters", cam_name)
            except Exception as e:
              self.send_error(500, f"Error deleting camera: {e}")
              return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"deleted"}')
            return

        if parsed_path.path == "/get_counts": # todo error fetching counts on first
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return
            
            counter = database.run_get("counters", cam_name)
            if counter:
              labeled_counts = {
                class_labels[int(k)]: len(v)
                for k, v in counter.data.items()
                if int(k) < len(class_labels)
              }
              self.send_200(labeled_counts)
              return
            else:
              database.run_put("counters", cam_name, RollingClassCounter(cam_name=cam_name))
              self.send_200([])
        
        if parsed_path.path == "/shutdown": 
          threading.Thread(target=self.server.shutdown).start()
          for proc in active_subprocesses:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
          sys.exit(0)

        if parsed_path.path == "/reset_counts":
          if not cam_name:
            self.send_error(400, "Missing cam parameter")
            return
          counter = database.run_get("counters",cam_name)
          if counter: counter.reset_counts()
          database.run_put("counters", cam_name, counter)
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.end_headers()
          self.wfile.write(b"{}")
          return


        if parsed_path.path == '/' and "cam" not in query:
          with open("mainview.html", "r", encoding="utf-8") as f: html = f.read()
          self.send_response(200)
          self.send_header('Content-type', 'text/html')
          self.end_headers()
          self.wfile.write(html.encode('utf-8'))
          return
                            
        if parsed_path.path == '/' or parsed_path.path == f'/{cam_name}':
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            start_param = parse_qs(parsed_path.query).get("start", [None])[0]
            try:
                start_time = max(float(start_param),0) if start_param is not None else None
            except ValueError:
                start_time = None

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('cameraview.html', 'r', encoding='utf-8') as f: html = f.read()
            replacements = {
                '{selected_dir}': selected_dir,
                '{class_labels}': json.dumps(class_labels),
                '{start_time}': str(start_time) if start_time is not None else 'null',
                '{cam_name}': cam_name
            }
            for placeholder, value in replacements.items(): html = html.replace(placeholder, value)
            self.wfile.write(html.encode("utf-8"))
            return
        
        requested_path = parsed_path.path.lstrip('/')
        if requested_path.startswith("cameras/"):
            requested_path = requested_path[len("cameras/"):]

        cam_name = requested_path[:requested_path.index("/")]
        vod = is_vod(cam_name)
        # todo hack
        if vod and "preview.png" not in requested_path: requested_path = requested_path.rsplit("/", 2)[0] + "/video/" + requested_path.rsplit("/", 1)[1]

        file_path = BASE_DIR / "cameras" / requested_path

        if not file_path.exists():
            self.send_error(404)
            return

        self.send_response(200)
        if file_path.suffix == '.m3u8':
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_header('Cache-Control', 'no-cache')
        elif file_path.suffix == '.ts':
            self.send_header('Content-Type', 'video/MP2T')
        elif file_path.suffix == '.png':
            self.send_header('Content-Type', 'image/jpeg')
        self.end_headers()

        with open(file_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if self.path.startswith("/analyse-footage"):
          params = parse_qs(parsed_path.query)
          filename = params.get("filename", [None])[0]
          chunk = int(params.get("chunk", [0])[0])
          total = int(params.get("total", [1])[0])
          if not filename:
            self.send_error(400, "Missing filename")
            return
          filename = os.path.basename(filename)
          upload_dir = BASE_DIR / "cameras"
          upload_dir.mkdir(exist_ok=True)
          length = int(self.headers.get("Content-Length", 0))
          if length <= 0:
            self.send_error(411, "Content-Length required")
            return
          final_path = upload_dir / filename
          temp_path = upload_dir / f"{filename}.part"
          with open(temp_path, "ab") as f:
            remaining = length
            while remaining > 0:
              data = self.rfile.read(min(1024 * 1024, remaining))
              if not data: break
              f.write(data)
              remaining -= len(data)
          if chunk == total - 1: temp_path.rename(final_path)
          self.send_200([])

        if parsed_path.path == "/event_thumbs":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)

            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            cam_name     = data.get("cam")
            selected_dir = data.get("folder")
            name_contains = data.get("name_contains")
            image_text   = data.get("image_text")
            similar_img  = data.get("similar_img")
            start = data.get("start")
            count = data.get("count")
            is_face = data.get("is_face") or False
            if is_face and not use_face: return
            if start is None: start, count = 0, 100
            uploaded_image = data.get("uploaded_image")
            if uploaded_image:
              if ',' in uploaded_image: uploaded_image = uploaded_image.split(',')[1]
              uploaded_image = base64.b64decode(uploaded_image)

            if cam_name:
              camera_dirs = [BASE_DIR / "cameras" / cam_name]
            else:
              camera_dirs = [d for d in (BASE_DIR / "cameras").iterdir() if d.is_dir()]

            if selected_dir:
              selected_dirs = [selected_dir]
            else:
              selected_dirs = list({
                subdir.name 
                for camera_dir in camera_dirs
                if (camera_dir / "streams").is_dir()
                for subdir in (camera_dir / "streams").iterdir() 
                if subdir.is_dir()
              })
            selected_dirs.append("video")

            if image_text and use_clip: self.object_finder._load_all_embeddings()
            if (uploaded_image or similar_img) and (use_clip or use_face): self.object_finder._load_all_embeddings(face=is_face)
            
            if uploaded_image and (use_clip or is_face):
              results = process_with_clip_lock(run_clip, self.object_finder, uploaded_image, start+count, cam_name, selected_dir, is_face)
              self.send_results(results, start, count)
              return
            
            if similar_img and (use_clip or is_face):
              results = process_with_clip_lock(run_clip, self.object_finder, similar_img, start+count, cam_name, selected_dir, is_face) # todo one with above
              self.send_results(results, start, count)
              return

            if image_text and use_clip:
              results = process_with_clip_lock(run_search, self.object_finder, image_text, start+count, cam_name, selected_dir)
              self.send_results(results, start, count)
              return

            image_data = []
            for camera_dir in camera_dirs:
              for selected_dir in selected_dirs:
                event_image_path = camera_dir / "event_images" / selected_dir
                if not event_image_path.exists(): continue
                event_images = sorted(
                  event_image_path.glob("*.jpg"),
                  key=lambda p: int(p.stem.split('_')[0]),
                  reverse=True
                )
                for img in event_images:
                  if name_contains and name_contains not in img.name: continue
                  ts = int(img.stem.split('_')[0])
                  image_url = f"/{img.relative_to(BASE_DIR)}"
                  image_data.append({
                    "url": image_url,
                    "timestamp": ts,
                    "filename": img.name,
                    "cam_name": camera_dir.name,
                    "folder": selected_dir,
                  })

            image_data.sort(key=image_sort_key, reverse=True)
            if start is not None and count is not None: image_data = image_data[start:start+count]

            response_data = {
              "images": image_data,
              "count": len(image_data),
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

def image_sort_key(item):
  try: return datetime.strptime(item["folder"], "%Y-%m-%d").timestamp() + item["timestamp"]
  except ValueError: return -1

def schedule_daily_restart(hls_streamer, videocapture, restart_time):
    while True:
        now = datetime.now().time()
        target = time_obj(restart_time[0], restart_time[1])
        if now >= target:
            delta = (24 * 3600) - ((now.hour * 3600 + now.minute * 60 + now.second) - (target.hour * 3600 + target.minute * 60))
        else:
            delta = ((target.hour * 3600 + target.minute * 60) - 
                    (now.hour * 3600 + now.minute * 60 + now.second))
        time.sleep(delta)
        videocapture.hls_proc.kill()
        sys.stdout.flush()
        python = sys.executable
        os.execv(python, [python] + sys.argv)

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_executable_args(): return ([sys.argv[0]], sys.argv[1:]) if getattr(sys, "frozen", False) else ([sys.executable, sys.argv[0]], sys.argv[1:])

def event_img_info(image): return {"ts": int(float(image.split('_')[0])), "object_id":int(image.split('_')[1]), "class_id":int(image.split('_')[2])}

def start_cam(rtsp, cam_name, model_variant='t', yolo_res=960):
    if not rtsp or not cam_name: return
    def upsert_arg(args, key, value):
        prefix = f"--{key}="
        for i, arg in enumerate(args):
            if arg.startswith(prefix):
                args[i] = f"{prefix}{value}"
                return args
        return args + [f"{prefix}{value}"]

    executable, base_args = get_executable_args()
    new_args = upsert_arg(base_args, "cam_name", cam_name)
    new_args = upsert_arg(new_args, "rtsp", rtsp)
    new_args = upsert_arg(new_args, "yolo_size", model_variant)
    new_args = upsert_arg(new_args, "yolo_res", yolo_res)
    new_args = upsert_arg(new_args, "use_clip", use_clip)
    proc = subprocess.Popen(executable + new_args, close_fds=True)
    active_subprocesses.append(proc)


live_link = dict()
alerts_on = True
is_live_lock = threading.Lock()
def check_upload_link(cam_name="camera"):
    global live_link
    global alerts_on
    query_params = urllib.parse.urlencode({
        "name": quote(cam_name),
        "session_token": userID
    })
    url = f"https://clearcam.org/get_stream_upload_link?{query_params}"
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        if response.status == 200:
            response_data = json.loads(response.read().decode('utf-8'))
            upload_link = response_data.get("upload_link")
            alerts_on_res = response_data.get("alerts_on")
            with is_live_lock:
               live_link[cam_name] = upload_link
               alerts_on = (alerts_on_res == 1)
        else:
            with is_live_lock:
                if cam_name in live_link: 
                    live_link[cam_name] = None

def upload_to_r2(file_path: Path, signed_url: str, max_retries: int = 0) -> bool:
    try:
        url_parts = urllib.parse.urlparse(signed_url)
        if url_parts.scheme == 'https':
            conn = http.client.HTTPSConnection(url_parts.netloc)
        else:
            conn = http.client.HTTPConnection(url_parts.netloc)
        
        with file_path.open('rb') as f:
            headers = {'Content-Type': 'application/octet-stream'}
            conn.request("PUT", url_parts.path + "?" + url_parts.query, body=f, headers=headers)
            response = conn.getresponse()
            if 200 <= response.status < 300:
                return True
            return False
    except Exception as e:
        print(f"Error uploading to R2: {e}")
        return False

object_finder_lock = threading.Lock()
def process_with_clip_lock(func, *args):
    if not object_finder_lock.acquire(timeout=30): return None
    try:
      return_q = multiprocessing.Queue()
      p = multiprocessing.Process(target=func, args=(return_q, *args))
      p.start()
      results = return_q.get(timeout=300)
      p.join()
      return results
    finally:
      object_finder_lock.release()

cams = dict()
active_subprocesses = []
import socket
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    def __init__(self, server_address, use_clip, face, RequestHandlerClass):
        ThreadingMixIn.__init__(self)
        HTTPServer.__init__(self, server_address, RequestHandlerClass)
        self.cleanup_stop_event = threading.Event()
        self.cleanup_thread = None
        max_gb = database.run_get("max_storage", None)
        if max_gb == {}:
          database.run_put("max_storage", "all", 256)
          max_gb = database.run_get("max_storage", None)
        self.max_gb = max_gb["all"]
        self.object_finder = ObjectFinder(prewarm=True, clip=use_clip, face=face) if (use_clip or face) else None
        self.object_finder_stop_event = threading.Event()
        self.object_finder_thread = None
        self._setup_cleanup_and_clip_thread()

    def finish_request(self, request, client_address):
      self.RequestHandlerClass(request, client_address, self, clip_instance=self.object_finder)

    def _setup_cleanup_and_clip_thread(self):
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
          self.cleanup_stop_event.clear()
          self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True, name="StorageCleanup")
          self.cleanup_thread.start()

        if (use_clip or use_face) and (self.object_finder_thread is None or not self.object_finder_thread.is_alive()):
          self.object_finder_stop_event.clear()
          self.object_finder_thread = threading.Thread(target=self._objects_task, daemon=True, name="CLIPMaintenance")
          self.object_finder_thread.start()

    def _cleanup_task(self):
        while not self.cleanup_stop_event.is_set():
            try:
                self._check_and_cleanup_storage()
            except Exception as e:
                print(f"Cleanup error: {e}")
            self.cleanup_stop_event.wait(timeout=600)

    def _objects_task(self):
      while not self.object_finder_stop_event.is_set():
        try:
          object_folders = self.object_finder.find_object_folders("data/cameras")
          for folder in object_folders:
            name = folder.split("/")[2]
            vod = is_vod(name)
            if vod and name in database.run_get("analysis_prog", None) and database.run_get("analysis_prog", None)[name]["Tracking"] < 100: continue
            embeddings, batch_paths = self.object_finder.precompute_embeddings(folder, vod=vod, database=database, cam_name=name, userID=userID, key=key)
            alerts = database.run_get("alerts", name)
            if userID:
              for path, embedding in zip(batch_paths, embeddings):
                for k, v in alerts.items():
                    if time.time() - v.last_det < 60 or not v.is_active(): continue
                    if v.desc is not None and hasattr(v, "desc_emb") and v.desc_emb is not None:
                        similarity = (v.desc_emb @ embedding.T).item()
                        print("sim =",similarity,v.desc,path)
                        if similarity > v.threshold:
                            send_notif(userID, f"Event Detected ({name}: {v.desc})")
                            alerts[k].last_det = time.time()
                            database.run_put("alerts", name, alerts[k], k)
                            seen_time = event_img_info(path.split("/")[-1].split(".jpg")[0])["ts"]
                            export_and_upload(cam_name=name, thumbnail=path, userID=userID, start=seen_time, length=20, key=key)
                            break

            if vod: database.run_delete("analysis_prog", folder.split("/")[2])
            # todo, move to own loop
            for k, alert in alerts.items():
              if alert.desc is not None and alert.desc_emb is None:
                alert.desc_emb = process_with_clip_lock(run_encode_text, self.object_finder, alert.desc)
                database.run_put("alerts", name, alert, id=k)

        except Exception as e:
          print(f"CLIP error: {e}")
        self.object_finder_stop_event.wait(timeout=1)

    def _check_and_cleanup_storage(self):
      total_size = sum(f.stat().st_size for f in (BASE_DIR / "cameras").glob('**/*') if f.is_file())
      size_gb = total_size / (1000 ** 3)
      free_gb = shutil.disk_usage(BASE_DIR / "cameras").free / (1000 ** 3)
      if size_gb > self.max_gb or free_gb < 5: self._cleanup_oldest_files() # todo unhardcode

    def _cleanup_oldest_files(self):
      camera_dirs = []
      for cam_dir in (BASE_DIR / "cameras").iterdir():
        if cam_dir.is_dir():
          dir_size = sum(f.stat().st_size for f in cam_dir.glob('**/*') if f.is_file())
          camera_dirs.append((cam_dir, dir_size))
      
      if not camera_dirs: return
          
      largest_cam_raw = max(camera_dirs, key=lambda x: x[1])[0]
      largest_cam_det = largest_cam_raw.with_stem(largest_cam_raw.stem + "_det")
      for largest_cam in [largest_cam_raw, largest_cam_det]:
        streams_dir = largest_cam / "streams"
        if not streams_dir.exists():
          shutil.rmtree(largest_cam)
          continue
            
        recordings = []
        for rec_dir in streams_dir.iterdir():
          if rec_dir.is_dir(): recordings.append((rec_dir, rec_dir.stat().st_ctime))
        if not recordings:
          shutil.rmtree(largest_cam)
          print(f"Deleted camera folder with empty streams: {largest_cam}")
          continue
        recordings.sort(key=lambda x: x[1])
        oldest_recording = recordings[0][0]
        shutil.rmtree(oldest_recording)
        event_images_dir = largest_cam.with_name(largest_cam.name) / Path("event_images") / Path(oldest_recording.name)
        object_images_dir = largest_cam.with_name(largest_cam.name) / Path("objects") / Path(oldest_recording.name)
        face_images_dir = largest_cam.with_name(largest_cam.name) / Path("objects") / Path(oldest_recording.name)
        #dets_dir = largest_cam.with_name(largest_cam.name) / Path("dets") / Path(oldest_recording.name)
        if event_images_dir.exists(): shutil.rmtree(event_images_dir)
        if object_images_dir.exists(): shutil.rmtree(object_images_dir)
        if face_images_dir.exists(): shutil.rmtree(face_images_dir)
        #if dets_dir.exists(): shutil.rmtree(dets_dir)
        print(f"Deleted oldest recording: {oldest_recording}")

    def server_close(self):
        if hasattr(self, 'cleanup_stop_event'):
            self.cleanup_stop_event.set()
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        if hasattr(self, 'clip_stop_event'):
            self.object_finder_stop_event.set()
        if hasattr(self, 'clip_thread') and self.object_finder_thread:
            self.object_finder_thread.join(timeout=5)

        super().server_close()

if __name__ == "__main__":
  multiprocessing.set_start_method("spawn", force=True)
  database = db()
  cams = database.run_get("links", None)
  url = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--rtsp=")), None)
  is_file = url.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) if url is not None else False
  classes = {"0","1","2","7"} # person, bike, car, truck, bird (14)

  userID = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--userid=")), None)
  key = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--key=")), None)
  use_clip = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--use_clip=")), None)
  if use_clip: use_clip = use_clip != "False" # str to bool
  use_face = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--use_face=")), None)
  if use_face: use_face = use_face != "False" # str to bool
  model_variant = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--yolo_size=")), None)
  yolo_res = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--yolo_res=")), 640)
  if not model_variant:
    model_variant = int(input("Select a YOLOv9 model from: \n1: tiny\n2: small\n3: medium\n4: large\n5: xl\nor an RFDETER Model from \n6: nano\n7: small\n8: medium\n9: large\nor press enter to skip (defaults to tiny):") or "1")
    if model_variant < 6:
      yolo_ress = {"1":320, "2":640, "3":960,"4":1280,"5":1536}
      yolo_res = yolo_ress[input("\nSelect a YOLOV9 resoltuion from \n1: 320\n2: 640\n3: 960\n4: 1280\n5: 1536\nor press enter to skip (defaults to 960):") or "3"]
    use_clip = input("Would you like to enable clip search on events? (y/n) (1.7GB model), or press enter to skip:") or False
    use_clip = use_clip in ["y", "Y"]
    use_face = input("Would you like to enable (experimental) face recognition search? (y/n), or press enter to skip:") or False
    use_face = use_face in ["y", "Y"]

  if url is None and userID is None:
    userID = input("enter your Clearcam user id or press Enter to skip: ")
    if len(userID) > 0:
      key = ""
      while len(key) < 1: key = input("enter a password for encryption: ")
      sys.argv.extend([f"--use_clip={use_clip}", f"--use_face={use_face}" ,f"--userid={userID}", f"--key={key}", f"--yolo_size={model_variant}"])
    else: userID = None

  if userID is not None and key is None:
    print("Error: key is required when userID is provided")
    sys.exit(1)
  
  if use_clip or use_face:
    from objects import ObjectFinder

  cam_name = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--cam_name=")), "my_camera")


  from ocsort_tracker import ocsort
  tracker = ocsort.OCSort(max_age=100)
  live_link = dict()
  
  if url is None:
    for cam_name in cams.keys():
      start_cam(rtsp=cams[cam_name],cam_name=cam_name,model_variant=model_variant, yolo_res=yolo_res)

  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  if url:
    model = YOLOv9(models[int(model_variant)], res=int(yolo_res)) if int(model_variant) < 6 else RFDETR(models[int(model_variant)])
    #model = RFDETR("small")
    cam = VideoCapture(url,cam_name=cam_name, vod=is_file)
    vod = url.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    hls_streamer = HLSStreamer(cam,cam_name=cam_name, vod=vod)
    cam.streamer = hls_streamer
  
  try:
    try:
      server = ThreadedHTTPServer(('0.0.0.0', 8080), use_clip=use_clip, face=use_face, RequestHandlerClass=HLSRequestHandler)
      print(f"Serving at http://{get_lan_ip()}:8080")
    except OSError as e:
      if e.errno == socket.errno.EADDRINUSE:
        print("Port in use, server not started.")
        server = None
      else:
          raise
    
    if url:
      hls_streamer.start()
      restart_time = (0, 0)
      threading.Thread(
        target=schedule_daily_restart,
        args=(hls_streamer, cam, restart_time),
        daemon=True
      ).start()

    if server:
      server.serve_forever()
    else:
      while True: time.sleep(3600)

  except KeyboardInterrupt:
    if url:
      hls_streamer.stop()
      cam.release()
      server.shutdown()

