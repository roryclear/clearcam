import uuid
import http
from pathlib import Path
import shutil
from collections import deque, defaultdict
import time
import os
import subprocess
import urllib
import json
import struct
from datetime import datetime
import threading
import cv2
import numpy as np
BASE_DIR = Path(__file__).parent.parent / "data"
from tinygrad import Tensor

def send_notif(session_token: str, text=None):
    host = "www.clearcam.org"
    endpoint = "/send" #/test
    boundary = f"Boundary-{uuid.uuid4()}"
    content_type = f"multipart/form-data; boundary={boundary}"
    lines = [
        f"--{boundary}",
        'Content-Disposition: form-data; name="session_token"',
        "",
        session_token,
        f"--{boundary}--",
        ""
    ]
    if text is not None:
      lines.extend([
      f"--{boundary}",
      'Content-Disposition: form-data; name="text"',
      "",
      text,
    ])
    body = "\r\n".join(lines).encode("utf-8")
    conn = http.client.HTTPSConnection(host)
    headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
    try:
        conn.request("POST", endpoint, body, headers)
        response = conn.getresponse()
        print(f"Status: {response.status} {response.reason}")
        print(response.read().decode())
    except Exception as e:
        print(f"Error sending session token: {e}")
    finally:
        conn.close()


def draw_bounding_boxes(orig_img_path, predictions, class_labels):
  color_dict = {
      label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256))
      for i, label in enumerate(class_labels)
  }
  font = cv2.FONT_HERSHEY_SIMPLEX

  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  orig_img = (
      cv2.imread(orig_img_path)
      if not isinstance(orig_img_path, np.ndarray)
      else cv2.imdecode(orig_img_path, 1)
  )

  height, width, _ = orig_img.shape
  box_thickness = int((height + width) / 400)
  font_scale = (height + width) / 2500
  object_count = defaultdict(int)
  
  for pred in predictions:
    if len(pred) == 7: # todo
      x1, y1, x2, y2, conf, class_id, _ = pred
    else:
      x1, y1, x2, y2, conf, class_id = pred
    if conf == 0:
        continue

    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]

    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)

    label = f"{class_labels[class_id]} {conf:.2f}"
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = (
        (y1 - 4, y1 - text_size[1] - 4)
        if y1 - text_size[1] - 4 > 0
        else (y1 + text_size[1], y1)
    )

    cv2.rectangle(
        orig_img,
        (x1, bg_y),
        (x1 + text_size[0], bg_y + text_size[1]),
        color,
        -1,
    )

    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(
        orig_img,
        label,
        (x1, label_y),
        font,
        font_scale,
        font_color,
        1,
        cv2.LINE_AA,
    )

    object_count[class_labels[class_id]] += 1
  return orig_img

def resize(img, new_size):
  img = img.permute(2,0,1)
  img = Tensor.interpolate(img, size=(new_size[1], new_size[0]), mode='linear', align_corners=False)
  img = img.permute(1, 2, 0)
  return img

def export_clip(stream_dir, output_path: Path, live=False, length=5, end=0, start=None):
  segments = sorted(stream_dir.glob("*.ts"), key=os.path.getmtime)
  recent_segments = deque()
  cutoff = os.path.getmtime(segments[0]) + start if start is not None else time.time() - length if live else time.time() - length
  end = os.path.getmtime(segments[0]) + start + length if start is not None else time.time() - end
  recent_raw = [f for f in segments if os.path.getmtime(f) >= cutoff and os.path.getmtime(f) <= end]
  recent_segments.extend(recent_raw)
  concat_list_path = stream_dir / "concat_list.txt"
  with open(concat_list_path, "w") as f: f.writelines(f"file '{segment.resolve()}'\n" for segment in recent_segments)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  ffmpeg_path = find_ffmpeg()
  if live:
    command = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-loglevel", "quiet",
        "-vf", "scale=-2:240,fps=24,format=yuv420p",
        "-c:v", "libx264",  
        "-pix_fmt", "yuv420p",
        "-preset", "veryslow",
        "-crf", "32",
        "-an",
        str(output_path)
    ]
    subprocess.run(command, check=True)
  else:
    with open(stream_dir / "concat_list.txt", "r") as f: print(" ".join(line.strip() for line in f))
    command = [
      ffmpeg_path,
      "-y",
      "-f", "concat",
      "-safe", "0",
      "-i", str(concat_list_path),
      "-c:v", "libx264",
      "-crf", "18",
      "-pix_fmt", "yuv420p",
      "-an",  # No audio
      str(output_path)
    ]
    subprocess.run(command, check=True)
    comp = 5
    file_size = 10*1024*1024
    with open(output_path, "rb") as f:
      file_data = f.read()
      file_size = len(file_data)
    while file_size >= 9*1024*1024: # max size 10MB, # todo, calculate time from ts files
      temp_output = output_path.with_stem(output_path.stem + "_compressed")
      command = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c:v", "libx264",
        "-crf", str(18 + comp),
        "-pix_fmt", "yuv420p",  # needed for android
        "-an",  # No audio
        str(temp_output)
      ]
      subprocess.run(command, check=True)
      os.replace(temp_output, output_path)
      with open(output_path, "rb") as f:
        file_data = f.read()
      file_size = len(file_data)
      comp += 5

def export_and_upload(cam_name, thumbnail, userID, key, start=None, end=0, length=20):
    os.makedirs(BASE_DIR / "cameras" / cam_name / "event_clips", exist_ok=True)
    mp4_filename = BASE_DIR / "cameras" / f"{cam_name}/event_clips/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
    temp_output = BASE_DIR / "cameras" / f"{cam_name}/event_clips/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_temp.mp4"
    export_clip(BASE_DIR / "cameras" / f"{cam_name}/streams/{datetime.now().strftime('%Y-%m-%d')}", Path(mp4_filename), length=length, start=start, end=end)
    subprocess.run(['ffmpeg', '-i', mp4_filename, '-i', str(thumbnail), '-map', '0', '-map', '1', '-c', 'copy', '-disposition:v:1', 'attached_pic', '-y', temp_output])
    os.replace(temp_output, mp4_filename)
    encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), key)
    threading.Thread(target=upload_file, args=(Path(f"""{mp4_filename}.aes"""), userID), daemon=True).start()
    os.unlink(mp4_filename)

def find_ffmpeg():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    common_paths = [
        '/opt/homebrew/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/usr/bin/ffmpeg'
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return 'ffmpeg'

def upload_file(file_path: Path, session_token: str):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        file_name = file_path.name
        file_size = len(file_data)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    try:
        params = {
            "filename": file_name,
            "session_token": session_token,
            "size": str(file_size)
        }
        query_string = urllib.parse.urlencode(params)
        url = f"https://clearcam.org/upload?{query_string}"
        
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                print(f"Failed to get upload URL: {response.status}")
                return False
            response_data = json.loads(response.read().decode('utf-8'))
            presigned_url = response_data.get("url")
            if not presigned_url:
                print("Invalid response - missing upload URL")
                return False
    except Exception as e:
        print(f"Error getting upload URL: {e}")
        return False
    success = False
    for attempt in range(10):
        try:
            url_parts = urllib.parse.urlparse(presigned_url)
            if url_parts.scheme == 'https':
                conn = http.client.HTTPSConnection(url_parts.netloc)
            else:
                conn = http.client.HTTPConnection(url_parts.netloc)
            
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(file_size)
            }
            conn.request("PUT", url_parts.path + "?" + url_parts.query, body=file_data, headers=headers)
            upload_response = conn.getresponse()
            
            if 200 <= upload_response.status < 300:
                os.unlink(file_path) # todo remove on failure or success
                print(f"File uploaded successfully on attempt {attempt + 1}")
                success = True
                conn.close()
                break
            else:
                print(f"Upload failed with status {upload_response.status} on attempt {attempt + 1}")
                conn.close()
        except Exception as e:
            print(f"Upload error on attempt {attempt + 1}: {e}")
        if attempt < 3: time.sleep(10 * attempt)

    try:
        file_path.unlink()
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Failed to delete file: {e}")
    return success

import aes
MAGIC_NUMBER = 0x4D41474943
HEADER_SIZE = 8
AES_BLOCK_SIZE = 16
AES_KEY_SIZE = 32

def prepare_key(key: str) -> bytes:
    key_bytes = key.encode('utf-8')[:AES_KEY_SIZE]
    return key_bytes.ljust(AES_KEY_SIZE, b'\0')

def pkcs7_pad(data: bytes, block_size: int) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def encrypt_cbc(data: bytes, key: bytes, iv: bytes) -> bytes:
    aes_cipher = aes.AES(key)
    encrypted = bytearray()
    prev_block = iv

    for i in range(0, len(data), AES_BLOCK_SIZE):
        block = data[i:i + AES_BLOCK_SIZE]
        xored = bytes([b ^ p for b, p in zip(block, prev_block)])
        encrypted_block = bytes(aes_cipher.encrypt(xored))
        encrypted += encrypted_block
        prev_block = encrypted_block
    return bytes(encrypted)

def encrypt_file(input_path: Path, output_path: Path, key: str):
    try:
        key_bytes = prepare_key(key)
        iv = os.urandom(AES_BLOCK_SIZE)

        with open(input_path, 'rb') as f:
            plaintext = f.read()
        data = struct.pack('<Q', MAGIC_NUMBER) + plaintext
        padded = pkcs7_pad(data, AES_BLOCK_SIZE)

        ciphertext = encrypt_cbc(padded, key_bytes, iv)

        with open(output_path, 'wb') as f:
            f.write(iv + ciphertext)

        return True

    except Exception as e:
        print(f"ENCRYPTION FAILED: {e}")
        return False