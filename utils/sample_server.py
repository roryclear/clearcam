from http.server import BaseHTTPRequestHandler, HTTPServer
import uuid
import http.client

PUSHOVER_TOKEN = "PUT_YOUR_TOKEN_HERE"
PUSHOVER_USER = "PUT_YOUR_TOKEN_HERE"

class Handler(BaseHTTPRequestHandler):
  def do_POST(self):
    length = int(self.headers.get("Content-Length", 0))
    body = self.rfile.read(length)
    content_type = self.headers.get("Content-Type", "")
    boundary = content_type.split("boundary=", 1)[1].encode()
    parts = body.split(b"--" + boundary)

    text = None
    body_text = None
    img = None
    img_filename = None

    for part in parts:
      if not part or part in (b"--\r\n", b"--"): continue
      part = part.lstrip(b"\r\n")

      headers_end = part.find(b"\r\n\r\n")
      if headers_end == -1: continue

      raw_headers = part[:headers_end].decode("utf-8", errors="replace")
      data = part[headers_end + 4:]
      if data.endswith(b"\r\n"): data = data[:-2]

      if 'name="session_token"' in raw_headers: continue
      elif 'name="text"' in raw_headers: text = data.decode("utf-8")
      elif 'name="body_text"' in raw_headers: body_text = data.decode("utf-8")
      elif 'name="img"' in raw_headers:
        img = data
        marker = 'filename="'
        if marker in raw_headers: img_filename = raw_headers.split(marker, 1)[1].split('"', 1)[0]

    print("POST", self.path)
    print("text:", text)
    print("body_text:", body_text)
    print("img:", img_filename, len(img) if img else None)
    if body_text is not None: text += " " + body_text
    if img is not None:
      with open("received.jpg", "wb") as f: f.write(img)

    pushover_send(token=PUSHOVER_TOKEN, user=PUSHOVER_USER, message=text, image="received.jpg" if img is not None else None)

    self.send_response(200)
    self.end_headers()

def pushover_send(token, user, message, image=None):
  boundary = uuid.uuid4().hex
  parts = [
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="token"\r\n\r\n'
    f"{token}\r\n",
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="user"\r\n\r\n'
    f"{user}\r\n",
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="message"\r\n\r\n'
    f"{message}\r\n",
  ]
  body = "".join(parts).encode()
  if image is not None:
    with open(image, "rb") as f: image_bytes = f.read()
    body += (
      f"--{boundary}\r\n"
      'Content-Disposition: form-data; name="attachment"; filename="f40.jpg"\r\n'
      "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_bytes + b"\r\n"
  body += f"--{boundary}--\r\n".encode()
  conn = http.client.HTTPSConnection("api.pushover.net")
  conn.request(
      "POST",
      "/1/messages.json",
      body=body,
      headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
      },
  )
  response = conn.getresponse()
  print(response.read().decode())
  conn.close()

HTTPServer(("0.0.0.0", 8081), Handler).serve_forever()