from http.server import BaseHTTPRequestHandler, HTTPServer

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
    if img is not None: with open("received.jpg", "wb") as f: f.write(img)
    self.send_response(200)
    self.end_headers()

HTTPServer(("0.0.0.0", 8081), Handler).serve_forever()