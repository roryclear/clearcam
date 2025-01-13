import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8080

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/get-segments":
            self.handle_get_segments()
        else:
            super().do_GET()

    def handle_get_segments(self):
        try:
            # Find all .mp4 files in the current directory
            segments = [f for f in os.listdir('.') if f.endswith('.mp4')]
            segments.sort(key=lambda x: int(x.split('.')[0]))

            # Respond with JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(segments).encode())
        except Exception as e:
            # Handle errors and send a response
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())

if __name__ == "__main__":
    # Start the server
    server = HTTPServer(("0.0.0.0", PORT), CustomHandler)
    print(f"Serving on http://localhost:{PORT}")
    server.serve_forever()