import os
import re
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
            # Get all .mp4 files in the current directory
            files = [f for f in os.listdir('.') if f.endswith('.mp4')]

            # Sort files: 
            # 1. First by numeric value if filename is a simple number like '0.mp4'
            # 2. Then by the number in 'output_YYYY-MM-DD_00000_15/50/18.mp4'
            sorted_files = sorted(files, key=self.sort_key)

            # Respond with JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sorted_files).encode())
        except Exception as e:
            # Handle errors and send a response
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())

    def sort_key(self, filename):
        # If the filename is in the 'output_YYYY-MM-DD_00000_15/50/18.mp4' format
        match = re.match(r'output_\d{4}-\d{2}-\d{2}_(\d+)', filename)
        if match:
            # Extract the numeric value after 'output_YYYY-MM-DD_'
            return int(match.group(1))
        # If it's a simple '0.mp4' or similar, sort by the numeric part
        try:
            return int(filename.split('.')[0])
        except ValueError:
            return float('inf')  # For any other non-numeric names, send to the end

if __name__ == "__main__":
    # Start the server
    server = HTTPServer(("0.0.0.0", PORT), CustomHandler)
    print(f"Serving on http://localhost:{PORT}")
    server.serve_forever()
