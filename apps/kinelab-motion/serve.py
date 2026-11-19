"""Development server for KineLab Motion.

Binds to localhost by default: browsers only grant camera access on secure
origins (HTTPS or localhost), and binding every interface would publish the
app directory to the local network. Pass --host 0.0.0.0 to opt in.

Usage: python serve.py [port] [--host HOST]
"""
import argparse
import http.server
import os


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.mjs': 'text/javascript',
        '.js': 'text/javascript',
    }

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description='Serve apps/kinelab-motion with no-cache headers.')
    parser.add_argument('port', nargs='?', type=int, default=4173)
    parser.add_argument('--host', default='127.0.0.1', help='Interface to bind (default: 127.0.0.1)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer((args.host, args.port), NoCacheHandler) as httpd:
        shown = 'localhost' if args.host in ('127.0.0.1', 'localhost') else args.host
        print(f'Serving apps/kinelab-motion at http://{shown}:{args.port} (no-cache, multi-threaded)')
        httpd.serve_forever()


if __name__ == '__main__':
    main()
