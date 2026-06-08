"""
Serve the time series comparison HTML with a local HTTP server.

Usage:
    python serve_timeseries.py                    # GUI folder picker
    python serve_timeseries.py --folder path/to/mape_results_run1_visuals

Starts a local server in the visuals folder and opens 05_timeseries.html
in the default browser. The HTML fetches prediction CSVs on demand.

Press Ctrl+C to stop the server.
"""

import argparse
import http.server
import os
import sys
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from functools import partial

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = str((_SCRIPT_DIR / '../../results').resolve())


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select visuals folder (containing 05_timeseries.html)",
        initialdir=_DEFAULT_RESULTS,
    )
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def main():
    parser = argparse.ArgumentParser(description="Serve time series comparison HTML")
    parser.add_argument('--folder', type=str, default=None,
                        help="Path to visuals folder (skip GUI picker)")
    parser.add_argument('--port', type=int, default=8765,
                        help="Port for local server (default: 8765)")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
    else:
        folder = pick_folder()

    html_file = folder / '05_timeseries.html'
    if not html_file.exists():
        print(f"05_timeseries.html not found in {folder}")
        print("Run run_v13.py first to generate it.")
        sys.exit(1)

    port = args.port
    # Serve from parent dir so relative paths to ../predictions/ work
    serve_root = str(folder.parent)
    folder_name = folder.name
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=serve_root)

    server = http.server.HTTPServer(('localhost', port), handler)
    url = f'http://localhost:{port}/{folder_name}/05_timeseries.html'

    print(f"Serving {folder}")
    print(f"Opening {url}")
    print("Press Ctrl+C to stop.\n")

    # Open browser after a short delay
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == '__main__':
    main()
