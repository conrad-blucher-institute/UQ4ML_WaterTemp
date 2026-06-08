"""
Serve tuning visualizations with a local HTTP server.

Usage:
    python serve_timeseries_v2.py                    # GUI folder picker
    python serve_timeseries_v2.py --folder path/to/mape_results_run1_visuals

Starts a local server in the visuals folder and opens the explorer and
time series plots in the default browser. Other HTML plots are also listed.

Press Ctrl+C to stop the server.
"""

import argparse
import http.server
import glob
import sys
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from functools import partial

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS = str((_SCRIPT_DIR / '../../results').resolve())

# Plots to auto-open in the browser (in order)
_AUTO_OPEN = ['01_unified_plot_*.html', '05_timeseries.html']


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select visuals folder (containing HTML plots)",
        initialdir=_DEFAULT_RESULTS,
    )
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def find_html_files(folder):
    """Return all HTML files in the visuals folder, sorted by name."""
    return sorted(folder.glob('*.html'))


def main(folder=None, port=8765):
    parser = argparse.ArgumentParser(description="Serve tuning visualization HTML files")
    parser.add_argument('--folder', type=str, default=None,
                        help="Path to visuals folder (skip GUI picker)")
    parser.add_argument('--port', type=int, default=8765,
                        help="Port for local server (default: 8765)")
    args = parser.parse_args()

    if folder is None:
        folder = Path(args.folder) if args.folder else pick_folder()
        port = args.port
    else:
        folder = Path(folder)

    html_files = find_html_files(folder)
    if not html_files:
        print(f"No HTML files found in {folder}")
        sys.exit(1)

    # Serve from parent dir so relative paths to ../predictions/ work
    serve_root = str(folder.parent)
    folder_name = folder.name
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=serve_root)

    server = http.server.HTTPServer(('localhost', port), handler)

    # Determine which plots to auto-open
    auto_open_files = []
    for pattern in _AUTO_OPEN:
        matches = sorted(glob.glob(str(folder / pattern)))
        auto_open_files.extend(matches)

    # Build URLs
    auto_open_urls = [
        f'http://localhost:{port}/{folder_name}/{Path(f).name}'
        for f in auto_open_files
    ]

    print(f"Serving {folder}")
    print(f"\nAuto-opening in browser:")
    for url in auto_open_urls:
        print(f"  {url}")

    # List other available plots
    other_files = [f for f in html_files if str(f) not in auto_open_files]
    if other_files:
        print(f"\nOther plots available at:")
        for f in other_files:
            print(f"  http://localhost:{port}/{folder_name}/{f.name}")

    print("\nPress Ctrl+C to stop.\n")

    # Open browser tabs after a short delay
    def open_tabs():
        for url in auto_open_urls:
            webbrowser.open(url)

    threading.Timer(0.5, open_tabs).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == '__main__':
    main()
