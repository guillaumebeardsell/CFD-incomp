"""Desktop launcher: serves the Flask app on localhost and opens a browser tab.

Used as the PyInstaller entry point. Running this script directly works too
(equivalent to `python -m app.app` but auto-opens the browser).
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _frozen_setup() -> None:
    if not getattr(sys, "frozen", False):
        return
    bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    user_dir = Path.home() / ".cfd-incomp"
    (user_dir / "numba_cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(user_dir / "numba_cache"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    _frozen_setup()
    from app.app import app  # imported after _frozen_setup so paths resolve

    port = int(os.environ.get("PORT", _free_port()))
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"CFD app running at {url}  (close this window to quit)", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
