import os
import sys
import threading
import time
import webbrowser

# PyInstaller non trova automaticamente il bundle SSL di certifi — fix esplicito
if getattr(sys, 'frozen', False):
    _cacert = os.path.join(sys._MEIPASS, 'certifi', 'cacert.pem')
    if os.path.exists(_cacert):
        os.environ['SSL_CERT_FILE'] = _cacert
        os.environ['REQUESTS_CA_BUNDLE'] = _cacert
    else:
        # fallback: prova certifi standard
        try:
            import certifi as _certifi
            os.environ['SSL_CERT_FILE'] = _certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = _certifi.where()
        except Exception:
            pass

import uvicorn

from main import app  # noqa: F401

_BOX_AVVIO = """\
╔══════════════════════════════════════════════╗
║   Consigliere di Stazione v1.0               ║
║   di I6502TR                                 ║
╠══════════════════════════════════════════════╣
║                                              ║
║   Avvio in corso, attendere...               ║
║                                              ║
╚══════════════════════════════════════════════╝
"""

_BOX_PRONTO = """\
╔══════════════════════════════════════════════╗
║   Consigliere di Stazione v1.0               ║
║   di I6502TR                                 ║
╠══════════════════════════════════════════════╣
║                                              ║
║   Server in esecuzione.                      ║
║                                              ║
║   Il browser si apre automaticamente.        ║
║   Se non si apre, vai manualmente su:        ║
║      http://localhost:8080                   ║
║                                              ║
║   Premi INVIO per spegnere il programma.     ║
║                                              ║
╚══════════════════════════════════════════════╝
"""


def _run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    print(_BOX_AVVIO)

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    time.sleep(3)

    print(_BOX_PRONTO)
    webbrowser.open("http://localhost:8080")

    try:
        input()
    except KeyboardInterrupt:
        pass

    sys.exit(0)
