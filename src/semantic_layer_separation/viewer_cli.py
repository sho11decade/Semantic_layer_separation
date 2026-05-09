from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("streamlit is not installed. Install dependencies and retry.")
        return 1

    app_path = Path(__file__).resolve().parent / "viewer" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    stcli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
