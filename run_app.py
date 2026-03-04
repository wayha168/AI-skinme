"""
Launch Streamlit UI with styled console header.
Run: python run_app.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from console_style import print_header, print_section, print_key_value, print_line
    print_header("Skinme AI", "Skin Assistant — Ingredients & Product Recommendations")
    print_section("Command", "─")
    print_key_value("Mode", "Streamlit UI")
    print_key_value("URL", "http://localhost:8501 (after start)")
    print_line()
except ImportError:
    print("\n  Skinme AI — Skin Assistant (Streamlit)\n")

# Run Streamlit with app.py
import streamlit.web.cli as stcli
sys.argv = ["streamlit", "run", str(_root / "app.py"), "--server.headless", "true"]
sys.exit(stcli.main())
