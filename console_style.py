"""
Console styling for Skinme AI — header, categories, and colored output.
Uses ANSI codes (supported in Windows 10+ Terminal and PowerShell).
"""
import sys

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Colors (bright where useful for terminal readability)
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"
WHITE = "\033[37m"

# Themed colors for Skinme AI
ACCENT = CYAN
SUCCESS = GREEN
WARN = YELLOW
ERROR = RED
INFO = BLUE
MUTED = DIM


def _enable_ansi_windows():
    """Enable ANSI in Windows console if needed."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def print_header(app_name: str = "Skinme AI", tagline: str = "Skin Assistant - Ingredients & Product Recommendations"):
    """Print a styled application header with optional tagline."""
    _enable_ansi_windows()
    width = 56
    # ASCII-safe box drawing for Windows compatibility
    top = "+" + "=" * (width - 2) + "+"
    bot = "+" + "=" * (width - 2) + "+"
    name_centered = app_name.center(width - 2)
    tag_centered = tagline.center(width - 2) if tagline else ""
    print()
    print(ACCENT + BOLD + top + RESET)
    print(ACCENT + "|" + RESET + WHITE + BOLD + name_centered + RESET + ACCENT + "|" + RESET)
    if tag_centered:
        print(ACCENT + "|" + RESET + MUTED + tag_centered + RESET + ACCENT + "|" + RESET)
    print(ACCENT + bot + RESET)
    print()


def print_section(title: str, char: str = "-"):
    """Print a category/section title with a line underneath."""
    _enable_ansi_windows()
    width = 50
    line = char * width
    print(ACCENT + BOLD + f"  {title}" + RESET)
    print(MUTED + f"  {line}" + RESET)


def print_success(msg: str):
    """Print a success message in green."""
    _enable_ansi_windows()
    print(SUCCESS + "  [OK] " + msg + RESET)


def print_info(msg: str):
    """Print an info message in blue."""
    _enable_ansi_windows()
    print(INFO + "  (i) " + msg + RESET)


def print_warning(msg: str):
    """Print a warning message in yellow."""
    _enable_ansi_windows()
    print(WARN + "  (!) " + msg + RESET)


def print_error(msg: str):
    """Print an error message in red."""
    _enable_ansi_windows()
    print(ERROR + "  (x) " + msg + RESET)


def print_line(char: str = "-", length: int = 50):
    """Print a subtle horizontal line."""
    _enable_ansi_windows()
    print(MUTED + "  " + char * length + RESET)


def print_key_value(key: str, value: str, indent: int = 2):
    """Print a key-value pair with aligned styling."""
    _enable_ansi_windows()
    pad = " " * indent
    print(pad + MUTED + key + ": " + RESET + WHITE + value + RESET)


def print_bullet(msg: str, indent: int = 2):
    """Print a bullet point."""
    _enable_ansi_windows()
    pad = " " * indent
    print(pad + ACCENT + "* " + RESET + msg)
