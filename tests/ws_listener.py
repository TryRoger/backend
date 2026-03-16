#!/usr/bin/env python3
"""
Live tail of ws_messages.log — shows all WebSocket messages in real time.

Usage:
    python tests/ws_listener.py              # tail from end (new messages only)
    python tests/ws_listener.py --all        # show entire log then follow
"""

import json
import sys
import time
import pathlib

LOG_FILE = pathlib.Path(__file__).resolve().parent.parent / "ws_messages.log"

COLORS = {
    "first_step":        "\033[36m",   # cyan
    "step_figure_agent": "\033[35m",   # magenta
    "roger":             "\033[32m",   # green
    "stepper":           "\033[33m",   # yellow
    "task_complete":     "\033[34m",   # blue
    "error":             "\033[31m",   # red
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

DIR_SYMBOLS = {"IN": "\033[33m>>>\033[0m", "OUT": "\033[36m<<<\033[0m"}


def format_line(line: str) -> str:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return line.rstrip()

    ts = data.pop("ts", "")
    direction = data.pop("dir", "?")
    tag = data.pop("tag", "unknown")

    arrow = DIR_SYMBOLS.get(direction, direction)
    color = COLORS.get(tag, "")

    # For delta messages, show just the text inline
    if data.get("type") == "delta":
        return f"{DIM}[{ts}]{RESET} {arrow} {color}[{tag}]{RESET} {data.get('text', '')}"

    preview = json.dumps(data, indent=2)
    if len(preview) > 600:
        preview = preview[:600] + "\n  ..."
    return f"{DIM}[{ts}]{RESET} {arrow} {color}{BOLD}[{tag}]{RESET}\n  {preview}"


def tail(show_all: bool = False):
    if not LOG_FILE.exists():
        print(f"{BOLD}Waiting for {LOG_FILE.name} to be created...{RESET}")
        while not LOG_FILE.exists():
            time.sleep(0.5)

    print(f"{BOLD}Tailing {LOG_FILE.name} (Ctrl+C to stop){RESET}\n")

    with open(LOG_FILE, "r") as f:
        if not show_all:
            f.seek(0, 2)  # seek to end

        while True:
            line = f.readline()
            if line:
                formatted = format_line(line)
                if formatted:
                    print(formatted)
            else:
                time.sleep(0.1)


if __name__ == "__main__":
    show_all = "--all" in sys.argv
    try:
        tail(show_all=show_all)
    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")
