"""An in-place terminal panel, so the README's GIF depicts the real tool.

The panel is redrawn at a fixed height on every update: a progress or elapsed
row, the committed English, and the offline translation underneath once it
exists. Nothing already printed is edited, the frame is simply reprinted, which
is also honest about the policy -- the English text only ever grows.
"""

from __future__ import annotations

import os
import shutil
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[38;5;111m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;179m"
GREY = "\033[38;5;243m"


def supports_ansi() -> bool:
    """Redrawing needs a real terminal; piped output must stay append-only."""
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")


def wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


class Panel:
    """Fixed-height frame redrawn in place.

    Args:
        title: shown in bold on the first row.
        subtitle: the policy configuration, so a recording says what produced it.
        body_lines: how many rows the translation gets. Fixed, because a frame
            that changes height cannot be redrawn by moving the cursor up.
    """

    def __init__(self, title: str, subtitle: str, body_lines: int = 5):
        self.title = title
        self.subtitle = subtitle
        self.body_lines = body_lines
        self.width = min(shutil.get_terminal_size((88, 24)).columns, 88)
        self.height = body_lines + 7
        self.ansi = supports_ansi()
        self._drawn = False

    def _bar(self, fraction: float) -> str:
        inner = self.width - 22
        filled = int(inner * max(0.0, min(fraction, 1.0)))
        return f"{BLUE}{'━' * filled}{GREY}{'━' * (inner - filled)}{RESET}"

    def update(
        self, status: str, text: str, fraction: float | None = None, offline: str | None = None
    ) -> None:
        if not self.ansi:
            return
        if self._drawn:
            sys.stdout.write(f"\033[{self.height}A")

        rows = [
            f"{BOLD}{self.title}{RESET}",
            f"{DIM}{self.subtitle}{RESET}",
            "",
            f"{self._bar(fraction)} {status}"
            if fraction is not None
            else f"{GREY}{'─' * (self.width - 22)}{RESET} {status}",
            "",
        ]
        body = wrap(text, self.width - 2)[-self.body_lines :]
        rows += [f"  {line}" for line in body]
        rows += [""] * (self.body_lines - len(body))
        rows.append(
            f"  {GREEN}offline:{RESET} {offline[: self.width - 12]}"
            if offline
            else f"  {GREY}offline: waiting for the whole utterance{RESET}"
        )
        rows.append("")

        sys.stdout.write("".join(f"\033[2K{row}\n" for row in rows))
        sys.stdout.flush()
        self._drawn = True

    def log(self, message: str) -> None:
        """Print above the panel without disturbing it."""
        if self.ansi and self._drawn:
            sys.stdout.write(f"\033[{self.height}A\033[2K{message}\n")
            sys.stdout.write("\033[J")
            self._drawn = False
        else:
            print(message)

    def close(self, text: str, offline: str | None = None) -> None:
        """Leave the final state on screen as ordinary scrollback."""
        if not self.ansi:
            print(f"{BOLD}streaming{RESET} {text}")
            if offline:
                print(f"{BOLD}offline  {RESET} {offline}")
            return
        self.update(f"{YELLOW}done{RESET}", text, 1.0 if self._drawn else None, offline)
