"""Run child processes and stream stdout into a log file."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Pattern, TextIO

from wssed_server.settings import subprocess_env


def stream_process(
    command: List[str],
    *,
    cwd: Path,
    log_file: TextIO,
    env_extra: Optional[Dict[str, str]] = None,
    epoch_pattern: Optional[Pattern[str]] = None,
    on_epoch: Optional[Callable[[int, str], None]] = None,
) -> int:
    """Run *command*, mirror stdout to *log_file*, return exit code."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=subprocess_env(env_extra),
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        if epoch_pattern and on_epoch:
            match = epoch_pattern.search(line.strip())
            if match:
                on_epoch(int(match.group(1)), line.strip())
    return process.wait()


EPOCH_LINE_RE = re.compile(r"^Epoch\s+(\d+):")
