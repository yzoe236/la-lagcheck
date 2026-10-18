"""CLI output should remain UTF-8 under legacy console encodings."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from lagcheck import cli


@pytest.mark.parametrize(
    ("args", "stream_name", "expected_text", "expected_code"),
    [
        (["run", "--help"], "stdout", "µs s⁻¹", 0),
        (
            ["run", "unused.json", "--output", "unused-out", "--reference-rate-ppm", "µ"],
            "stderr",
            "µ",
            2,
        ),
    ],
)
def test_cli_reconfigures_standard_streams_to_utf8(
    args: list[str], stream_name: str, expected_text: str, expected_code: int
) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "ascii:strict"
    env["PYTHONUTF8"] = "0"

    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-B", "-m", "lagcheck", *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
    )

    output = getattr(result, stream_name).decode("utf-8")
    assert result.returncode == expected_code, output
    assert expected_text in output


def test_cli_preserves_standard_stream_error_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stream:
        def __init__(self, errors: str) -> None:
            self.errors = errors
            self.options: dict[str, str] | None = None

        def reconfigure(self, **options: str) -> None:
            self.options = options

    stdout = Stream("surrogateescape")
    stderr = Stream("backslashreplace")
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    cli._configure_utf8_stdio()

    assert stdout.options == {"encoding": "utf-8", "errors": "surrogateescape"}
    assert stderr.options == {"encoding": "utf-8", "errors": "backslashreplace"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX argv can contain undecodable bytes")
def test_cli_unrecognized_surrogate_after_subcommand_keeps_argparse_error_exit() -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "ascii:backslashreplace"
    env["PYTHONUTF8"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8=0",
            "-B",
            "-m",
            "lagcheck",
            "run",
            "config.json",
            "--output",
            "output.csv",
            "\udcff",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
    )

    output = result.stderr.decode("utf-8")
    assert result.returncode == 2, output
    assert "unrecognized arguments" in output
    assert r"\udcff" in output
