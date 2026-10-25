"""Require actual viewer process exit within a separate cleanup deadline."""

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time


CLEANUP_MARKER = "Viewer cleanup starting"


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return result


def request_window_close(process):
    """Ask only this test child to close before terminating a failed test."""
    if process.poll() is not None or os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]

    @callback_type
    def visit(window, unused):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value == process.pid:
            title = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(window, title, len(title))
            if title.value.startswith("Caustra |"):
                user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE
        return True

    user32.EnumWindows(visit, 0)


def stop_failed_child(process):
    """The production viewer is unchanged; forcibly stop only a failed test."""
    if process.poll() is not None:
        return
    try:
        request_window_close(process)
    except OSError as error:
        print(f"Shutdown test: could not post close: {error}", file=sys.stderr)
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"Shutdown test: terminating timed-out test child PID {process.pid}", file=sys.stderr)
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(command, startup_timeout, exit_timeout):
    events = queue.Queue()
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def read_output():
        try:
            for line in process.stdout:
                events.put((time.monotonic(), line))
        finally:
            events.put((time.monotonic(), None))

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    cleanup_started = None
    output_closed = False
    validation_output = False
    failure = None

    def consume(event):
        nonlocal cleanup_started, output_closed, validation_output
        observed, line = event
        if line is None:
            output_closed = True
            return
        print(line, end="", flush=True)
        if line.startswith("[Vulkan validation]"):
            validation_output = True
        if line.rstrip("\r\n") == CLEANUP_MARKER and cleanup_started is None:
            cleanup_started = observed

    try:
        while True:
            try:
                consume(events.get(timeout=0.02))
                while not events.empty():
                    consume(events.get_nowait())
            except queue.Empty:
                pass
            now = time.monotonic()
            result = process.poll()
            if result is not None and output_closed:
                if cleanup_started is None:
                    failure = f"child exited without the '{CLEANUP_MARKER}' marker"
                elif result != 0:
                    failure = f"child exited with code {result}"
                elif validation_output:
                    failure = "Vulkan validation reported a message, including during cleanup"
                elif now - cleanup_started > exit_timeout:
                    failure = f"process exit exceeded the {exit_timeout:g} s cleanup budget"
                else:
                    print(f"Shutdown test passed: process exited with code 0 in {(now-cleanup_started)*1000:.3f} ms after cleanup began", flush=True)
                    return 0
                break
            if cleanup_started is not None:
                if now - cleanup_started > exit_timeout:
                    failure = f"process still running {exit_timeout:g} s after cleanup began"
                    break
            elif now - started > startup_timeout:
                failure = f"no cleanup marker within the {startup_timeout:g} s startup/run budget"
                break
        print(f"Shutdown test failed: {failure}", file=sys.stderr, flush=True)
        return 1
    finally:
        # A failed deadline must not leave hidden viewers behind or change the
        # production lifetime rules. This process handle belongs only to Popen.
        stop_failed_child(process)
        reader.join(timeout=1)
        while not events.empty():
            consume(events.get_nowait())
        if not reader.is_alive():
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-timeout", type=positive_seconds, default=300)
    parser.add_argument("--exit-timeout", type=positive_seconds, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a viewer command is required after --")
    try:
        return run(command, args.startup_timeout, args.exit_timeout)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Shutdown test failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
