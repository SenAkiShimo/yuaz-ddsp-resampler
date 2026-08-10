#!/usr/bin/env python3

import json

import os

import socket

import subprocess

import sys

import time

from pathlib import Path


ENGINE_VERSION = "0.2.7-alpha.8-rc.3.2"

DEFAULT_PORT = 47871


def project_root():

    return Path(__file__).resolve().parents[2]


def load_config(root):

    path = root / "config.json"

    if not path.exists():

        raise RuntimeError("Not configured. Run scripts/configure-macos.command first.")

    return json.loads(path.read_text(encoding="utf-8")), path


def send(host, port, payload, timeout=300):

    with socket.create_connection((host, port), timeout=timeout) as sock:

        sock.settimeout(timeout)

        stream = sock.makefile("rwb")

        stream.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

        stream.flush()

        line = stream.readline()

        if not line:

            raise RuntimeError("Engine closed the connection.")

        return json.loads(line.decode("utf-8"))


def ping(host, port):

    try:

        return send(host, port, {"action": "ping", "client_version": ENGINE_VERSION}, timeout=1)

    except Exception:

        return None


def start_server(root, config_path, host, port):

    lock = root / ".engine-start.lock"

    owner = False

    try:

        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        os.write(fd, f"{os.getpid()} {time.time()}".encode())

        os.close(fd)

        owner = True

    except FileExistsError:

        if lock.exists() and time.time() - lock.stat().st_mtime > 180:

            lock.unlink(missing_ok=True)

            return start_server(root, config_path, host, port)


    if owner:

        logs = root / "logs"

        logs.mkdir(exist_ok=True)

        log = open(logs / "engine.log", "ab", buffering=0)

        env = os.environ.copy()

        env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        subprocess.Popen(

            [sys.executable, "-m", "yuaz_ddsp_resampler.server", "--config", str(config_path)],

            cwd=str(root), stdout=log, stderr=log, start_new_session=True, env=env,

        )


    deadline = time.time() + 180

    while time.time() < deadline:

        status = ping(host, port)

        if status and status.get("ready") and status.get("engine_version") == ENGINE_VERSION:

            lock.unlink(missing_ok=True)

            return

        if status and status.get("ready") and status.get("engine_version") != ENGINE_VERSION:

            raise RuntimeError(f"Engine version mismatch: {status.get('engine_version')}")

        if status and status.get("error") and status.get("error") != "engine loading":

            raise RuntimeError(status["error"])

        time.sleep(0.25)

    raise RuntimeError("Engine startup timed out. See logs/engine.log.")


def parse_request(argv):

    if len(argv) < 5:

        raise RuntimeError("Expected classic UTAU resampler arguments.")

    args = list(argv) + [""] * 20

    return {

        "input": args[1], "output": args[2], "tone": args[3], "velocity": float(args[4] or 100),

        "flags": args[5] or "", "offset": float(args[6] or 0), "length": float(args[7] or 1000),

        "consonant": float(args[8] or 0), "cutoff": float(args[9] or 0), "volume": float(args[10] or 100),

        "modulation": float(args[11] or 0), "tempo": args[12] or "!120", "pitch": args[13] or "AA",

    }


def main():

    root = project_root()

    config, config_path = load_config(root)

    host = config.get("host", "127.0.0.1")

    port = int(config.get("port", DEFAULT_PORT))

    request = parse_request(sys.argv)

    status = ping(host, port)

    if not status or not status.get("ready") or status.get("engine_version") != ENGINE_VERSION:

        start_server(root, config_path, host, port)

    response = send(host, port, {"action": "render", "request": request}, timeout=600)

    if not response.get("ok"):

        raise RuntimeError(response.get("error", "Render failed."))


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(f"Yuaz DDSP Resampler: {exc}", file=sys.stderr)

        sys.exit(1)

