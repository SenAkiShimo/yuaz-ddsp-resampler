#!/usr/bin/env python3
import hashlib
import json
import os
import platform
import socket
from importlib import metadata as importlib_metadata
import subprocess
import sys
import time
from pathlib import Path

ENGINE_VERSION = "0.2.8ai.13"
DEFAULT_PORT = 47885


def project_root():
    return Path(__file__).resolve().parents[2]


def validate_runtime_environment(root):
    marker = root / ".venv" / "RUNTIME_ENVIRONMENT.json"
    lock = root / "requirements.lock.txt"
    if not marker.is_file() or not lock.is_file():
        raise RuntimeError("Pinned runtime environment metadata is missing. Run setup-macos.command again.")
    data = json.loads(marker.read_text(encoding="utf-8"))
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    if data.get("requirements_hash") != lock_hash:
        raise RuntimeError("Runtime dependency lock changed. Run setup-macos.command again; refusing silent environment drift.")
    if data.get("python_version") != platform.python_version():
        raise RuntimeError(
            f"Python runtime changed from {data.get('python_version')} to {platform.python_version()}. "
            "Run setup-macos.command again; refusing silent acoustic runtime drift."
        )
    dist_names = {"torch":"torch", "numpy":"numpy", "librosa":"librosa", "soundfile":"soundfile", "pyyaml":"PyYAML"}
    expected = data.get("packages") or {}
    for key, dist in dist_names.items():
        if key not in expected:
            continue
        try:
            actual = importlib_metadata.version(dist).split('+')[0]
        except Exception as exc:
            raise RuntimeError(f"Pinned package {dist} is unavailable: {exc}") from exc
        if actual != expected[key]:
            raise RuntimeError(
                f"Runtime package drift detected for {dist}: {actual} != {expected[key]}. "
                "Run setup-macos.command again; refusing silent acoustic runtime drift."
            )


def load_config(root):
    path = root / "config.json"
    if not path.exists():
        raise RuntimeError("Not configured. Run scripts/configure-macos.command first.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("engine_version") != ENGINE_VERSION:
        raise RuntimeError(f"Config/version mismatch: {config.get('engine_version')} != {ENGINE_VERSION}")
    return config, path


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



def port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except Exception:
        return False


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _status_matches(status, runtime_id, root):
    return bool(
        status
        and status.get("ready")
        and status.get("engine_version") == ENGINE_VERSION
        and status.get("runtime_id") == runtime_id
        and Path(status.get("runtime_root", "")).expanduser().resolve() == root.resolve()
    )


def start_server(root, config_path, host, port, runtime_id):
    status = ping(host, port)
    if _status_matches(status, runtime_id, root):
        return
    if status and status.get("ready"):
        raise RuntimeError(
            "Port is occupied by a different Yuaz runtime: "
            f"version={status.get('engine_version')} runtime={status.get('runtime_id')} root={status.get('runtime_root')}"
        )
    if status is None and port_is_open(host, port):
        raise RuntimeError(
            f"Port {port} is already occupied by a non-0.2.8ai.13 service. Refusing to start on an ambiguous runtime port."
        )

    lock = root / ".engine-start.lock"
    owner = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {"pid": os.getpid(), "created_at": time.time(), "runtime_id": runtime_id}
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.close(fd)
            owner = True
            break
        except FileExistsError:
            stale = False
            try:
                data = json.loads(lock.read_text(encoding="utf-8"))
                stale = (not _pid_alive(data.get("pid"))) or (time.time() - lock.stat().st_mtime > 20)
            except Exception:
                stale = time.time() - lock.stat().st_mtime > 20
            status = ping(host, port)
            if _status_matches(status, runtime_id, root):
                return
            if stale:
                lock.unlink(missing_ok=True)
                continue
            # Another live client owns startup. Keep checking instead of giving up;
            # if that owner stalls, the lock becomes stale after 20 seconds and
            # this client can recover it.
            time.sleep(0.2)
            continue

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

    deadline = time.time() + 150
    try:
        while time.time() < deadline:
            status = ping(host, port)
            if _status_matches(status, runtime_id, root):
                return
            if status and status.get("ready"):
                raise RuntimeError(
                    "Engine identity mismatch during startup: "
                    f"version={status.get('engine_version')} runtime={status.get('runtime_id')} root={status.get('runtime_root')}"
                )
            if status and status.get("error") and status.get("error") != "engine loading":
                raise RuntimeError(status["error"])
            time.sleep(0.2)
        raise RuntimeError("Engine startup timed out. See logs/engine.log.")
    finally:
        if owner:
            lock.unlink(missing_ok=True)


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
    validate_runtime_environment(root)
    config, config_path = load_config(root)
    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", DEFAULT_PORT))
    runtime_id = str(config.get("runtime_id") or ENGINE_VERSION)
    request = parse_request(sys.argv)
    status = ping(host, port)
    if not _status_matches(status, runtime_id, root):
        start_server(root, config_path, host, port, runtime_id)
    response = send(host, port, {"action": "render", "request": request, "runtime_id": runtime_id}, timeout=600)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Render failed."))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Yuaz DDSP Resampler {ENGINE_VERSION}: {exc}", file=sys.stderr)
        sys.exit(1)
