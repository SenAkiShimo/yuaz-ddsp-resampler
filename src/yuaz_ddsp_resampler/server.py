#!/usr/bin/env python3
import argparse
import json
import os
import socketserver
import threading
import traceback
from pathlib import Path

from .core import YuazDDSPResamplerEngine
from .state import atomic_write_json

ENGINE_VERSION = "0.2.8ai.15"


class State:
    engine = None
    ready = False
    error = None
    runtime_id = None
    runtime_root = None
    active_renders = 0
    active_lock = threading.Lock()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline(16 * 1024 * 1024)
            request = json.loads(line.decode("utf-8"))
            action = request.get("action", "render")
            if action == "ping":
                response = {
                    "ok": True, "ready": State.ready, "error": State.error,
                    "engine_version": ENGINE_VERSION, "runtime_id": State.runtime_id,
                    "runtime_root": str(State.runtime_root), "pid": os.getpid(),
                    "active_renders": State.active_renders,
                }
            elif not State.ready:
                response = {"ok": False, "ready": False, "error": State.error or "engine loading"}
            elif request.get("runtime_id") and request.get("runtime_id") != State.runtime_id:
                response = {"ok": False, "error": "Runtime identity mismatch; refusing cross-version render."}
            elif action == "render":
                with State.active_lock:
                    State.active_renders += 1
                try:
                    response = State.engine.render(request["request"])
                    self._log_request(request["request"], response)
                finally:
                    with State.active_lock:
                        State.active_renders = max(0, State.active_renders - 1)
            elif action == "shutdown":
                response = {"ok": True}
                self.server.shutdown_requested = True
            else:
                response = {"ok": False, "error": f"Unknown action: {action}"}
        except Exception as exc:
            traceback.print_exc()
            response = {"ok": False, "error": str(exc)}
        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))

    def _log_request(self, request, response):
        try:
            root = State.runtime_root
            path = root / "logs" / "render_requests.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "request": request, "result": response, "engine_version": ENGINE_VERSION,
                    "runtime_id": State.runtime_id, "pid": os.getpid(),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.resolve()
    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", 47887))
    State.runtime_id = str(config.get("runtime_id") or ENGINE_VERSION)
    State.runtime_root = root
    pidfile = root / "engine.pid"
    atomic_write_json(pidfile, {
        "pid": os.getpid(), "engine_version": ENGINE_VERSION,
        "runtime_id": State.runtime_id, "runtime_root": str(root), "config": str(config_path),
    })
    try:
        with Server((host, port), Handler) as server:
            server.shutdown_requested = False
            try:
                State.engine = YuazDDSPResamplerEngine(
                    config["yuaz_repo"], config["checkpoint"],
                    transition_ms=config.get("transition_ms", 70),
                    use_rvq=config.get("use_rvq", False),
                    output_sr=config.get("output_sr", 44100),
                    registry_path=config.get("registry_path"),
                    ddsp_synthesis_sr=config.get("ddsp_synthesis_sr", 48000),
                    fullband_crossover_start_hz=config.get("ddsp_fullband_crossover_start_hz", 8800.0),
                    fullband_crossover_full_hz=config.get("ddsp_fullband_crossover_full_hz", 12100.0),
                    ai12_upperband_head_enabled=config.get("ai12_upperband_head_enabled", True),
                    ai12_upperband_head_start_hz=config.get("ai12_upperband_head_start_hz", 8400.0),
                    ai12_upperband_head_full_hz=config.get("ai12_upperband_head_full_hz", 12400.0),
                    ai13_upperband_guard_enabled=config.get("ai13_upperband_guard_enabled", True),
                    ai13_upperband_head_start_hz=config.get("ai13_upperband_head_start_hz", 8200.0),
                    ai13_upperband_head_full_hz=config.get("ai13_upperband_head_full_hz", 13800.0),
                )
                State.ready = True
                print(f"READY {ENGINE_VERSION} {State.runtime_id} {root}", flush=True)
            except Exception as exc:
                State.error = str(exc)
                print(f"LOAD ERROR: {exc}", flush=True)
            while not server.shutdown_requested:
                server.handle_request()
    finally:
        pidfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
