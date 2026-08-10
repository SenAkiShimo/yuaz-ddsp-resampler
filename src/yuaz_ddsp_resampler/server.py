#!/usr/bin/env python3

import argparse

import json

import socketserver

import traceback

from pathlib import Path


from .core import YuazDDSPResamplerEngine


ENGINE_VERSION = "0.2.7-alpha.8-rc.3.2"


class State:

    engine = None

    ready = False

    error = None


class Handler(socketserver.StreamRequestHandler):

    def handle(self):

        try:

            line = self.rfile.readline(16 * 1024 * 1024)

            request = json.loads(line.decode("utf-8"))

            action = request.get("action", "render")

            if action == "ping":

                response = {"ok": True, "ready": State.ready, "error": State.error, "engine_version": ENGINE_VERSION}

            elif not State.ready:

                response = {"ok": False, "ready": False, "error": State.error or "engine loading"}

            elif action == "render":

                response = State.engine.render(request["request"])

                self._log_request(request["request"], response)

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

            root = Path(__file__).resolve().parents[2]

            path = root / "logs" / "render_requests.jsonl"

            path.parent.mkdir(exist_ok=True)

            with path.open("a", encoding="utf-8") as stream:

                stream.write(json.dumps({"request": request, "result": response, "engine_version": ENGINE_VERSION}, ensure_ascii=False) + "\n")

        except Exception:

            pass


class Server(socketserver.ThreadingTCPServer):

    allow_reuse_address = True

    daemon_threads = True


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    host = config.get("host", "127.0.0.1")

    port = int(config.get("port", 47871))

    with Server((host, port), Handler) as server:

        server.shutdown_requested = False

        try:

            State.engine = YuazDDSPResamplerEngine(

                config["yuaz_repo"], config["checkpoint"],

                transition_ms=config.get("transition_ms", 70),

                use_rvq=config.get("use_rvq", False),

                output_sr=config.get("output_sr", 44100),

                registry_path=config.get("registry_path"),

            )

            State.ready = True

            print("READY", flush=True)

        except Exception as exc:

            State.error = str(exc)

            print(f"LOAD ERROR: {exc}", flush=True)

        while not server.shutdown_requested:

            server.handle_request()


if __name__ == "__main__":

    main()

