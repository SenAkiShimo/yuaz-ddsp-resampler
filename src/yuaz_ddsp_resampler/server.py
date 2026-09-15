#!/usr/bin/env python3
import argparse
import json
import os
import socketserver
import threading
import traceback
from pathlib import Path

import numpy as np

from . import state as _state

_original_resolve_active_state = _state.resolve_active_state
_original_lookup_local_record = _state.lookup_local_record


def _resolve_active_state_readonly_ai14(bank, allow_legacy=True, verify=True):
    state, info = _original_resolve_active_state(bank, allow_legacy=allow_legacy, verify=verify)
    if state is None:
        return state, info
    info = dict(info or {})
    info["source"] = "0.2.8ai.14-readonly"
    info["read_only_fallback"] = True
    info["compatibility_source_version"] = "0.2.8ai.14"
    return state, info


def _lookup_local_record_runtime_compatible(input_path):
    try:
        return _original_lookup_local_record(input_path)
    except RuntimeError as exc:
        if "no valid pinned state can be resolved" in str(exc):
            return None
        raise


def _state_error_allows_base_fallback(exc):
    text = str(exc)
    markers = (
        "no valid pinned state can be resolved",
        "no ai.14 base-checkpoint provenance",
        "trained against a different Yuaz base checkpoint",
        "Pinned learned-control pack is missing",
        "Unsupported AI control model format",
        "AI control model has incompatible controls",
        "Error(s) in loading state_dict",
        "size mismatch",
    )
    return any(marker in text for marker in markers)


_state.resolve_active_state = _resolve_active_state_readonly_ai14
_state.lookup_local_record = _lookup_local_record_runtime_compatible

from . import core as _core
from .controls import parse_yuaz_controls
from .core import YuazDDSPResamplerEngine
from .neural_runtime import NeuralWaveformRuntimeRoute
from .state import atomic_write_json

ENGINE_VERSION = "0.3.0"


class State:
    engine = None
    neural_route = None
    ready = False
    error = None
    runtime_id = None
    runtime_root = None
    active_renders = 0
    active_lock = threading.Lock()
    refiner_ab = threading.local()
    articulation_ab = threading.local()


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
                    "neural_waveform": State.neural_route.describe() if State.neural_route else {"loaded": False},
                }
            elif not State.ready:
                response = {"ok": False, "ready": False, "error": State.error or "engine loading"}
            elif request.get("runtime_id") and request.get("runtime_id") != State.runtime_id:
                response = {"ok": False, "error": "Runtime identity mismatch; refusing cross-version render."}
            elif action == "render":
                controls = parse_yuaz_controls(request["request"].get("flags", ""))
                State.refiner_ab.requested = bool(controls.refiner_bypass_enabled)
                State.refiner_ab.available = False
                State.refiner_ab.bypassed = False
                State.articulation_ab.requested = bool(controls.articulation_trajectory_bypass_enabled)
                State.articulation_ab.bypassed = False
                with State.active_lock:
                    State.active_renders += 1
                try:
                    response = State.engine.render(request["request"])
                    if State.neural_route is not None and isinstance(response, dict):
                        response.update(State.neural_route.stats())
                    if isinstance(response, dict):
                        response.update({
                            "fidelity_refiner_bypass_requested": bool(getattr(State.refiner_ab, "requested", False)),
                            "fidelity_refiner_available": bool(getattr(State.refiner_ab, "available", False)),
                            "fidelity_refiner_bypassed": bool(getattr(State.refiner_ab, "bypassed", False)),
                            "articulation_trajectory_bypass_requested": bool(getattr(State.articulation_ab, "requested", False)),
                            "articulation_trajectory_bypassed": bool(getattr(State.articulation_ab, "bypassed", False)),
                        })
                    self._log_request(request["request"], response)
                finally:
                    with State.active_lock:
                        State.active_renders = max(0, State.active_renders - 1)
                    State.refiner_ab.requested = False
                    State.refiner_ab.available = False
                    State.refiner_ab.bypassed = False
                    State.articulation_ab.requested = False
                    State.articulation_ab.bypassed = False
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
    port = int(config.get("port", 47889))
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
                State.neural_route = NeuralWaveformRuntimeRoute(root, config, device=State.engine.device)
                State.neural_route.install_patch(_core)

                original_articulation_hybrid_mix = _core.articulation_hybrid_mix

                def articulation_hybrid_mix_runtime_ab(
                    original, generated, sr, source_f0, target_f0, regions,
                    source_fixed_ms, target_fixed_ms, target_ms, canonical_template=None,
                ):
                    requested = bool(getattr(State.articulation_ab, "requested", False))
                    if requested:
                        # Zero trajectory preserves v1 timing/raw-prefix behavior while
                        # removing canonical/local spectral trajectory shaping.
                        canonical_template = {
                            "trajectory": np.zeros((129, 32), dtype=np.float32),
                            "energy_delta": np.zeros(32, dtype=np.float32),
                            "n_fft": 256,
                            "frames": 32,
                            "coherence": 1.0,
                            "source_count": 0,
                        }
                        State.articulation_ab.bypassed = True
                    mixed, stats = original_articulation_hybrid_mix(
                        original, generated, sr, source_f0, target_f0, regions,
                        source_fixed_ms, target_fixed_ms, target_ms,
                        canonical_template=canonical_template,
                    )
                    if requested and isinstance(stats, dict):
                        stats = dict(stats)
                        stats["trajectory_transfer_used"] = False
                        stats["trajectory_source"] = "bypassed-zero-template"
                        stats["trajectory_gain_rms_db"] = 0.0
                        stats["trajectory_strength"] = 0.0
                    return mixed, stats

                _core.articulation_hybrid_mix = articulation_hybrid_mix_runtime_ab

                original_models_for_input = State.engine._models_for_input

                def models_for_input_runtime_compatible(path):
                    try:
                        result = original_models_for_input(path)
                    except RuntimeError as exc:
                        if _state_error_allows_base_fallback(exc):
                            result = (None, None, [], None)
                        else:
                            raise
                    neural_active = State.neural_route.select_record(result[3])
                    requested = bool(getattr(State.refiner_ab, "requested", False))
                    refiner_available = bool(result[1] is not None)
                    bypassed = bool(requested and neural_active and refiner_available)
                    State.refiner_ab.available = refiner_available
                    State.refiner_ab.bypassed = bypassed
                    if bypassed:
                        return result[0], None, result[2], result[3]
                    return result

                State.engine._models_for_input = models_for_input_runtime_compatible
                State.ready = True
                neural_info = State.neural_route.describe()
                print(
                    f"READY {ENGINE_VERSION} {State.runtime_id} {root} "
                    f"refiner_ab=YQ1 articulation_trajectory_ab=YA1 "
                    f"neural_loaded={neural_info.get('loaded')} "
                    f"neural_checkpoint={neural_info.get('checkpoint')}",
                    flush=True,
                )
            except Exception as exc:
                State.error = str(exc)
                print(f"LOAD ERROR: {exc}", flush=True)
            while not server.shutdown_requested:
                server.handle_request()
    finally:
        pidfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
