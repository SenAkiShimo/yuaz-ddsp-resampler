#!/usr/bin/env python3
import argparse
import concurrent.futures
import fnmatch
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ID = "Bill13579/vocalset-mirror"
REVISION = "main"
USER_AGENT = "Yuaz-DDSP-Resampler/0.2.8ai.11-gender-training"
PRESETS = {
    "gender-core": ["data/*.parquet", "README.md"],
}



def human_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


def _request(url, *, headers=None, timeout=60):
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        h.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _next_link(header):
    if not header:
        return None
    for piece in header.split(","):
        piece = piece.strip()
        if 'rel="next"' not in piece:
            continue
        left = piece.find("<")
        right = piece.find(">", left + 1)
        if left >= 0 and right > left:
            return piece[left + 1:right]
    return None


def list_repo_files(endpoint, patterns, *, timeout=60):
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/api/datasets/{REPO_ID}/tree/{REVISION}?recursive=true&expand=false"
    selected = {}
    scanned = 0
    while url:
        with _request(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            next_raw = _next_link(response.headers.get("Link"))
            next_url = urllib.parse.urljoin(url, next_raw) if next_raw else None
        if not isinstance(payload, list):
            raise RuntimeError("Repository tree API returned an unexpected response.")
        for item in payload:
            scanned += 1
            if not isinstance(item, dict) or item.get("type") != "file":
                continue
            path = str(item.get("path") or "")
            if not path or not any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
                continue
            size = int(item.get("size") or 0)
            selected[path] = size
        print(f"\rListing repository tree... {scanned} entries scanned", end="", flush=True)
        url = next_url
    print()
    return [{"path": p, "size": selected[p]} for p in sorted(selected)]


def manifest_path(local_dir, preset):
    return local_dir / f".yuaz-vocalset-{preset}-manifest.json"


def load_or_build_manifest(local_dir, preset, endpoint, *, refresh=False):
    path = manifest_path(local_dir, preset)
    if not refresh and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("repo_id") == REPO_ID and data.get("revision") == REVISION and data.get("preset") == preset:
                files = data.get("files")
                if isinstance(files, list) and files:
                    print(f"Using cached file manifest: {path}")
                    return data
        except Exception:
            pass
    files = list_repo_files(endpoint, PRESETS[preset])
    if not files:
        raise RuntimeError(
            "No files matched this preset. The mirror may not expose the repository tree correctly, "
            "or the upstream dataset layout may have changed."
        )
    data = {
        "format": 1,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "preset": preset,
        "listed_from": endpoint,
        "created_at": time.time(),
        "files": files,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def local_progress(local_dir, files):
    complete = 0
    complete_bytes = 0
    partial_bytes = 0
    total = 0
    for item in files:
        rel = item["path"]
        size = int(item.get("size") or 0)
        total += size
        dest = local_dir / rel
        part = dest.with_name(dest.name + ".part")
        if dest.is_file() and (size <= 0 or dest.stat().st_size == size):
            complete += 1
            complete_bytes += size if size > 0 else dest.stat().st_size
            continue
        if part.is_file() and size > 0:
            partial_bytes += min(part.stat().st_size, size)
    remaining = max(0, total - complete_bytes - partial_bytes)
    return complete, total, complete_bytes, partial_bytes, remaining


def print_plan(local_dir, files):
    complete, total, complete_bytes, partial_bytes, remaining = local_progress(local_dir, files)
    print(f"Matched files: {len(files)}")
    print(f"Files already complete: {complete}")
    if partial_bytes:
        print(f"Reusable partial bytes: {human_bytes(partial_bytes)}")
    print(f"Remaining download: {human_bytes(remaining)}")
    print(f"Selected preset total: {human_bytes(total)}")
    return remaining


def _download_url(endpoint, rel):
    quoted = urllib.parse.quote(rel, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/{REVISION}/{quoted}?download=true"


def _download_one(endpoint, local_dir, item, *, progress_cb=None, stop_event=None, timeout=45, retries=12):
    rel = item["path"]
    expected = int(item.get("size") or 0)
    dest = local_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (expected <= 0 or dest.stat().st_size == expected):
        return rel, expected if expected > 0 else dest.stat().st_size, False
    part = dest.with_name(dest.name + ".part")
    url = _download_url(endpoint, rel)
    delay = 1.0
    last_error = None
    for attempt in range(1, retries + 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("cancelled")
        existing = part.stat().st_size if part.is_file() else 0
        if expected > 0 and existing > expected:
            part.unlink(missing_ok=True)
            existing = 0
        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                append = existing > 0 and status == 206
                if existing > 0 and not append:
                    # Endpoint ignored Range and is restarting the file. Remove the old
                    # partial byte count from the live progress tracker before overwrite.
                    if progress_cb is not None:
                        progress_cb(-existing, rel, force=True)
                    existing = 0
                mode = "ab" if append else "wb"
                with part.open(mode) as stream:
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            raise RuntimeError("cancelled")
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        if progress_cb is not None:
                            progress_cb(len(block), rel)
            got = part.stat().st_size
            if expected > 0 and got != expected:
                raise IOError(f"size mismatch after download: got {got}, expected {expected}")
            os.replace(part, dest)
            return rel, expected if expected > 0 else dest.stat().st_size, True
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                raise
            last_error = exc
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("cancelled")
        if attempt >= retries:
            break
        time.sleep(delay)
        delay = min(delay * 1.7, 15.0)
    raise RuntimeError(f"Failed after {retries} attempts: {rel}: {last_error}")


class LiveProgress:
    def __init__(self, initial_bytes, total_bytes, initial_files, total_files):
        self.lock = threading.Lock()
        self.bytes = int(initial_bytes)
        self.total_bytes = int(total_bytes)
        self.files = int(initial_files)
        self.total_files = int(total_files)
        self.start_bytes = int(initial_bytes)
        self.start = time.time()
        self.last_print = 0.0
        self.last_bytes = int(initial_bytes)
        self.last_time = self.start
        self.ema_speed = 0.0

    def _render(self, current_file="", force=False):
        now = time.time()
        if not force and now - self.last_print < 0.5:
            return
        dt = max(0.001, now - self.last_time)
        inst = max(0.0, (self.bytes - self.last_bytes) / dt)
        self.ema_speed = inst if self.ema_speed <= 0 else (0.25 * inst + 0.75 * self.ema_speed)
        self.last_bytes = self.bytes
        self.last_time = now
        self.last_print = now
        pct = (100.0 * self.bytes / self.total_bytes) if self.total_bytes > 0 else 0.0
        eta = ""
        if self.ema_speed > 1 and self.total_bytes > self.bytes:
            sec = int((self.total_bytes - self.bytes) / self.ema_speed)
            if sec < 3600:
                eta = f" | ETA {sec//60:02d}:{sec%60:02d}"
            else:
                eta = f" | ETA {sec//3600:d}:{(sec%3600)//60:02d}:{sec%60:02d}"
        label = Path(current_file).name if current_file else ""
        if len(label) > 28:
            label = "…" + label[-27:]
        suffix = f" | {label}" if label else ""
        print(
            f"\r[{pct:6.2f}%] {self.files}/{self.total_files} files | "
            f"{human_bytes(self.bytes)}/{human_bytes(self.total_bytes)} | "
            f"{human_bytes(self.ema_speed)}/s{eta}{suffix}   ",
            end="", flush=True,
        )

    def add_bytes(self, delta, current_file="", force=False):
        with self.lock:
            self.bytes = max(0, self.bytes + int(delta))
            self._render(current_file, force=force)

    def file_done(self, current_file=""):
        with self.lock:
            self.files += 1
            self._render(current_file, force=True)

    def finish(self):
        with self.lock:
            self._render(force=True)
            print()


def download_files(endpoint, local_dir, files, *, workers=4):
    pending = []
    complete, _, complete_bytes, partial_bytes, _ = local_progress(local_dir, files)
    total_bytes = sum(int(x.get("size") or 0) for x in files)
    for item in files:
        dest = local_dir / item["path"]
        expected = int(item.get("size") or 0)
        if not (dest.is_file() and (expected <= 0 or dest.stat().st_size == expected)):
            pending.append(item)
    if not pending:
        print("All selected files are already complete.")
        return
    workers = max(1, min(int(workers), len(pending), 8))
    print(f"Downloading {len(pending)} files with {workers} concurrent workers.")
    print("Progress is byte-based and refreshes while large files are still downloading.")
    print("Interrupted downloads stay as .part files; re-run with either route to resume them.")
    stop_event = threading.Event()
    progress = LiveProgress(complete_bytes + partial_bytes, total_bytes, complete, len(files))
    failures = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_map = {
        pool.submit(
            _download_one, endpoint, local_dir, item,
            progress_cb=progress.add_bytes, stop_event=stop_event,
        ): item for item in pending
    }
    interrupted = False
    try:
        for future in concurrent.futures.as_completed(future_map):
            item = future_map[future]
            try:
                _, _, _ = future.result()
                progress.file_done(item["path"])
            except RuntimeError as exc:
                if str(exc) == "cancelled" and stop_event.is_set():
                    continue
                failures.append((item["path"], str(exc)))
            except Exception as exc:
                failures.append((item["path"], str(exc)))
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        for f in future_map:
            f.cancel()
        print("\nStopping workers safely. Existing .part bytes are preserved.")
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
        progress.finish()
    if interrupted:
        print("Interrupted. Re-run the same setup command and choose either route to resume.")
        return False
    if failures:
        preview = "\n".join(f"  - {p}: {e}" for p, e in failures[:10])
        raise RuntimeError(
            f"{len(failures)} files failed to download. Re-run to resume. First failures:\n{preview}"
        )
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh-manifest", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint.rstrip("/")
    workers = max(1, min(int(args.workers), 12))
    print(f"Repository: {REPO_ID}")
    print(f"Preset: {args.preset}")
    print(f"Metadata/download endpoint: {endpoint}")
    print(f"Destination: {local_dir}")
    allowed_endpoints = {"https://hf-mirror.com", "https://huggingface.co"}
    if endpoint not in allowed_endpoints:
        raise RuntimeError(
            "Unsupported endpoint. Use https://hf-mirror.com or https://huggingface.co."
        )
    manifest = load_or_build_manifest(local_dir, args.preset, endpoint, refresh=args.refresh_manifest)
    files = manifest["files"]
    print_plan(local_dir, files)
    if args.dry_run:
        return
    completed = download_files(endpoint, local_dir, files, workers=workers)
    if completed is False:
        raise SystemExit(130)
    marker = local_dir / ".yuaz-vocalset-preset"
    marker.write_text(args.preset + "\n", encoding="utf-8")
    print(f"Download ready: {local_dir}")


if __name__ == "__main__":
    main()
