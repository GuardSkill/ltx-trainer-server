#!/usr/bin/env python3
"""LTX Trainer API Server.

Provides HTTP endpoints to:
  - Download video datasets from signed-URL CSVs
  - Queue and run LoRA training jobs
  - Browse/download sample videos and model checkpoints
  - Cancel running/queued jobs

Port: 8777  (proxied via FRP to train_ltx23.dev.ad2.cc)
"""

import csv
import io
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ltx_api")

_SERVER_START_TIME = time.time()

# ─── Paths & Defaults ─────────────────────────────────────────────────────────

BASE_DIR = Path("/root/lisiyuan/LTX-2/packages/ltx-trainer")
DATASETS_DIR = BASE_DIR / "datasets" / "autotraindata"
JOBS_DIR = BASE_DIR / "autotrain_jobs"
CONFIGS_DIR = BASE_DIR / "configs"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_PATH = "/root/models/LTX-2.3/ltx-2.3-22b-dev.safetensors"
TEXT_ENCODER_PATH = "/root/lisiyuan/Models/gemma-3-12b-it-qat-q4_0-unquantized/"
RESOLUTION_BUCKETS = "544x960x241;960x544x241;544x960x257"


def _auto_resolution_buckets(video_paths: list[Path]) -> str:
    """Sample up to 30 videos, find the 10th-percentile frame count,
    round down to the nearest valid 8n+1, and return a bucket string."""
    import subprocess
    sample = video_paths[:30]
    counts: list[int] = []
    for vp in sample:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", str(vp)],
                capture_output=True, text=True, timeout=15,
            )
            n = int(r.stdout.strip())
            if n > 0:
                counts.append(n)
        except Exception:
            pass

    if not counts:
        log.warning("Could not probe any videos; falling back to default buckets")
        return RESOLUTION_BUCKETS

    counts.sort()
    p10 = counts[max(0, len(counts) // 10)]   # 10th-percentile
    # round down to nearest 8n+1, minimum 25 frames (~1 s)
    n = max((p10 - 1) // 8, 3)
    frames = n * 8 + 1
    log.info("Auto bucket: sampled %d videos, p10=%d frames → using %d frames", len(counts), p10, frames)
    return f"544x960x{frames};960x544x{frames}"

DEFAULT_STEPS = 6000
DEFAULT_RANK = 32
DEFAULT_WITH_AUDIO = True
DEFAULT_CHECKPOINT_INTERVAL = 500
DEFAULT_VALIDATION_INTERVAL = 1000
DEFAULT_DOWNLOAD_LIMIT = 200

# ─── Models ───────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    DOWNLOAD = "download"
    TRAIN = "train"


class Job(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    name: str
    created_at: str
    updated_at: str
    details: dict[str, Any] = {}
    pid: Optional[int] = None
    error: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _sanitize_name(name: str, maxlen: int = 24) -> str:
    """Turn a job name into a safe filesystem suffix (keeps CJK, alphanumeric, underscore)."""
    safe = re.sub(r'[\s\-]+', '_', name.strip())
    safe = re.sub(r'[^\w\u4e00-\u9fff]', '', safe)
    return safe[:maxlen].strip('_')


def _job_output_dir(job: "Job") -> Path:
    """Return the output directory for a training job, using stored path when available."""
    stored = job.details.get("output_dir")
    return Path(stored) if stored else OUTPUTS_DIR / f"autotrain_{job.job_id}"


def _job_precomputed_dir(job: "Job") -> Path:
    """Return the precomputed latents directory for a training job, using stored path when available."""
    stored = job.details.get("precomputed_dir")
    return Path(stored) if stored else BASE_DIR / f".precomputed_autotrain_{job.job_id}"


def _kill_proc_group(proc: "subprocess.Popen", term_timeout: int = 5) -> None:
    """Send SIGTERM to the process group; always follow up with SIGKILL.

    The top-level process may be a thin launcher (e.g. `uv run`) that exits
    quickly while child training processes (pytorch workers) live on as
    orphans.  We save the pgid upfront and always send SIGKILL at the end so
    the entire group is guaranteed dead.
    """
    # Save pgid before the process might exit and become un-queryable
    pgid: Optional[int] = None
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pass

    # SIGTERM the whole process group first
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass

    # Wait briefly for graceful shutdown
    try:
        proc.wait(timeout=term_timeout)
    except (subprocess.TimeoutExpired, Exception):
        pass

    # Always SIGKILL the process group — the uv wrapper may have exited cleanly
    # while child pytorch workers are still running and holding GPU memory.
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


# ─── Job Manager ──────────────────────────────────────────────────────────────


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._train_queue: list[str] = []
        self._active_train: Optional[str] = None
        self._active_proc: Optional[subprocess.Popen] = None

        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

        self._load_persisted()
        threading.Thread(target=self._queue_loop, daemon=True, name="queue-loop").start()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_persisted(self) -> None:
        for f in JOBS_DIR.glob("*.json"):
            try:
                job = Job.model_validate_json(f.read_text())
                if job.status == JobStatus.RUNNING:
                    # Check whether the recorded process is still alive.
                    # If the PID is still running, adopt it; otherwise mark failed.
                    pid_alive = False
                    if job.pid:
                        try:
                            os.kill(job.pid, 0)  # signal 0 = existence check
                            pid_alive = True
                        except OSError:
                            pass
                    if pid_alive:
                        log.info("Job %s (%s) PID %d still alive — keeping as running", job.job_id, job.name, job.pid)
                        self._active_train = job.job_id
                        # Restart log-tail thread so progress tracking resumes
                        threading.Thread(
                            target=self._resume_tail,
                            args=(job.job_id,),
                            daemon=True,
                            name=f"tail-{job.job_id}",
                        ).start()
                    else:
                        # PID is dead.  Check if training completed before the server went down.
                        cur = job.details.get("current_step", 0)
                        total = job.details.get("total_steps", 0)
                        if total > 0 and cur >= total:
                            job.status = JobStatus.DONE
                            job.details = {**job.details, "phase": "done"}
                            log.info("Job %s (%s) PID dead but reached step %d/%d — marking DONE", job.job_id, job.name, cur, total)
                        else:
                            job.status = JobStatus.FAILED
                            job.error = "Server restarted while job was running"
                            log.info("Job %s (%s) PID dead at step %d/%d — marking FAILED", job.job_id, job.name, cur, total)
                        job.updated_at = datetime.now().isoformat()
                        self._persist(job)
                elif job.status in (JobStatus.CANCELLED, JobStatus.FAILED) and job.pid:
                    # Ensure stale processes from cancelled/failed jobs are cleaned up
                    # so they don't hold GPU memory when the server restarts.
                    try:
                        os.kill(job.pid, 0)
                        pid_alive = True
                    except OSError:
                        pid_alive = False
                    if pid_alive:
                        log.warning("Job %s (%s) is %s but PID %d still alive — killing", job.job_id, job.name, job.status, job.pid)
                        try:
                            pgid = os.getpgid(job.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except Exception:
                            try:
                                os.kill(job.pid, signal.SIGKILL)
                            except Exception:
                                pass
                self._jobs[job.job_id] = job
                if job.job_type == JobType.TRAIN and job.status == JobStatus.PENDING:
                    self._train_queue.append(job.job_id)
            except Exception as e:
                log.warning("Failed to load job %s: %s", f, e)

    def _resume_tail(self, job_id: str) -> None:
        """Re-attach log-tail progress tracking for a job whose process survived a server restart.

        Also detects when the adopted process exits and transitions the job to DONE or FAILED,
        then releases _active_train so the queue can proceed.
        """
        import re
        log_file = JOBS_DIR / f"{job_id}.log"
        step_re = re.compile(r"Step (\d+)/(\d+)")
        loss_re = re.compile(r"Loss:\s*([\d.]+)")
        from collections import deque
        window: deque = deque(maxlen=10)
        # Wait until the log file exists
        for _ in range(30):
            if log_file.exists():
                break
            time.sleep(1)
        if not log_file.exists():
            return

        # Track how long we've seen no new lines (to detect process exit)
        idle_seconds = 0

        with open(log_file, "r", errors="replace") as fh:
            fh.seek(0, 2)  # seek to end — only track new lines
            while True:
                job = self._jobs.get(job_id)
                if not job or job.status != JobStatus.RUNNING:
                    break
                line = fh.readline()
                if not line:
                    time.sleep(1)
                    idle_seconds += 1
                    # After 5 s of silence, check if the adopted PID is still alive.
                    # This catches the case where training finished while the server
                    # was down — no new log lines will ever arrive.
                    if idle_seconds >= 5:
                        idle_seconds = 0
                        pid = self._jobs[job_id].pid
                        pid_alive = False
                        if pid:
                            try:
                                os.kill(pid, 0)
                                pid_alive = True
                            except OSError:
                                pass
                        if not pid_alive:
                            # Process has exited.  Determine outcome from current progress.
                            det = self._jobs[job_id].details
                            cur = det.get("current_step", 0)
                            total = det.get("total_steps", 0)
                            if total > 0 and cur >= total:
                                # Reached the final step — treat as successful completion.
                                new_det = {**det, "phase": "done"}
                                self.update_job(job_id, status=JobStatus.DONE, details=new_det)
                                log.info("Job %s (adopted): process exited at step %d/%d — marking DONE", job_id, cur, total)
                            else:
                                self.update_job(
                                    job_id, status=JobStatus.FAILED,
                                    error=f"Process exited unexpectedly at step {cur}/{total}",
                                )
                                log.warning("Job %s (adopted): process exited at step %d/%d — marking FAILED", job_id, cur, total)
                            # Release the queue so pending jobs can start.
                            with self._lock:
                                if self._active_train == job_id:
                                    self._active_train = None
                            break
                    continue
                idle_seconds = 0  # reset idle counter whenever we get a line
                m = step_re.search(line)
                if m:
                    cur, total = int(m.group(1)), int(m.group(2))
                    now_ts = time.time()
                    window.append((now_ts, cur))
                    lm = loss_re.search(line)
                    loss_val = float(lm.group(1)) if lm else None
                    det = {**self._jobs[job_id].details, "current_step": cur, "total_steps": total}
                    if loss_val is not None:
                        det["loss"] = loss_val
                    if len(window) >= 2:
                        t0, s0 = window[0]; t1, s1 = window[-1]
                        elapsed = t1 - t0
                        if elapsed > 0 and s1 > s0:
                            rate = (s1 - s0) / elapsed * 60
                            det["step_rate"] = round(rate, 2)
                            det["eta_minutes"] = round((total - cur) / rate, 1) if rate > 0 else None
                    self.update_job(job_id, details=det)

    def _persist(self, job: Job) -> None:
        (JOBS_DIR / f"{job.job_id}.json").write_text(job.model_dump_json(indent=2))

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_job(self, job_type: JobType, name: str, details: dict = {}, job_id: Optional[str] = None) -> Job:
        job_id = job_id or uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            name=name,
            created_at=now,
            updated_at=now,
            details=details,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
            if job_type == JobType.TRAIN:
                self._train_queue.append(job_id)
        return job

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = datetime.now().isoformat()
            self._persist(job)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
                return False

            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now().isoformat()
            self._persist(job)

            if self._active_train == job_id:
                if self._active_proc:
                    # Normal path: process was started by this server instance
                    proc = self._active_proc
                    self._active_proc = None
                    threading.Thread(
                        target=self._kill_and_release,
                        args=(job_id, proc),
                        daemon=True,
                        name=f"kill-{job_id}",
                    ).start()
                elif job.pid:
                    # Adopted path: server restarted and re-adopted a live PID,
                    # but _active_proc was never set.  Kill by stored PID directly.
                    pid = job.pid
                    threading.Thread(
                        target=self._kill_pid_and_release,
                        args=(job_id, pid),
                        daemon=True,
                        name=f"kill-{job_id}",
                    ).start()

            if job_id in self._train_queue:
                self._train_queue.remove(job_id)

            return True

    def _kill_and_release(self, job_id: str, proc: "subprocess.Popen") -> None:
        """Kill the process group and only then release _active_train so the
        queue loop won't start the next job before GPU memory is freed."""
        _kill_proc_group(proc)
        with self._lock:
            if self._active_train == job_id:
                self._active_train = None
        log.info("Job %s killed and _active_train released", job_id)

    def _kill_pid_and_release(self, job_id: str, pid: int) -> None:
        """Kill a process group by PID (used when _active_proc is unavailable,
        e.g. after server restart adopted a live process)."""
        log.info("Job %s: killing adopted PID %d by pgid", job_id, pid)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(5)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        with self._lock:
            if self._active_train == job_id:
                self._active_train = None
        log.info("Job %s (PID %d) killed and _active_train released", job_id, pid)

    def queue_position(self, job_id: str) -> int:
        try:
            return self._train_queue.index(job_id) + 1
        except ValueError:
            return 0

    # ── Download ───────────────────────────────────────────────────────────────

    def start_download(self, job_id: str, csv_path: Path, categories: list[dict]) -> None:
        threading.Thread(
            target=self._do_download,
            args=(job_id, csv_path, categories),
            daemon=True,
            name=f"dl-{job_id}",
        ).start()

    def _do_download(self, job_id: str, csv_path: Path, categories: list[dict]) -> None:
        try:
            self.update_job(job_id, status=JobStatus.RUNNING)
            cat_limit: dict[str, int] = {c["name"]: c.get("limit", DEFAULT_DOWNLOAD_LIMIT) for c in categories}
            counts: dict[str, int] = {n: 0 for n in cat_limit}
            tasks: list[tuple[str, str, Path]] = []

            row_counter = 0
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    row_counter += 1
                    name = row.get("workflow_name", "").strip()
                    if name not in cat_limit or counts[name] >= cat_limit[name]:
                        continue
                    # Parse the files JSON array; skip row if column is missing or invalid
                    raw_files = row.get("files", "")
                    if not raw_files:
                        log.debug("Download job %s row %d: empty/missing 'files' column — skipped", job_id, row_counter)
                        continue
                    try:
                        files = json.loads(raw_files)
                    except Exception as exc:
                        log.warning("Download job %s row %d: cannot parse 'files' JSON (%s) — skipped", job_id, row_counter, exc)
                        continue
                    if not files:
                        continue
                    url = files[0]
                    # Use the 'id' column if present; fall back to a counter-based name
                    row_id = row.get("id") or f"row{row_counter}"
                    ext = os.path.splitext(url.split("?")[0])[1] or ".mp4"
                    dest = DATASETS_DIR / name / f"{row_id}{ext}"
                    tasks.append((name, url, dest))
                    counts[name] += 1

            done = failed = 0
            total = len(tasks)

            def _dl_one(args: tuple) -> bool:
                _, url, dest = args
                if dest.exists():
                    return True
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        dest.write_bytes(resp.read())
                    return True
                except Exception as e:
                    log.warning("Download failed %s: %s", dest.name, e)
                    return False

            with ThreadPoolExecutor(max_workers=8) as pool:
                for ok in pool.map(_dl_one, tasks):
                    if ok:
                        done += 1
                    else:
                        failed += 1
                    self.update_job(job_id, details={"total": total, "done": done, "failed": failed, "categories": counts})

            status = JobStatus.DONE if failed == 0 else JobStatus.FAILED
            self.update_job(job_id, status=status, details={"total": total, "done": done, "failed": failed, "categories": counts})
            log.info("Download job %s complete: %d ok, %d failed", job_id, done, failed)
        except Exception as e:
            self.update_job(job_id, status=JobStatus.FAILED, error=str(e))
            log.error("Download job %s failed: %s", job_id, e)
        finally:
            try:
                csv_path.unlink()
            except Exception:
                pass

    # ── Training Queue ─────────────────────────────────────────────────────────

    def _queue_loop(self) -> None:
        while True:
            time.sleep(3)
            job_id: Optional[str] = None
            with self._lock:
                if self._active_train or not self._train_queue:
                    continue
                # Find next non-cancelled job
                while self._train_queue:
                    candidate = self._train_queue[0]
                    job = self._jobs.get(candidate)
                    if job and job.status == JobStatus.PENDING:
                        job_id = candidate
                        self._train_queue.pop(0)
                        self._active_train = job_id
                        break
                    self._train_queue.pop(0)

            if job_id:
                try:
                    self._run_train(job_id)
                except Exception as e:
                    log.error("Train job %s crashed: %s", job_id, e)
                    self.update_job(job_id, status=JobStatus.FAILED, error=str(e))
                with self._lock:
                    self._active_train = None
                    self._active_proc = None

    def _run_train(self, job_id: str) -> None:
        job = self._jobs[job_id]
        d = job.details
        log_file = JOBS_DIR / f"{job_id}.log"

        def _run_subprocess(cmd: list[str], phase: str) -> int:
            self.update_job(job_id, details={**self._jobs[job_id].details, "phase": phase})
            with open(log_file, "a") as lf:
                lf.write(f"\n=== {phase.upper()} {datetime.now().isoformat()} ===\n")
                lf.flush()
                proc = subprocess.Popen(
                    cmd, cwd=BASE_DIR,
                    stdout=lf, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
            with self._lock:
                self._active_proc = proc
                if job_id in self._jobs:
                    self._jobs[job_id].pid = proc.pid
                    self._persist(self._jobs[job_id])

            # For training phase: parse step progress from log in background
            stop_tail = threading.Event()
            if phase == "training":
                def _tail_log() -> None:
                    import re
                    from collections import deque
                    step_re = re.compile(r"Step (\d+)/(\d+)")
                    loss_re = re.compile(r"Loss:\s*([\d.]+)")
                    # sliding window: (timestamp, step) pairs for rate calc
                    window: deque = deque(maxlen=10)
                    with open(log_file, "r", errors="replace") as f:
                        f.seek(0, 2)  # seek to end
                        while not stop_tail.is_set():
                            line = f.readline()
                            if not line:
                                time.sleep(1)
                                continue
                            m = step_re.search(line)
                            if m:
                                cur, total = int(m.group(1)), int(m.group(2))
                                now_ts = time.time()
                                window.append((now_ts, cur))
                                lm = loss_re.search(line)
                                loss_val = float(lm.group(1)) if lm else None
                                det = {**self._jobs[job_id].details,
                                       "current_step": cur, "total_steps": total}
                                if loss_val is not None:
                                    det["loss"] = loss_val
                                # Compute step_rate and eta_minutes from window
                                if len(window) >= 2:
                                    t0, s0 = window[0]
                                    t1, s1 = window[-1]
                                    elapsed = t1 - t0
                                    if elapsed > 0 and s1 > s0:
                                        rate = (s1 - s0) / elapsed * 60  # steps/min
                                        remaining = total - cur
                                        det["step_rate"] = round(rate, 2)
                                        det["eta_minutes"] = round(remaining / rate, 1) if rate > 0 else None
                                self.update_job(job_id, details=det)
                threading.Thread(target=_tail_log, daemon=True, name=f"tail-{job_id}").start()

            ret = proc.wait()
            stop_tail.set()
            return ret

        # ── Step 1 & 2: Build dataset JSON + Precompute (skip if reusing) ────────
        trigger = d.get("trigger", "")
        data_sources = d.get("data_sources")  # list of {data_dir, caption, trigger} or None

        # Compute validation prompt fallback from caption(s)
        if data_sources:
            first = data_sources[0]
            _t, _c = first.get("trigger", ""), first.get("caption", "")
            _val_fallback = f"{_t}, {_c}".strip(", ") if _t else _c
        else:
            caption = d["caption"]
            full_caption = f"{trigger}, {caption}".strip(", ") if trigger else caption
            _val_fallback = full_caption

        reuse_src = d.get("reuse_precomputed_from_job")
        if reuse_src:
            # Reuse precomputed latents from a previous job — skip precomputation entirely
            src_job = self._jobs.get(reuse_src)
            precomputed_dir = _job_precomputed_dir(src_job) if src_job else BASE_DIR / f".precomputed_autotrain_{reuse_src}"
            if not precomputed_dir.exists():
                raise RuntimeError(f"Precomputed dir not found: {precomputed_dir}")
            log.info("Reusing precomputed data from job %s: %s", reuse_src, precomputed_dir)
            self.update_job(job_id, status=JobStatus.RUNNING)
        else:
            # Build dataset entries (multi-source or single-source)
            if data_sources:
                dataset = []
                for src in data_sources:
                    src_dir = Path(src["data_dir"])
                    src_t = src.get("trigger", "")
                    src_c = src["caption"]
                    src_full = f"{src_t}, {src_c}".strip(", ") if src_t else src_c
                    src_videos = sorted(src_dir.glob("*.mp4")) + sorted(src_dir.glob("*.webm"))
                    for v in src_videos:
                        dataset.append({"caption": src_full, "media_path": str(v.relative_to(BASE_DIR))})
                if not dataset:
                    raise ValueError("No videos found in any data_sources")
                log.info("Multi-source dataset: %d entries from %d sources", len(dataset), len(data_sources))
            else:
                data_dir = Path(d["data_dir"])
                videos = sorted(data_dir.glob("*.mp4")) + sorted(data_dir.glob("*.webm"))
                if not videos:
                    raise ValueError(f"No videos in {data_dir}")
                dataset = [{"caption": full_caption, "media_path": str(v.relative_to(BASE_DIR))} for v in videos]

            json_path = BASE_DIR / f"autotrain_{job_id}.json"
            json_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))

            # Auto-detect bucket frame count from actual video lengths
            all_video_paths = [BASE_DIR / e["media_path"] for e in dataset]
            res_buckets = _auto_resolution_buckets(all_video_paths)

            # Use stored precomputed_dir from details (includes job name suffix for new jobs)
            precomputed_dir = _job_precomputed_dir(self._jobs[job_id])
            precompute_cmd = [
                "uv", "run", "python", "scripts/process_dataset.py",
                str(json_path),
                "--resolution-buckets", res_buckets,
                "--output-dir", str(precomputed_dir),
                "--model-path", MODEL_PATH,
                "--text-encoder-path", TEXT_ENCODER_PATH,
            ]
            # Trigger is already embedded in each caption in the JSON
            # (both single-source full_caption and multi-source src_full include
            # the per-source trigger).  Do NOT also pass --lora-trigger or the
            # trigger word will be prepended twice by process_captions.py.
            if d.get("with_audio", DEFAULT_WITH_AUDIO):
                precompute_cmd += ["--with-audio"]

            self.update_job(job_id, status=JobStatus.RUNNING)
            ret = _run_subprocess(precompute_cmd, "precomputing")

            if self._jobs.get(job_id) and self._jobs[job_id].status == JobStatus.CANCELLED:
                return
            if ret != 0:
                raise RuntimeError(f"process_dataset.py exited {ret}")

        # ── Step 3: Generate training config ───────────────────────────────────
        steps: int = d.get("steps", DEFAULT_STEPS)
        rank: int = d.get("rank", DEFAULT_RANK)
        with_audio: bool = d.get("with_audio", DEFAULT_WITH_AUDIO)
        ckpt_interval: int = d.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)
        val_interval: int = d.get("validation_interval", DEFAULT_VALIDATION_INTERVAL)
        val_prompt: str = d.get("validation_prompt") or _val_fallback
        load_checkpoint: Optional[str] = d.get("load_checkpoint")
        # Use stored output_dir from details (includes job name suffix for new jobs)
        output_dir = _job_output_dir(self._jobs[job_id])

        fp8_quant: bool = d.get("fp8_quant", False)
        high_capacity: bool = d.get("high_capacity", False)
        alpha: Optional[int] = d.get("alpha")
        config_yaml = _build_config(
            precomputed_dir=str(precomputed_dir),
            output_dir=str(output_dir),
            steps=steps, rank=rank, with_audio=with_audio,
            ckpt_interval=ckpt_interval, val_interval=val_interval,
            val_prompt=val_prompt,
            load_checkpoint=load_checkpoint,
            fp8_quant=fp8_quant,
            high_capacity=high_capacity,
            alpha=alpha,
        )
        config_path = CONFIGS_DIR / f"autotrain_{job_id}.yaml"
        config_path.write_text(config_yaml)

        # ── Step 4: Train ──────────────────────────────────────────────────────
        train_cmd = ["uv", "run", "python", "scripts/train.py", str(config_path), "--disable-progress-bars"]
        ret = _run_subprocess(train_cmd, "training")

        if self._jobs.get(job_id) and self._jobs[job_id].status == JobStatus.CANCELLED:
            return
        if ret != 0:
            raise RuntimeError(f"train.py exited {ret}")

        self.update_job(job_id, status=JobStatus.DONE, details={**self._jobs[job_id].details, "phase": "done"})
        log.info("Train job %s complete", job_id)


# ─── Config Builder ───────────────────────────────────────────────────────────


def _build_config(
    precomputed_dir: str, output_dir: str,
    steps: int, rank: int, with_audio: bool,
    ckpt_interval: int, val_interval: int, val_prompt: str,
    load_checkpoint: Optional[str] = None,
    fp8_quant: bool = False,
    high_capacity: bool = False,
    alpha: Optional[int] = None,
) -> str:
    effective_alpha = alpha if alpha is not None else rank
    audio_str = "true" if with_audio else "false"
    escaped_prompt = val_prompt.replace('"', '\\"')
    ckpt_value = f'"{load_checkpoint}"' if load_checkpoint else "null"
    # When resuming, use constant LR to avoid restarting the warmup schedule
    scheduler = "constant" if load_checkpoint else "linear"
    quantization_value = '"fp8-quanto"' if fp8_quant else "null"
    # audio_ff layers: trained whenever with_audio is enabled
    # video ff layers: only trained in high_capacity mode
    extra_modules = ""
    if with_audio:
        extra_modules += '\n    - "audio_ff.net.0.proj"\n    - "audio_ff.net.2"'
    if high_capacity:
        extra_modules += '\n    - "ff.net.0.proj"\n    - "ff.net.2"'
        if not with_audio:
            # high_capacity without audio: no audio_ff layers
            pass
    return f"""model:
  model_path: "{MODEL_PATH}"
  text_encoder_path: "{TEXT_ENCODER_PATH}"
  training_mode: "lora"
  load_checkpoint: {ckpt_value}

lora:
  rank: {rank}
  alpha: {effective_alpha}
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"{extra_modules}

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.75
  with_audio: {audio_str}
  audio_latents_dir: "audio_latents"

optimization:
  learning_rate: 1e-4
  steps: {steps}
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw"
  scheduler_type: "{scheduler}"
  scheduler_params: {{}}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: {quantization_value}
  load_text_encoder_in_8bit: false

data:
  preprocessed_data_root: "{precomputed_dir}"
  num_dataloader_workers: 4

validation:
  prompts:
    - "{escaped_prompt}"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [544, 960, 241]
  frame_rate: 25.0
  seed: 42
  inference_steps: 18
  interval: {val_interval}
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_av"
  generate_audio: {audio_str}
  skip_initial_validation: false

checkpoints:
  interval: {ckpt_interval}
  keep_last_n: -1
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {{}}

hub:
  push_to_hub: false
  hub_model_id: null

wandb:
  enabled: false
  project: "ltx-2-trainer"
  entity: null
  tags: ["ltx2", "lora", "autotrain"]
  log_validation_videos: true

seed: 42
output_dir: "{output_dir}"
"""


# ─── FastAPI App ──────────────────────────────────────────────────────────────

jm = JobManager()
app = FastAPI(title="LTX Trainer API", version="1.0.0", description=__doc__)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request schemas ────────────────────────────────────────────────────────────


class CategorySpec(BaseModel):
    name: str
    limit: int = DEFAULT_DOWNLOAD_LIMIT


class DataSource(BaseModel):
    data_dir: str                        # Relative path under autotraindata/ OR absolute
    caption: str                         # Caption for videos in this source
    trigger: str = ""                    # Per-source trigger word (prepended to caption)


class TrainRequest(BaseModel):
    name: str                            # Human-readable job name
    data_dir: str = ""                   # Relative path under autotraindata/ OR absolute (single-source)
    caption: str = ""                    # Caption for all videos (single-source)
    trigger: str = ""                    # LoRA trigger word (prepended to caption)
    data_sources: Optional[list[DataSource]] = None  # Multi-source: each dataset with its own caption
    steps: int = DEFAULT_STEPS
    rank: int = DEFAULT_RANK
    alpha: Optional[int] = None          # LoRA alpha; defaults to rank value if not set
    with_audio: bool = DEFAULT_WITH_AUDIO
    validation_prompt: Optional[str] = None   # Override validation prompt
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    validation_interval: int = DEFAULT_VALIDATION_INTERVAL
    fp8_quant: bool = False              # Enable fp8 quantization to reduce VRAM usage
    high_capacity: bool = False          # Also train feed-forward layers (more LoRA capacity, slower)
    # Resume / continue training from checkpoint
    load_checkpoint: Optional[str] = None   # Path to .safetensors or checkpoint dir
    resume_from_job: Optional[str] = None   # job_id of a previous API job (auto-resolves latest ckpt)
    # Reuse precomputed latents from a previous job (skips precompute step)
    reuse_precomputed_from_job: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/", tags=["ui"], include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api", tags=["info"])
def api_info():
    return {
        "service": "LTX Trainer API",
        "version": "1.0.0",
        "ui": "http://localhost:8777/",
        "docs": "http://localhost:8777/docs",
    }


@app.post("/api/download", tags=["download"])
async def download_videos(
    csv_file: UploadFile = File(..., description="Signed-URL CSV file"),
    categories: str = Form(..., description='JSON array: [{"name":"...", "limit":80}, ...]'),
):
    """Upload a CSV and download videos for specified categories."""
    try:
        cats = [CategorySpec(**c) if isinstance(c, dict) else c for c in json.loads(categories)]
    except Exception as e:
        raise HTTPException(400, f"Invalid categories JSON: {e}")

    tmp_csv = JOBS_DIR / f"upload_{uuid.uuid4().hex[:8]}.csv"
    tmp_csv.write_bytes(await csv_file.read())

    job = jm.create_job(
        JobType.DOWNLOAD,
        name=csv_file.filename or "download",
        details={"categories": [c.model_dump() for c in cats]},
    )
    jm.start_download(job.job_id, tmp_csv, [c.model_dump() for c in cats])
    return {"job_id": job.job_id, "status": job.status, "categories": [c.model_dump() for c in cats]}


def _resolve_data_dir(raw: str) -> Path:
    """Resolve a data_dir string (relative or absolute) to an existing absolute Path."""
    p = Path(raw)
    if p.is_absolute():
        if not p.is_dir():
            raise HTTPException(400, f"data_dir is not a directory: {p}")
        return p
    for candidate in [DATASETS_DIR / raw, BASE_DIR / raw, p]:
        if candidate.exists():
            if not candidate.is_dir():
                raise HTTPException(400, f"data_dir is not a directory: {candidate}")
            return candidate
    raise HTTPException(400, f"data_dir not found: {raw}")


@app.post("/api/train", tags=["train"])
def queue_train(req: TrainRequest):
    """Queue a LoRA training job. Multiple jobs are processed sequentially."""
    if not req.data_sources and not req.data_dir:
        raise HTTPException(400, "Either data_dir or data_sources must be provided")
    if not req.data_sources and not req.caption:
        raise HTTPException(400, "caption is required when using data_dir")

    # Resolve data source(s)
    resolved_sources: Optional[list[dict]] = None
    if req.data_sources:
        resolved_sources = []
        for src in req.data_sources:
            src_dir = _resolve_data_dir(src.data_dir)
            resolved_sources.append({"data_dir": str(src_dir), "caption": src.caption, "trigger": src.trigger})
    else:
        data_dir = _resolve_data_dir(req.data_dir)

    # Resolve checkpoint path for resume
    resolved_ckpt: Optional[str] = None
    if req.resume_from_job:
        src_job = jm.get_job(req.resume_from_job)
        if not src_job:
            raise HTTPException(400, f"resume_from_job not found: {req.resume_from_job}")
        ckpt_dir = _job_output_dir(src_job) / "checkpoints"
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.safetensors")):
            raise HTTPException(400, f"No checkpoints found for job {req.resume_from_job}")
        resolved_ckpt = str(ckpt_dir)  # trainer will pick latest automatically
    elif req.load_checkpoint:
        resolved_ckpt = req.load_checkpoint

    # Validate reuse_precomputed_from_job if provided
    if req.reuse_precomputed_from_job:
        src_job = jm.get_job(req.reuse_precomputed_from_job)
        if not src_job:
            raise HTTPException(400, f"Job {req.reuse_precomputed_from_job} not found")
        pre_dir = _job_precomputed_dir(src_job)
        if not pre_dir.exists():
            raise HTTPException(400, f"No precomputed data found for job {req.reuse_precomputed_from_job}")

    # Pre-generate job_id so we can embed dir paths (including name suffix) into details
    job_id = uuid.uuid4().hex[:12]
    safe = _sanitize_name(req.name)
    details = {
        "data_dir": str(data_dir) if not resolved_sources else "",
        "caption": req.caption,
        "trigger": req.trigger,
        "data_sources": resolved_sources,
        "steps": req.steps,
        "rank": req.rank,
        "alpha": req.alpha,
        "with_audio": req.with_audio,
        "fp8_quant": req.fp8_quant,
        "high_capacity": req.high_capacity,
        "validation_prompt": req.validation_prompt,
        "checkpoint_interval": req.checkpoint_interval,
        "validation_interval": req.validation_interval,
        "load_checkpoint": resolved_ckpt,
        "resume_from_job": req.resume_from_job,
        "reuse_precomputed_from_job": req.reuse_precomputed_from_job,
        "phase": "pending",
        "output_dir": str(OUTPUTS_DIR / f"autotrain_{job_id}_{safe}"),
        "precomputed_dir": str(BASE_DIR / f".precomputed_autotrain_{job_id}_{safe}"),
    }
    job = jm.create_job(JobType.TRAIN, name=req.name, details=details, job_id=job_id)
    return {"job_id": job.job_id, "queue_position": jm.queue_position(job.job_id), "status": job.status}


@app.post("/api/train/batch", tags=["train"])
def queue_train_batch(reqs: list[TrainRequest]):
    """Queue multiple LoRA training jobs at once. Returns list of job results in submission order."""
    results = []
    for req in reqs:
        try:
            if not req.data_sources and not req.data_dir:
                results.append({"name": req.name, "error": "Either data_dir or data_sources must be provided"})
                continue
            if not req.data_sources and not req.caption:
                results.append({"name": req.name, "error": "caption is required when using data_dir"})
                continue

            # Resolve data source(s)
            b_resolved_sources: Optional[list[dict]] = None
            if req.data_sources:
                b_resolved_sources = []
                for src in req.data_sources:
                    try:
                        src_dir = _resolve_data_dir(src.data_dir)
                    except HTTPException as he:
                        raise ValueError(he.detail)
                    b_resolved_sources.append({"data_dir": str(src_dir), "caption": src.caption, "trigger": src.trigger})
            else:
                try:
                    data_dir = _resolve_data_dir(req.data_dir)
                except HTTPException as he:
                    results.append({"name": req.name, "error": he.detail})
                    continue

            resolved_ckpt: Optional[str] = None
            if req.resume_from_job:
                src_job = jm.get_job(req.resume_from_job)
                if not src_job:
                    results.append({"name": req.name, "error": f"resume_from_job not found: {req.resume_from_job}"})
                    continue
                ckpt_dir = _job_output_dir(src_job) / "checkpoints"
                if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.safetensors")):
                    results.append({"name": req.name, "error": f"No checkpoints found for job {req.resume_from_job}"})
                    continue
                resolved_ckpt = str(ckpt_dir)
            elif req.load_checkpoint:
                resolved_ckpt = req.load_checkpoint

            if req.reuse_precomputed_from_job:
                rsrc_job = jm.get_job(req.reuse_precomputed_from_job)
                if not rsrc_job:
                    results.append({"name": req.name, "error": f"Job {req.reuse_precomputed_from_job} not found"})
                    continue
                pre_dir = _job_precomputed_dir(rsrc_job)
                if not pre_dir.exists():
                    results.append({"name": req.name, "error": f"No precomputed data found for job {req.reuse_precomputed_from_job}"})
                    continue

            b_job_id = uuid.uuid4().hex[:12]
            b_safe = _sanitize_name(req.name)
            details = {
                "data_dir": str(data_dir) if not b_resolved_sources else "",
                "caption": req.caption,
                "trigger": req.trigger,
                "data_sources": b_resolved_sources,
                "steps": req.steps,
                "rank": req.rank,
                "with_audio": req.with_audio,
                "fp8_quant": req.fp8_quant,
                "high_capacity": req.high_capacity,
                "validation_prompt": req.validation_prompt,
                "checkpoint_interval": req.checkpoint_interval,
                "validation_interval": req.validation_interval,
                "load_checkpoint": resolved_ckpt,
                "resume_from_job": req.resume_from_job,
                "reuse_precomputed_from_job": req.reuse_precomputed_from_job,
                "phase": "pending",
                "output_dir": str(OUTPUTS_DIR / f"autotrain_{b_job_id}_{b_safe}"),
                "precomputed_dir": str(BASE_DIR / f".precomputed_autotrain_{b_job_id}_{b_safe}"),
            }
            job = jm.create_job(JobType.TRAIN, name=req.name, details=details, job_id=b_job_id)
            results.append({"name": req.name, "job_id": job.job_id, "queue_position": jm.queue_position(job.job_id), "status": job.status})
        except Exception as e:
            results.append({"name": req.name, "error": str(e)})
    return {"jobs": results, "submitted": sum(1 for r in results if "job_id" in r), "failed": sum(1 for r in results if "error" in r)}


@app.get("/api/health", tags=["info"])
def health():
    """Service health check."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _SERVER_START_TIME),
        "active_train": jm._active_train,
        "queue_length": len(jm._train_queue),
    }


@app.get("/api/summary", tags=["info"])
def summary():
    """Return job counts by type/status and estimated queue wait time."""
    jobs = jm.list_jobs()
    counts: dict[str, dict[str, int]] = {"train": {}, "download": {}}
    for j in jobs:
        t = j.job_type.value
        s = j.status.value
        counts[t][s] = counts[t].get(s, 0) + 1

    # Estimate queue ETA: pending train jobs × average step time
    pending_train = [j for j in jobs if j.job_type == JobType.TRAIN and j.status == JobStatus.PENDING]
    active_train = next((j for j in jobs if j.job_type == JobType.TRAIN and j.status == JobStatus.RUNNING), None)

    queue_eta_minutes = None
    if active_train:
        det = active_train.details
        rate = det.get("step_rate")
        cur = det.get("current_step", 0)
        total = det.get("total_steps") or det.get("steps", DEFAULT_STEPS)
        if rate and rate > 0:
            active_remaining = (total - cur) / rate  # minutes
            pending_minutes = sum(j.details.get("steps", DEFAULT_STEPS) / rate for j in pending_train)
            queue_eta_minutes = round(active_remaining + pending_minutes, 1)

    return {
        "counts": counts,
        "active_train_job": jm._active_train,
        "pending_train_count": len(pending_train),
        "queue_eta_minutes": queue_eta_minutes,
    }


@app.get("/api/jobs", tags=["jobs"])
def list_jobs(type: Optional[str] = None, status: Optional[str] = None):
    """List all jobs (newest first). Optional filters: type=train|download, status=pending|running|done|failed|cancelled."""
    jobs = jm.list_jobs()
    if type:
        jobs = [j for j in jobs if j.job_type.value == type]
    if status:
        jobs = [j for j in jobs if j.status.value == status]
    return jobs


@app.get("/api/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    """Get a job's status and details."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    resp = job.model_dump()
    resp["queue_position"] = jm.queue_position(job_id)
    return resp


@app.get("/api/jobs/{job_id}/log", tags=["jobs"])
def get_log(job_id: str, tail: int = 200):
    """Tail the training/precompute log for a job."""
    if not jm.get_job(job_id):
        raise HTTPException(404, "Job not found")
    log_file = JOBS_DIR / f"{job_id}.log"
    if not log_file.exists():
        return {"log": ""}
    lines = log_file.read_text(errors="replace").splitlines()
    return {"log": "\n".join(lines[-tail:]), "total_lines": len(lines)}


@app.delete("/api/jobs/{job_id}", tags=["jobs"])
def cancel_job(job_id: str):
    """Cancel a pending or running job. Next queued job starts automatically."""
    ok = jm.cancel_job(job_id)
    if not ok:
        raise HTTPException(400, "Cannot cancel: job not found or already finished")
    return {"job_id": job_id, "status": "cancelled"}


@app.delete("/api/jobs/{job_id}/remove", tags=["jobs"])
def remove_job(job_id: str, delete_outputs: bool = False):
    """Permanently remove a finished/failed/cancelled job from history.

    For training jobs, set delete_outputs=true to also remove checkpoints and
    sample videos. Precomputed latent cache is NEVER deleted so that
    reuse_precomputed_from_job still works on other jobs.
    """
    import shutil

    with jm._lock:
        job = jm._jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            raise HTTPException(400, "Cannot remove an active job; cancel it first")
        del jm._jobs[job_id]
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
        log_file = JOBS_DIR / f"{job_id}.log"
        if log_file.exists():
            log_file.unlink()

    deleted_outputs = False
    if delete_outputs and job.job_type == JobType.TRAIN:
        output_dir = _job_output_dir(job)
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
            deleted_outputs = True
        # Also clean up the temp dataset JSON and config YAML for this job
        for tmp in [BASE_DIR / f"autotrain_{job_id}.json",
                    CONFIGS_DIR / f"autotrain_{job_id}.yaml"]:
            if tmp.exists():
                tmp.unlink()
        # NOTE: .precomputed_autotrain_{job_id}/ is intentionally NOT deleted
        # so that reuse_precomputed_from_job keeps working for other jobs.

    return {"job_id": job_id, "removed": True, "deleted_outputs": deleted_outputs}


@app.get("/api/train/{job_id}/samples", tags=["results"])
def list_samples(job_id: str):
    """List validation sample videos generated during training."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    sample_dir = _job_output_dir(job) / "samples"
    if not sample_dir.exists():
        return {"samples": [], "sample_dir": str(sample_dir)}
    files = sorted(sample_dir.glob("*.mp4"))
    return {"samples": [f.name for f in files], "count": len(files)}


@app.get("/api/train/{job_id}/samples/{filename}", tags=["results"])
def download_sample(job_id: str, filename: str):
    """Download a validation sample video."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = _job_output_dir(job) / "samples" / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.get("/api/train/{job_id}/checkpoints", tags=["results"])
def list_checkpoints(job_id: str):
    """List saved LoRA checkpoint files."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    ckpt_dir = _job_output_dir(job) / "checkpoints"
    if not ckpt_dir.exists():
        return {"checkpoints": [], "checkpoint_dir": str(ckpt_dir)}
    files = sorted(ckpt_dir.glob("*.safetensors"))
    return {
        "checkpoints": [{"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1)} for f in files],
        "count": len(files),
    }


@app.get("/api/train/{job_id}/resume-info", tags=["results"])
def resume_info(job_id: str):
    """Return info needed to resume training from this job's latest checkpoint."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.job_type != JobType.TRAIN:
        raise HTTPException(400, "Not a training job")

    ckpt_dir = _job_output_dir(job) / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("*.safetensors")) if ckpt_dir.exists() else []

    # Check for reusable precomputed data (own job's precomputed dir)
    pre_dir = _job_precomputed_dir(job)
    has_precomputed = pre_dir.exists() and any(pre_dir.iterdir())

    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("step_")[1])
        except Exception:
            return -1

    if not checkpoints:
        return {
            "job_id": job_id,
            "has_checkpoints": False,
            "checkpoints": [],
            "has_precomputed": has_precomputed,
            "suggested_restart": {
                "name": f"{job.name} (重新训练)",
                "data_dir": job.details.get("data_dir", ""),
                "caption": job.details.get("caption", ""),
                "trigger": job.details.get("trigger", ""),
                "steps": job.details.get("steps", DEFAULT_STEPS),
                "rank": job.details.get("rank", DEFAULT_RANK),
                "alpha": job.details.get("alpha"),
                "with_audio": job.details.get("with_audio", DEFAULT_WITH_AUDIO),
                "fp8_quant": job.details.get("fp8_quant", False),
                "high_capacity": job.details.get("high_capacity", False),
                "checkpoint_interval": job.details.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL),
                "validation_interval": job.details.get("validation_interval", DEFAULT_VALIDATION_INTERVAL),
                "reuse_precomputed_from_job": job_id if has_precomputed else None,
                "data_sources": job.details.get("data_sources") or None,
            } if has_precomputed else None,
        }

    ckpt_list = sorted(checkpoints, key=_step)
    latest = ckpt_list[-1]
    trained_steps = _step(latest)
    original_steps = job.details.get("steps", DEFAULT_STEPS)
    remaining = max(0, original_steps - trained_steps)

    return {
        "job_id": job_id,
        "has_checkpoints": True,
        "has_precomputed": has_precomputed,
        "latest_checkpoint": str(latest),
        "trained_steps": trained_steps,
        "original_steps": original_steps,
        "remaining_steps": remaining,
        "checkpoints": [{"name": c.name, "step": _step(c), "size_mb": round(c.stat().st_size / 1e6, 1)} for c in ckpt_list],
        "suggested_resume": {
            "name": f"{job.name} (续训 step{trained_steps}→{original_steps})",
            "data_dir": job.details.get("data_dir", ""),
            "caption": job.details.get("caption", ""),
            "trigger": job.details.get("trigger", ""),
            "steps": remaining,
            "rank": job.details.get("rank", DEFAULT_RANK),
            "with_audio": job.details.get("with_audio", DEFAULT_WITH_AUDIO),
            "fp8_quant": job.details.get("fp8_quant", False),
            "high_capacity": job.details.get("high_capacity", False),
            "checkpoint_interval": job.details.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL),
            "validation_interval": job.details.get("validation_interval", DEFAULT_VALIDATION_INTERVAL),
            "resume_from_job": job_id,
        },
    }


@app.get("/api/train/{job_id}/checkpoints/{filename}", tags=["results"])
def download_checkpoint(job_id: str, filename: str):
    """Download a LoRA checkpoint (.safetensors)."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = _job_output_dir(job) / "checkpoints" / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Checkpoint not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename=filename)


@app.get("/api/datasets", tags=["datasets"])
def list_datasets():
    """List downloaded dataset directories under autotraindata/."""
    if not DATASETS_DIR.exists():
        return {"datasets": []}
    datasets = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir():
            files = list(d.glob("*.mp4")) + list(d.glob("*.webm"))
            datasets.append({"name": d.name, "path": str(d), "file_count": len(files)})
    return {"datasets": datasets}


@app.post("/api/csv/analyze", tags=["datasets"])
async def analyze_csv(csv_file: UploadFile = File(..., description="CSV file to analyze")):
    """Analyze a CSV and return unique workflow_name values with counts."""
    content = (await csv_file.read()).decode("utf-8-sig")
    counts: dict[str, int] = {}
    for row in csv.DictReader(io.StringIO(content)):
        name = row.get("workflow_name", "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    categories = [{"name": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return {"categories": categories, "total_rows": sum(counts.values())}


def _safe_dataset_name(name: str) -> str:
    """Raise if name is unsafe (path traversal etc.), else return stripped name."""
    name = name.strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, f"Invalid dataset name: {name!r}")
    return name


class CreateDatasetRequest(BaseModel):
    name: str


class MergeDatasetRequest(BaseModel):
    sources: list[str]          # source dataset names (under autotraindata/)
    target: str                 # target dataset name (created if not exists)
    move: bool = False          # move files instead of copy


@app.post("/api/datasets", tags=["datasets"])
def create_dataset(req: CreateDatasetRequest):
    """Create a new empty dataset directory under autotraindata/."""
    name = _safe_dataset_name(req.name)
    target = DATASETS_DIR / name
    if target.exists():
        raise HTTPException(400, f"Dataset '{name}' already exists")
    target.mkdir(parents=True)
    return {"name": name, "path": str(target), "file_count": 0}


@app.post("/api/datasets/merge", tags=["datasets"])
def merge_datasets(req: MergeDatasetRequest):
    """Copy (or move) all video files from source datasets into the target dataset."""
    import shutil
    target_name = _safe_dataset_name(req.target)
    target_dir = DATASETS_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = copied = skipped = 0
    for src_name in req.sources:
        src_name = _safe_dataset_name(src_name)
        src_dir = DATASETS_DIR / src_name
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
                continue
            dest = target_dir / f.name
            # Avoid collision
            if dest.exists():
                stem, suf = f.stem, f.suffix
                dest = target_dir / f"{stem}_{src_name}{suf}"
            if dest.exists():
                skipped += 1
                continue
            if req.move:
                shutil.move(str(f), str(dest))
                moved += 1
            else:
                shutil.copy2(str(f), str(dest))
                copied += 1
        # Remove empty source dir after move
        if req.move and src_dir != target_dir:
            try:
                if not any(src_dir.iterdir()):
                    src_dir.rmdir()
            except Exception:
                pass

    files_now = list(target_dir.glob("*.mp4")) + list(target_dir.glob("*.webm"))
    return {
        "target": target_name,
        "copied": copied,
        "moved": moved,
        "skipped": skipped,
        "file_count": len(files_now),
    }


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


@app.post("/api/datasets/{name}/upload", tags=["datasets"])
async def upload_to_dataset(name: str, files: list[UploadFile] = File(...)):
    """Upload video files into an existing dataset directory."""
    name = _safe_dataset_name(name)
    target_dir = DATASETS_DIR / name
    if not target_dir.is_dir():
        raise HTTPException(404, f"Dataset '{name}' not found")

    saved = skipped = 0
    for uf in files:
        suffix = Path(uf.filename or "").suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            skipped += 1
            continue
        dest = target_dir / (Path(uf.filename).name)
        if dest.exists():
            # Avoid overwrite — add numeric suffix
            stem, suf = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = target_dir / f"{stem}_{i}{suf}"
                i += 1
        dest.write_bytes(await uf.read())
        saved += 1

    files_now = list(target_dir.glob("*.mp4")) + list(target_dir.glob("*.webm"))
    return {"name": name, "saved": saved, "skipped": skipped, "file_count": len(files_now)}


@app.delete("/api/datasets/{name}", tags=["datasets"])
def delete_dataset(name: str):
    """Delete a dataset directory and all its contents."""
    import shutil
    name = _safe_dataset_name(name)
    target = DATASETS_DIR / name
    if not target.exists():
        raise HTTPException(404, f"Dataset '{name}' not found")
    if not target.is_dir():
        raise HTTPException(400, "Not a directory")
    shutil.rmtree(str(target))
    return {"name": name, "deleted": True}


_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


@app.get("/api/datasets/{name}/videos", tags=["datasets"])
def list_dataset_videos(name: str):
    """List video files in a dataset directory."""
    name = _safe_dataset_name(name)
    target_dir = DATASETS_DIR / name
    if not target_dir.is_dir():
        raise HTTPException(404, f"Dataset '{name}' not found")
    files = sorted(
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_SUFFIXES
    )
    return {
        "name": name,
        "videos": [{"filename": f.name, "size_mb": round(f.stat().st_size / 1e6, 2)} for f in files],
        "count": len(files),
    }


@app.get("/api/datasets/{name}/videos/{filename}", tags=["datasets"])
def get_dataset_video(name: str, filename: str):
    """Stream a video file from a dataset (supports Range requests for seeking)."""
    name = _safe_dataset_name(name)
    filename = Path(filename).name  # strip any path traversal
    path = DATASETS_DIR / name / filename
    if not path.exists() or not path.is_file() or path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(404, "Video not found")
    media_type = _VIDEO_MEDIA_TYPES.get(path.suffix.lower(), "video/mp4")
    return FileResponse(str(path), media_type=media_type)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8777, log_level="info")
