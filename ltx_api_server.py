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
import json
import logging
import os
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

# ─── Paths & Defaults ─────────────────────────────────────────────────────────

BASE_DIR = Path("/root/lisiyuan/LTX-2/packages/ltx-trainer")
DATASETS_DIR = BASE_DIR / "datasets" / "autotraindata"
JOBS_DIR = BASE_DIR / "autotrain_jobs"
CONFIGS_DIR = BASE_DIR / "configs"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_PATH = "/root/models/LTX-2.3/ltx-2.3-22b-dev.safetensors"
TEXT_ENCODER_PATH = "/root/lisiyuan/Models/gemma-3-12b-it-qat-q4_0-unquantized/"
RESOLUTION_BUCKETS = "544x960x241;960x544x241;544x960x257"

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
                self._jobs[job.job_id] = job
                if job.job_type == JobType.TRAIN and job.status == JobStatus.PENDING:
                    self._train_queue.append(job.job_id)
            except Exception as e:
                log.warning("Failed to load job %s: %s", f, e)

    def _persist(self, job: Job) -> None:
        (JOBS_DIR / f"{job.job_id}.json").write_text(job.model_dump_json(indent=2))

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_job(self, job_type: JobType, name: str, details: dict = {}) -> Job:
        job_id = uuid.uuid4().hex[:12]
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

            if self._active_train == job_id and self._active_proc:
                try:
                    os.killpg(os.getpgid(self._active_proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                self._active_proc = None
                self._active_train = None

            if job_id in self._train_queue:
                self._train_queue.remove(job_id)

            return True

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

            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    name = row.get("workflow_name", "")
                    if name not in cat_limit or counts[name] >= cat_limit[name]:
                        continue
                    try:
                        files = json.loads(row["files"])
                    except Exception:
                        continue
                    if not files:
                        continue
                    url = files[0]
                    row_id = row["id"]
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
                    step_re = re.compile(r"Step (\d+)/(\d+)")
                    loss_re = re.compile(r"Loss:\s*([\d.]+)")
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
                                lm = loss_re.search(line)
                                loss_val = float(lm.group(1)) if lm else None
                                det = {**self._jobs[job_id].details,
                                       "current_step": cur, "total_steps": total}
                                if loss_val is not None:
                                    det["loss"] = loss_val
                                self.update_job(job_id, details=det)
                threading.Thread(target=_tail_log, daemon=True, name=f"tail-{job_id}").start()

            ret = proc.wait()
            stop_tail.set()
            return ret

        # ── Step 1: Build dataset JSON ─────────────────────────────────────────
        data_dir = Path(d["data_dir"])
        caption = d["caption"]
        trigger = d.get("trigger", "")
        full_caption = f"{trigger}, {caption}".strip(", ") if trigger else caption

        videos = sorted(data_dir.glob("*.mp4")) + sorted(data_dir.glob("*.webm"))
        if not videos:
            raise ValueError(f"No videos in {data_dir}")

        dataset = [{"caption": full_caption, "media_path": str(v.relative_to(BASE_DIR))} for v in videos]
        json_path = BASE_DIR / f"autotrain_{job_id}.json"
        json_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))

        # ── Step 2: Precompute ─────────────────────────────────────────────────
        precomputed_dir = BASE_DIR / f".precomputed_autotrain_{job_id}"
        precompute_cmd = [
            "uv", "run", "python", "scripts/process_dataset.py",
            str(json_path),
            "--resolution-buckets", RESOLUTION_BUCKETS,
            "--output-dir", str(precomputed_dir),
            "--model-path", MODEL_PATH,
            "--text-encoder-path", TEXT_ENCODER_PATH,
        ]
        if trigger:
            precompute_cmd += ["--lora-trigger", trigger]
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
        val_prompt: str = d.get("validation_prompt") or full_caption
        load_checkpoint: Optional[str] = d.get("load_checkpoint")
        output_dir = OUTPUTS_DIR / f"autotrain_{job_id}"

        config_yaml = _build_config(
            precomputed_dir=str(precomputed_dir),
            output_dir=str(output_dir),
            steps=steps, rank=rank, with_audio=with_audio,
            ckpt_interval=ckpt_interval, val_interval=val_interval,
            val_prompt=val_prompt,
            load_checkpoint=load_checkpoint,
        )
        config_path = CONFIGS_DIR / f"autotrain_{job_id}.yaml"
        config_path.write_text(config_yaml)

        # ── Step 4: Train ──────────────────────────────────────────────────────
        train_cmd = ["uv", "run", "python", "scripts/train.py", str(config_path)]
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
) -> str:
    audio_str = "true" if with_audio else "false"
    escaped_prompt = val_prompt.replace('"', '\\"')
    ckpt_value = f'"{load_checkpoint}"' if load_checkpoint else "null"
    # When resuming, use constant LR to avoid restarting the warmup schedule
    scheduler = "constant" if load_checkpoint else "linear"
    return f"""model:
  model_path: "{MODEL_PATH}"
  text_encoder_path: "{TEXT_ENCODER_PATH}"
  training_mode: "lora"
  load_checkpoint: {ckpt_value}

lora:
  rank: {rank}
  alpha: {rank}
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.5
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
  quantization: null
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


class TrainRequest(BaseModel):
    name: str                            # Human-readable job name
    data_dir: str                        # Relative path under autotraindata/ OR absolute
    caption: str                         # Caption for all videos + validation prompt base
    trigger: str = ""                    # LoRA trigger word (prepended to caption)
    steps: int = DEFAULT_STEPS
    rank: int = DEFAULT_RANK
    with_audio: bool = DEFAULT_WITH_AUDIO
    validation_prompt: Optional[str] = None   # Override validation prompt
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    validation_interval: int = DEFAULT_VALIDATION_INTERVAL
    # Resume / continue training
    load_checkpoint: Optional[str] = None   # Path to .safetensors or checkpoint dir
    resume_from_job: Optional[str] = None   # job_id of a previous API job (auto-resolves latest ckpt)


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


@app.post("/api/train", tags=["train"])
def queue_train(req: TrainRequest):
    """Queue a LoRA training job. Multiple jobs are processed sequentially."""
    # Resolve data_dir
    data_dir = Path(req.data_dir)
    if not data_dir.is_absolute():
        for candidate in [DATASETS_DIR / req.data_dir, BASE_DIR / req.data_dir, Path(req.data_dir)]:
            if candidate.exists():
                data_dir = candidate
                break
        else:
            raise HTTPException(400, f"data_dir not found: {req.data_dir}")

    if not data_dir.is_dir():
        raise HTTPException(400, f"data_dir is not a directory: {data_dir}")

    # Resolve checkpoint path for resume
    resolved_ckpt: Optional[str] = None
    if req.resume_from_job:
        src_job = jm.get_job(req.resume_from_job)
        if not src_job:
            raise HTTPException(400, f"resume_from_job not found: {req.resume_from_job}")
        ckpt_dir = OUTPUTS_DIR / f"autotrain_{req.resume_from_job}" / "checkpoints"
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.safetensors")):
            raise HTTPException(400, f"No checkpoints found for job {req.resume_from_job}")
        resolved_ckpt = str(ckpt_dir)  # trainer will pick latest automatically
    elif req.load_checkpoint:
        resolved_ckpt = req.load_checkpoint

    details = {
        "data_dir": str(data_dir),
        "caption": req.caption,
        "trigger": req.trigger,
        "steps": req.steps,
        "rank": req.rank,
        "with_audio": req.with_audio,
        "validation_prompt": req.validation_prompt,
        "checkpoint_interval": req.checkpoint_interval,
        "validation_interval": req.validation_interval,
        "load_checkpoint": resolved_ckpt,
        "resume_from_job": req.resume_from_job,
        "phase": "pending",
    }
    job = jm.create_job(JobType.TRAIN, name=req.name, details=details)
    return {"job_id": job.job_id, "queue_position": jm.queue_position(job.job_id), "status": job.status}


@app.get("/api/jobs", tags=["jobs"])
def list_jobs():
    """List all jobs (newest first)."""
    return jm.list_jobs()


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


@app.get("/api/train/{job_id}/samples", tags=["results"])
def list_samples(job_id: str):
    """List validation sample videos generated during training."""
    if not jm.get_job(job_id):
        raise HTTPException(404, "Job not found")
    sample_dir = OUTPUTS_DIR / f"autotrain_{job_id}" / "samples"
    if not sample_dir.exists():
        return {"samples": [], "sample_dir": str(sample_dir)}
    files = sorted(sample_dir.glob("*.mp4"))
    return {"samples": [f.name for f in files], "count": len(files)}


@app.get("/api/train/{job_id}/samples/{filename}", tags=["results"])
def download_sample(job_id: str, filename: str):
    """Download a validation sample video."""
    path = OUTPUTS_DIR / f"autotrain_{job_id}" / "samples" / filename
    if not path.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.get("/api/train/{job_id}/checkpoints", tags=["results"])
def list_checkpoints(job_id: str):
    """List saved LoRA checkpoint files."""
    if not jm.get_job(job_id):
        raise HTTPException(404, "Job not found")
    ckpt_dir = OUTPUTS_DIR / f"autotrain_{job_id}" / "checkpoints"
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

    ckpt_dir = OUTPUTS_DIR / f"autotrain_{job_id}" / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("*.safetensors")) if ckpt_dir.exists() else []

    if not checkpoints:
        return {"job_id": job_id, "has_checkpoints": False, "checkpoints": []}

    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("step_")[1])
        except Exception:
            return -1

    ckpt_list = sorted(checkpoints, key=_step)
    latest = ckpt_list[-1]
    trained_steps = _step(latest)
    original_steps = job.details.get("steps", DEFAULT_STEPS)
    remaining = max(0, original_steps - trained_steps)

    return {
        "job_id": job_id,
        "has_checkpoints": True,
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
            "checkpoint_interval": job.details.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL),
            "validation_interval": job.details.get("validation_interval", DEFAULT_VALIDATION_INTERVAL),
            "resume_from_job": job_id,
        },
    }


@app.get("/api/train/{job_id}/checkpoints/{filename}", tags=["results"])
def download_checkpoint(job_id: str, filename: str):
    """Download a LoRA checkpoint (.safetensors)."""
    path = OUTPUTS_DIR / f"autotrain_{job_id}" / "checkpoints" / filename
    if not path.exists():
        raise HTTPException(404, "Checkpoint not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename=filename)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8777, log_level="info")
