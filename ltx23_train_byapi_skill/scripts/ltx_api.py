#!/usr/bin/env python3
"""LTX Trainer API CLI helper.

A thin wrapper around the LTX Trainer HTTP API so you don't have to memorise curl syntax.

Usage examples:
  python3 ltx_api.py status
  python3 ltx_api.py jobs
  python3 ltx_api.py download --csv ./signed.csv --categories "健身房镜头亲吻:80,对镜抖臀:60"
  python3 ltx_api.py train --name "GymKiss" --data-dir "健身房镜头亲吻" --caption "..."
  python3 ltx_api.py watch <job_id>
  python3 ltx_api.py log <job_id>
  python3 ltx_api.py samples <job_id>
  python3 ltx_api.py checkpoints <job_id>
  python3 ltx_api.py cancel <job_id>
  python3 ltx_api.py download-sample <job_id> <filename> [output_path]
  python3 ltx_api.py download-ckpt <job_id> <filename> [output_path]
  python3 ltx_api.py batch-train --config batch.json
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("LTX_API", "http://train_ltx23.dev.ad2.cc")

# ─── HTTP helpers ─────────────────────────────────────────────────────────────


def _get(path: str) -> dict:
    url = API_BASE + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[error] HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def _post_json(path: str, payload: dict) -> dict:
    url = API_BASE + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[error] HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def _delete(path: str) -> dict:
    url = API_BASE + path
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"[error] HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def _download_file(path: str, dest: Path) -> None:
    url = API_BASE + path
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def _post_multipart(path: str, fields: dict, files: dict) -> dict:
    """Multipart form-data POST (for CSV upload)."""
    import email.generator
    import io
    import mimetypes
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase

    boundary = "----LTXBoundary" + str(int(time.time()))
    body_parts = []

    for key, value in fields.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        )

    for key, (filename, file_bytes, content_type) in files.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        )
        body_parts_bytes = "".join(body_parts).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        body_parts = []
        body = body_parts_bytes

    if body_parts:
        body = "".join(body_parts).encode() + f"--{boundary}--\r\n".encode()

    url = API_BASE + path
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_resp = e.read().decode(errors="replace")
        print(f"[error] HTTP {e.code}: {body_resp}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


# ─── Display helpers ──────────────────────────────────────────────────────────

STATUS_ICON = {"pending": "⏳", "running": "▶", "done": "✅", "failed": "❌", "cancelled": "✖"}
PHASE_LABEL = {"precomputing": "预处理中", "training": "训练中", "done": "完成", "pending": "等待"}


def _fmt_job(j: dict) -> str:
    icon = STATUS_ICON.get(j["status"], "?")
    d = j.get("details", {})
    phase = PHASE_LABEL.get(d.get("phase", ""), d.get("phase", ""))
    progress = ""
    if j["status"] == "running" and d.get("phase") == "training" and d.get("total_steps"):
        cur = d.get("current_step", 0)
        total = d["total_steps"]
        pct = int(cur / total * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        loss = f"  Loss={d['loss']:.4f}" if d.get("loss") is not None else ""
        progress = f"\n     [{bar}] {cur}/{total} ({pct}%){loss}"
    qp = f"  队列#{j['queue_position']}" if j.get("queue_position", 0) > 0 else ""
    ts = j.get("created_at", "")[:16].replace("T", " ")
    line = f"{icon} [{j['job_id'][:8]}] {j['name']}  ({j['status']}"
    if phase and j["status"] == "running":
        line += f" · {phase}"
    line += f"){qp}  {ts}"
    if j.get("error"):
        line += f"\n     ⚠ {j['error']}"
    return line + progress


# ─── Commands ─────────────────────────────────────────────────────────────────


def cmd_status(_args: argparse.Namespace) -> None:
    try:
        result = _get("/api/jobs")
        running = sum(1 for j in result if j["status"] == "running")
        pending = sum(1 for j in result if j["status"] == "pending")
        print(f"✅ API 在线  |  总任务 {len(result)}  运行中 {running}  排队 {pending}")
        print(f"   地址: {API_BASE}")
    except SystemExit:
        print(f"❌ API 离线或无法访问: {API_BASE}")


def cmd_jobs(args: argparse.Namespace) -> None:
    jobs = _get("/api/jobs")
    if not jobs:
        print("暂无任务")
        return
    status_filter = getattr(args, "filter", None)
    for j in jobs:
        if status_filter and j["status"] != status_filter:
            continue
        print(_fmt_job(j))


def cmd_download(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV 文件不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Parse categories: "类别A:80,类别B:60" or JSON
    cats_raw = args.categories
    categories = []
    if cats_raw.strip().startswith("["):
        categories = json.loads(cats_raw)
    else:
        for item in cats_raw.split(","):
            item = item.strip()
            if ":" in item:
                name, limit = item.rsplit(":", 1)
                categories.append({"name": name.strip(), "limit": int(limit)})
            else:
                categories.append({"name": item, "limit": 80})

    print(f"📤 上传 {csv_path.name}，下载类别：")
    for c in categories:
        print(f"   • {c['name']}  最多 {c['limit']} 条")

    csv_bytes = csv_path.read_bytes()
    result = _post_multipart(
        "/api/download",
        fields={"categories": json.dumps(categories)},
        files={"csv_file": (csv_path.name, csv_bytes, "text/csv")},
    )
    print(f"✅ 下载任务已创建: {result['job_id']}")
    print(f"   用 'python3 ltx_api.py watch {result['job_id']}' 监控进度")


def cmd_resume_info(args: argparse.Namespace) -> None:
    d = _get(f"/api/train/{args.job_id}/resume-info")
    if not d.get("has_checkpoints"):
        print(f"暂无 checkpoint (job: {args.job_id})")
        return
    print(f"✅ 找到 {len(d['checkpoints'])} 个 checkpoint")
    print(f"   最新：{d['latest_checkpoint'].split('/')[-1]}")
    print(f"   已训练：{d['trained_steps']} / {d['original_steps']} 步")
    print(f"   建议续训步数：{d['remaining_steps']}")
    print("\n续训命令：")
    s = d["suggested_resume"]
    print(f'  python3 ltx_api.py train \\')
    print(f'    --name "{s["name"]}" \\')
    print(f'    --data-dir "{s["data_dir"]}" \\')
    print(f'    --caption "{s["caption"][:40]}..." \\')
    if s.get("trigger"):
        print(f'    --trigger "{s["trigger"]}" \\')
    print(f'    --steps {s["steps"]} \\')
    print(f'    --resume-from-job {args.job_id}')


def cmd_train(args: argparse.Namespace) -> None:
    payload = {
        "name": args.name,
        "data_dir": args.data_dir,
        "caption": args.caption,
        "trigger": args.trigger or "",
        "steps": args.steps,
        "rank": args.rank,
        "with_audio": not args.no_audio,
        "checkpoint_interval": args.checkpoint_interval,
        "validation_interval": args.validation_interval,
    }
    if args.validation_prompt:
        payload["validation_prompt"] = args.validation_prompt
    if getattr(args, "resume_from_job", None):
        payload["resume_from_job"] = args.resume_from_job
    elif getattr(args, "load_checkpoint", None):
        payload["load_checkpoint"] = args.load_checkpoint

    print("📋 训练参数：")
    for k, v in payload.items():
        print(f"   {k}: {v}")

    result = _post_json("/api/train", payload)
    print(f"\n✅ 训练任务已创建: {result['job_id']}  队列位置: #{result['queue_position']}")
    print(f"   用 'python3 ltx_api.py watch {result['job_id']}' 监控进度")


def cmd_batch_train(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)
    jobs_config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"📋 批量创建 {len(jobs_config)} 个训练任务...")
    for i, job in enumerate(jobs_config, 1):
        result = _post_json("/api/train", job)
        print(f"  [{i}/{len(jobs_config)}] ✅ {job.get('name')} → {result['job_id']} (队列#{result['queue_position']})")


def cmd_watch(args: argparse.Namespace) -> None:
    job_id = args.job_id
    print(f"👀 监控任务 {job_id}（Ctrl+C 退出）")
    last_step = -1
    try:
        while True:
            j = _get(f"/api/jobs/{job_id}")
            d = j.get("details", {})
            cur_step = d.get("current_step", 0)

            if cur_step != last_step or j["status"] in ("done", "failed", "cancelled"):
                ts = datetime.now_str()
                print(f"\r[{ts}] {_fmt_job(j)}", end="", flush=True)
                last_step = cur_step

            if j["status"] in ("done", "failed", "cancelled"):
                print()
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n已退出监控")


class datetime:
    @staticmethod
    def now_str() -> str:
        import datetime as _dt
        return _dt.datetime.now().strftime("%H:%M:%S")


def cmd_log(args: argparse.Namespace) -> None:
    tail = getattr(args, "tail", 100)
    result = _get(f"/api/jobs/{args.job_id}/log?tail={tail}")
    print(result.get("log", "（暂无日志）"))


def cmd_samples(args: argparse.Namespace) -> None:
    result = _get(f"/api/train/{args.job_id}/samples")
    samples = result.get("samples", [])
    if not samples:
        print("暂无采样视频")
        return
    print(f"📹 共 {len(samples)} 个采样视频：")
    for s in samples:
        step = s.replace("step_", "").replace("_1.mp4", "")
        url = f"{API_BASE}/api/train/{args.job_id}/samples/{urllib.parse.quote(s)}"
        print(f"  • {s}  →  {url}")


def cmd_checkpoints(args: argparse.Namespace) -> None:
    result = _get(f"/api/train/{args.job_id}/checkpoints")
    ckpts = result.get("checkpoints", [])
    if not ckpts:
        print("暂无 checkpoint")
        return
    print(f"💾 共 {len(ckpts)} 个 checkpoint：")
    for c in ckpts:
        url = f"{API_BASE}/api/train/{args.job_id}/checkpoints/{urllib.parse.quote(c['name'])}"
        print(f"  • {c['name']}  ({c['size_mb']} MB)  →  {url}")


def cmd_cancel(args: argparse.Namespace) -> None:
    result = _delete(f"/api/jobs/{args.job_id}")
    print(f"✖ 任务 {args.job_id} 已取消")


def cmd_download_sample(args: argparse.Namespace) -> None:
    dest = Path(args.output) if args.output else Path(args.filename)
    print(f"⬇ 下载采样视频 {args.filename} → {dest}")
    _download_file(f"/api/train/{args.job_id}/samples/{urllib.parse.quote(args.filename)}", dest)
    print(f"✅ 已保存到 {dest}")


def cmd_download_ckpt(args: argparse.Namespace) -> None:
    dest = Path(args.output) if args.output else Path(args.filename)
    print(f"⬇ 下载 checkpoint {args.filename} → {dest}")
    _download_file(f"/api/train/{args.job_id}/checkpoints/{urllib.parse.quote(args.filename)}", dest)
    print(f"✅ 已保存到 {dest}")


# ─── CLI Parser ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ltx_api", description="LTX Trainer API CLI")
    p.add_argument("--api", default=None, help=f"API base URL (default: {API_BASE})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="检查 API 服务是否在线")

    pj = sub.add_parser("jobs", help="查看任务列表")
    pj.add_argument("--filter", choices=["pending", "running", "done", "failed", "cancelled"], help="按状态筛选")

    pd = sub.add_parser("download", help="上传 CSV 并下载视频数据")
    pd.add_argument("--csv", required=True, help="CSV 文件路径")
    pd.add_argument("--categories", required=True, help='类别列表，格式："类别A:80,类别B:60" 或 JSON 数组')

    pt = sub.add_parser("train", help="创建训练任务")
    pt.add_argument("--name", required=True, help="任务名称")
    pt.add_argument("--data-dir", required=True, dest="data_dir", help="数据目录（类别名或路径）")
    pt.add_argument("--caption", required=True, help="训练描述/Caption")
    pt.add_argument("--trigger", default="", help="触发词（可选）")
    pt.add_argument("--steps", type=int, default=8000)
    pt.add_argument("--rank", type=int, default=32)
    pt.add_argument("--no-audio", action="store_true", help="禁用音频训练")
    pt.add_argument("--checkpoint-interval", type=int, default=500, dest="checkpoint_interval")
    pt.add_argument("--validation-interval", type=int, default=1000, dest="validation_interval")
    pt.add_argument("--validation-prompt", default=None, dest="validation_prompt")
    pt.add_argument("--resume-from-job", default=None, dest="resume_from_job",
                    help="从指定 job_id 的最新 checkpoint 续训")
    pt.add_argument("--load-checkpoint", default=None, dest="load_checkpoint",
                    help="直接指定 checkpoint 路径（.safetensors 或 checkpoints/ 目录）")

    pb = sub.add_parser("batch-train", help="从 JSON 配置批量创建训练任务")
    pb.add_argument("--config", required=True, help="批量配置 JSON 文件路径")

    pri = sub.add_parser("resume-info", help="查询某任务的续训信息（checkpoint 列表 + 建议参数）")
    pri.add_argument("job_id")

    pw = sub.add_parser("watch", help="持续监控任务进度")
    pw.add_argument("job_id", help="任务 ID")

    pl = sub.add_parser("log", help="查看任务日志")
    pl.add_argument("job_id")
    pl.add_argument("--tail", type=int, default=100)

    ps = sub.add_parser("samples", help="列出采样视频")
    ps.add_argument("job_id")

    pc = sub.add_parser("checkpoints", help="列出模型 checkpoint")
    pc.add_argument("job_id")

    pca = sub.add_parser("cancel", help="取消任务")
    pca.add_argument("job_id")

    pds = sub.add_parser("download-sample", help="下载采样视频")
    pds.add_argument("job_id")
    pds.add_argument("filename")
    pds.add_argument("output", nargs="?", default=None)

    pdc = sub.add_parser("download-ckpt", help="下载模型 checkpoint")
    pdc.add_argument("job_id")
    pdc.add_argument("filename")
    pdc.add_argument("output", nargs="?", default=None)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    global API_BASE
    if args.api:
        API_BASE = args.api.rstrip("/")

    commands = {
        "status": cmd_status,
        "jobs": cmd_jobs,
        "download": cmd_download,
        "train": cmd_train,
        "batch-train": cmd_batch_train,
        "resume-info": cmd_resume_info,
        "watch": cmd_watch,
        "log": cmd_log,
        "samples": cmd_samples,
        "checkpoints": cmd_checkpoints,
        "cancel": cmd_cancel,
        "download-sample": cmd_download_sample,
        "download-ckpt": cmd_download_ckpt,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
