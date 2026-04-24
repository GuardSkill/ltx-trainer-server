# LTX Trainer API 端点参考

**Base URL**: `http://train_ltx23.dev.ad2.cc`（本机：`http://localhost:8777`）

---

## 服务状态

### GET /api/health

服务健康检查，用于确认服务在线。

**Response:**
```json
{"status": "ok", "uptime_seconds": 3600, "active_train": "abc123", "queue_length": 2}
```

---

### GET /api/summary

返回任务数量统计和队列预计等待时间。

**Response:**
```json
{
  "counts": {
    "train": {"pending": 2, "running": 1, "done": 5},
    "download": {"done": 3}
  },
  "active_train_job": "abc123",
  "pending_train_count": 2,
  "queue_eta_minutes": 135.5
}
```

- `queue_eta_minutes`：当前运行任务剩余时间 + 所有排队任务预计时间之和（基于当前训练速度估算，可能为 null）

---

## 数据集管理 /api/datasets

### GET /api/datasets

列出 `autotraindata/` 下所有已下载数据集目录。

**Response:**
```json
{
  "datasets": [
    {"name": "健身房镜头亲吻", "path": "/root/.../autotraindata/健身房镜头亲吻", "file_count": 80},
    {"name": "对镜抖臀", "path": "/root/.../autotraindata/对镜抖臀", "file_count": 80}
  ]
}
```

---

### POST /api/datasets

创建新的空数据集目录。

**JSON body:**
```json
{"name": "新数据集名称"}
```

**Response:**
```json
{"name": "新数据集名称", "path": "...", "file_count": 0}
```

- 名称不能包含 `/`、`\`、`.` 开头，否则返回 400

---

### POST /api/datasets/merge

将多个来源数据集的视频文件合并（复制或移动）到目标数据集。

**JSON body:**
```json
{
  "sources": ["数据集A", "数据集B"],
  "target": "合并后数据集名",
  "move": false
}
```

- `move: false` = 复制（来源保留），`move: true` = 移动（来源文件删除，空目录自动清理）
- 目标不存在时自动创建
- 文件名冲突时自动加来源数据集名后缀

**Response:**
```json
{
  "target": "合并后数据集名",
  "copied": 120,
  "moved": 0,
  "skipped": 5,
  "file_count": 120
}
```

---

### POST /api/datasets/{name}/upload

上传视频文件到指定数据集目录（multipart）。

**Form data:**
- `files`: 一个或多个视频文件（支持 .mp4 .webm .mov .avi .mkv）

**Response:**
```json
{"name": "健身房镜头亲吻", "saved": 10, "skipped": 2, "file_count": 90}
```

- `skipped`：非视频文件或同名文件（自动重命名避免覆盖）

---

### DELETE /api/datasets/{name}

彻底删除数据集目录及其所有文件（不可撤销）。

**Response:**
```json
{"name": "健身房镜头亲吻", "deleted": true}
```

---

### GET /api/datasets/{name}/videos

列出数据集中所有视频文件的文件名和大小。

**Response:**
```json
{
  "name": "健身房镜头亲吻",
  "videos": [
    {"filename": "abc123.mp4", "size_mb": 12.4},
    {"filename": "def456.mp4", "size_mb": 8.7}
  ],
  "count": 80
}
```

---

### GET /api/datasets/{name}/videos/{filename}

流式返回视频文件（支持 Range 请求，可 seek）。可直接在浏览器或视频播放器中打开。

- Content-Type 根据扩展名自动设置（mp4 / webm / mov / avi / mkv）
- 可用于 agent 下载单个视频进行内容分析

---

## 下载 /api/download & CSV 分析

### POST /api/csv/analyze

分析上传的 CSV 文件，返回其中所有 `workflow_name` 的唯一值及条数。不执行任何下载。

**Form data:**
- `csv_file`: CSV 文件（multipart upload）

**Response:**
```json
{
  "categories": [
    {"name": "健身房镜头亲吻", "count": 234},
    {"name": "对镜抖臀", "count": 156}
  ],
  "total_rows": 390
}
```

`categories` 按条数降序排列。

---

### POST /api/download

上传 CSV 并按类别下载视频到 `autotraindata/`，返回下载 job。

**Form data:**
- `csv_file`: CSV 文件（multipart upload）
- `categories`: JSON 数组字符串 `[{"name":"类别名","limit":80}]`

**Response:**
```json
{"job_id": "abc123", "status": "pending", "categories": [...]}
```

---

## 任务管理 /api/jobs

### GET /api/jobs

返回任务列表（最新优先）。支持过滤。

**Query params:**
- `type`: `train` 或 `download`（不传则返回所有）
- `status`: `pending` / `running` / `done` / `failed` / `cancelled`（不传则返回所有）

**Example:**
```bash
# 只看运行中的训练任务
curl -s "http://train_ltx23.dev.ad2.cc/api/jobs?type=train&status=running"
```

**Response:** Job 对象数组（见下方 Job 结构）

---

### GET /api/jobs/{job_id}

获取单个任务详情。

**Response:**
```json
{
  "job_id": "xyz789",
  "job_type": "train",
  "status": "running",
  "name": "GymKiss LoRA",
  "created_at": "2026-04-22T10:00:00",
  "updated_at": "2026-04-22T10:30:00",
  "details": {
    "phase": "training",
    "current_step": 1500,
    "total_steps": 8000,
    "loss": 0.0312,
    "step_rate": 2.3,
    "eta_minutes": 28.0,
    "data_dir": "/root/.../健身房镜头亲吻",
    "caption": "...",
    "trigger": "GymKiss",
    "steps": 8000,
    "rank": 32,
    "with_audio": true
  },
  "pid": 12345,
  "queue_position": 0
}
```

**job_type:**
- `download` — 下载任务
- `train` — 训练任务

**status:**
- `pending` — 队列等待
- `running` — 执行中
- `done` — 完成
- `failed` — 失败（见 `error` 字段）
- `cancelled` — 已取消

**details.phase（训练任务）:** `pending` → `precomputing` → `training` → `done`

---

### GET /api/jobs/{job_id}/log?tail=100

获取任务日志最后 N 行。

**Response:**
```json
{"log": "Step 1500/8000 - Loss: 0.0312...", "total_lines": 2340}
```

---

### DELETE /api/jobs/{job_id}

取消 pending 或 running 状态的任务。

**Response:**
```json
{"job_id": "xyz789", "status": "cancelled"}
```

---

### DELETE /api/jobs/{job_id}/remove

彻底删除已完成/失败/取消的任务记录。不可撤销，不能对 pending/running 任务操作。

**Query params:**
- `delete_outputs=true`：同时删除 checkpoint 和采样视频（仅训练任务有效）

**注意**：无论 `delete_outputs` 是否为 true，`.precomputed_autotrain_{job_id}/` 目录**永远不会被删除**，以便其他任务通过 `reuse_precomputed_from_job` 复用。

**Response:**
```json
{"job_id": "abc123", "removed": true, "deleted_outputs": false}
```

- `deleted_outputs`：是否成功删除了输出目录

---

## 训练 /api/train

### POST /api/train

创建 LoRA 训练任务，加入队列顺序执行。

**JSON body:**
```json
{
  "name": "GymKiss LoRA",
  "data_dir": "健身房镜头亲吻",
  "caption": "在健身房镜头前，男女两人激情接吻，情绪热烈。",
  "trigger": "GymKiss",
  "steps": 8000,
  "rank": 32,
  "with_audio": true,
  "fp8_quant": false,
  "high_capacity": false,
  "checkpoint_interval": 500,
  "validation_interval": 1000,
  "validation_prompt": null,
  "resume_from_job": null,
  "load_checkpoint": null
}
```

- `data_dir`：类别名（在 `datasets/autotraindata/` 下查找）、相对路径或绝对路径
- `validation_prompt`：null 时自动用 `trigger + caption` 拼接
- `fp8_quant`：开启 fp8 量化，可减少约 30-40% 显存占用，训练速度略降，默认 false
- `high_capacity`：开启后额外训练视频 feed-forward 层（`ff.net.0.proj` / `ff.net.2`），LoRA 容量更大，速度稍慢，默认 false
  - 注：`audio_ff` 层由 `with_audio` 控制，开启音频训练时自动加入，与 `high_capacity` 无关
- `resume_from_job`：填上一个训练 job_id，自动使用其最新 checkpoint 续训
- `load_checkpoint`：手动指定 .safetensors 路径

**Response:**
```json
{"job_id": "xyz789", "queue_position": 1, "status": "pending"}
```

---

### POST /api/train/batch

一次性批量提交多个训练任务，按顺序加入队列。适合自动化场景。

**JSON body:** TrainRequest 数组（参数同 `POST /api/train`）

```json
[
  {"name": "任务A", "data_dir": "数据集A", "caption": "...", "trigger": "TrigA"},
  {"name": "任务B", "data_dir": "数据集B", "caption": "...", "trigger": "TrigB"}
]
```

**Response:**
```json
{
  "jobs": [
    {"name": "任务A", "job_id": "abc123", "queue_position": 1, "status": "pending"},
    {"name": "任务B", "job_id": "def456", "queue_position": 2, "status": "pending"}
  ],
  "submitted": 2,
  "failed": 0
}
```

- 某项校验失败时该项返回 `{"name": "...", "error": "..."}` 并跳过，其余正常提交

---

### GET /api/train/{job_id}/samples

列出训练中生成的采样视频。

**Response:**
```json
{"samples": ["step_001000_1.mp4", "step_002000_1.mp4"], "count": 2}
```

---

### GET /api/train/{job_id}/samples/{filename}

下载采样视频（mp4）。

---

### GET /api/train/{job_id}/checkpoints

列出保存的模型 checkpoint。

**Response:**
```json
{
  "checkpoints": [
    {"name": "lora_weights_step_00500.safetensors", "size_mb": 651.2}
  ],
  "count": 1
}
```

---

### GET /api/train/{job_id}/checkpoints/{filename}

下载模型文件（.safetensors）。

---

### GET /api/train/{job_id}/resume-info

获取续训所需信息（已训步数、剩余步数、最新 checkpoint 路径）。

**Response:**
```json
{
  "job_id": "xyz789",
  "has_checkpoints": true,
  "latest_checkpoint": "/root/.../checkpoints/lora_weights_step_04000.safetensors",
  "trained_steps": 4000,
  "original_steps": 8000,
  "remaining_steps": 4000,
  "checkpoints": [...],
  "suggested_resume": {
    "name": "GymKiss LoRA (续训 step4000→8000)",
    "data_dir": "...",
    "caption": "...",
    "trigger": "GymKiss",
    "steps": 4000,
    "rank": 32,
    "with_audio": true,
    "resume_from_job": "xyz789"
  }
}
```

---

## 训练输出目录结构

目录名格式：`autotrain_{job_id}_{任务名安全字符}`，方便在文件系统中直接辨认。

```
outputs/autotrain_{job_id}_{name}/
├── checkpoints/
│   ├── lora_weights_step_00500.safetensors
│   └── lora_weights_step_01000.safetensors
└── samples/
    ├── step_001000_1.mp4
    └── step_002000_1.mp4

.precomputed_autotrain_{job_id}_{name}/   ← VAE latents 缓存，永不自动删除
```

任务的实际路径通过 `GET /api/jobs/{job_id}` 的 `details.output_dir` 和 `details.precomputed_dir` 字段获取。

---

## 错误响应格式

```json
{"detail": "Dataset 'xxx' not found"}
```

HTTP 状态码：`400` 参数错误 / `404` 资源不存在
