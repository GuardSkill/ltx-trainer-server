# LTX Trainer API 端点参考

**Base URL**: `http://train_ltx23.dev.ad2.cc`（本机：`http://localhost:8777`）

---

## POST /api/download

上传 CSV 并按类别下载视频。

**Form data:**
- `csv_file`: CSV 文件（multipart upload）
- `categories`: JSON 数组字符串 `[{"name":"类别名","limit":80}]`

**Response:**
```json
{"job_id": "abc123", "status": "pending", "categories": [...]}
```

下载的视频保存到：`/root/lisiyuan/LTX-2/packages/ltx-trainer/datasets/autotraindata/<类别名>/`

---

## POST /api/train

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
  "checkpoint_interval": 500,
  "validation_interval": 1000,
  "validation_prompt": null
}
```

- `data_dir`：类别名（在 `datasets/autotraindata/` 下查找）、相对路径或绝对路径
- `trigger`：可选，留空字符串 `""` 表示不使用触发词
- `validation_prompt`：可选，若为 null 则使用 `trigger + caption`

**Response:**
```json
{"job_id": "xyz789", "queue_position": 1, "status": "pending"}
```

---

## GET /api/jobs

返回所有任务列表（最新优先）。

**Response:** Job 对象数组

---

## GET /api/jobs/{job_id}

获取单个任务详情。

**Response:**
```json
{
  "job_id": "xyz789",
  "job_type": "train",
  "status": "running",
  "name": "GymKiss LoRA",
  "created_at": "2026-04-21T10:00:00",
  "updated_at": "2026-04-21T10:30:00",
  "details": {
    "phase": "training",
    "current_step": 1500,
    "total_steps": 8000,
    "loss": 0.0312,
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

**status 状态值：**
- `pending` — 在队列中等待
- `running` — 执行中（precomputing 或 training）
- `done` — 完成
- `failed` — 失败（见 `error` 字段）
- `cancelled` — 已取消

**details.phase 阶段值（训练任务）：**
- `pending` → `precomputing` → `training` → `done`

---

## GET /api/jobs/{job_id}/log?tail=100

获取任务日志的最后 N 行。

**Response:**
```json
{"log": "Step 1500/8000 - Loss: 0.0312...", "total_lines": 2340}
```

---

## DELETE /api/jobs/{job_id}

取消任务（running 或 pending 状态）。队列中下一个任务自动开始。

**Response:**
```json
{"job_id": "xyz789", "status": "cancelled"}
```

---

## GET /api/train/{job_id}/samples

列出训练中生成的采样视频。

**Response:**
```json
{
  "samples": ["step_001000_1.mp4", "step_002000_1.mp4"],
  "count": 2
}
```

---

## GET /api/train/{job_id}/samples/{filename}

下载采样视频（mp4）。

---

## GET /api/train/{job_id}/checkpoints

列出保存的模型 checkpoint。

**Response:**
```json
{
  "checkpoints": [
    {"name": "lora_weights_step_00500.safetensors", "size_mb": 651.2},
    {"name": "lora_weights_step_01000.safetensors", "size_mb": 651.2}
  ],
  "count": 2
}
```

---

## GET /api/train/{job_id}/checkpoints/{filename}

下载模型文件（.safetensors）。

---

## 训练输出目录结构

```
outputs/autotrain_{job_id}/
├── checkpoints/
│   ├── lora_weights_step_00500.safetensors
│   └── lora_weights_step_01000.safetensors
├── samples/
│   ├── step_001000_1.mp4
│   └── step_002000_1.mp4
└── training_config.yaml
```

---

## 错误响应格式

```json
{"detail": "data_dir not found: 健身房镜头亲吻"}
```

HTTP 状态码：`400` 参数错误 / `404` 资源不存在
