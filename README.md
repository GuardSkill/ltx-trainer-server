# LTX Trainer Server

LTX-2.3 视频 LoRA 训练的 HTTP API 服务 + Web UI，配套 [ltx-trainer](../ltx-trainer) 使用。

---

## 与 ltx-trainer 的关系

```
LTX-2/packages/
├── ltx-trainer/               ← 训练核心（模型、脚本、数据、输出）
│   ├── scripts/
│   │   ├── process_dataset.py    预处理：VAE 编码 + 文本嵌入
│   │   └── train.py              训练主程序
│   ├── datasets/autotraindata/   视频数据集目录
│   ├── .precomputed_autotrain_*/ 预处理缓存（VAE latents）
│   ├── configs/                  自动生成的训练 YAML
│   ├── outputs/                  训练输出（checkpoint / 采样视频）
│   └── autotrain_jobs/           任务状态 JSON + 日志
│
└── ltx-trainer-server/        ← 本项目：API 服务层
    ├── ltx_api_server.py         FastAPI 服务主文件
    ├── static/index.html         Web UI（单文件，无构建步骤）
    ├── start_train_api.sh        启动脚本（含 FRP 内网穿透）
    └── ltx23_train_byapi_skill/  Claude AI Skill 配置
```

**ltx-trainer-server 不包含任何模型或训练逻辑**，它只负责：
- 把用户的操作请求转化为对 `ltx-trainer` 脚本的调用
- 管理任务队列（训练是单任务顺序执行）
- 提供 HTTP API 和 Web UI 供远程访问

---

## 安装前提

**本项目依赖 `ltx-trainer` 的 Python 环境，必须先完成 ltx-trainer 的安装。**

请参考官方文档完成环境配置（模型下载、uv 虚拟环境、依赖安装）：

👉 **[ltx-trainer 快速开始指南](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/quick-start.md)**

完成上述步骤后，回到本项目继续。

---

## 快速启动

```bash
# 必须在 ltx-trainer 目录运行（uv 虚拟环境在那里）
cd /root/lisiyuan/LTX-2/packages/ltx-trainer

# 启动 API 服务（含 FRP 内网穿透，后台运行）
nohup bash ../ltx-trainer-server/start_train_api.sh > /var/log/ltx_train_api.log 2>&1 &

# 或者直接前台运行（方便查日志）
uv run python ../ltx-trainer-server/ltx_api_server.py
```

服务启动后：
- **Web UI**：`http://localhost:8777` 或 `http://train_ltx23.dev.ad2.cc`
- **API 文档（Swagger）**：`http://localhost:8777/docs`

---

## Web UI 功能

浏览器打开服务地址即可使用全功能界面，无需安装任何客户端。

### ⬇ 下载数据 Tab

| 功能 | 说明 |
|------|------|
| CSV 上传分析 | 拖入或点击上传 CSV → 自动解析所有 `workflow_name` 类别和条数 → 勾选要下载的类别 → 同步到下载列表 |
| 批量下载 | 指定类别名和数量上限，提交后异步下载，进度实时显示 |
| 数据集管理 | 在「已下载数据集」面板中：新建目录、上传视频文件、合并多个数据集、删除数据集 |
| 重用下载配置 | 已完成的下载任务可一键「重用配置」，把类别列表填回表单，重新上传 CSV 再次下载 |

### 🚀 创建训练 Tab

| 功能 | 说明 |
|------|------|
| 数据集选择 | 下拉框直接选择已有数据集，自动填入目录路径 |
| 验证 Prompt | 留空时自动用 `trigger + caption` 拼接 |
| 续训 / 重跑 | 输入历史任务 ID 点「查询」，自动判断：有 checkpoint 则续训，仅有预处理数据则跳过预处理重跑 |

### 任务面板（右侧主区域）

| 功能 | 说明 |
|------|------|
| 训练/下载分开显示 | `🎬 训练任务` 和 `⬇ 下载任务` 独立 tab，各自有状态筛选 |
| 实时进度 | 训练任务显示步数进度条和实时 Loss |
| 日志查看 | 点击任务卡 → 日志 tab，实时滚动 |
| 采样视频 | 训练中每隔 N 步自动生成，可在线播放和下载 |
| Checkpoint | 列出所有保存的 `.safetensors` 文件，支持下载 |
| 取消训练 | 强制终止训练进程（SIGTERM → 5秒超时 → SIGKILL） |
| 删除下载记录 | 已完成的下载任务可彻底从列表删除 |

---

## API 端点总览

Base URL: `http://train_ltx23.dev.ad2.cc`（本机：`http://localhost:8777`）

### 服务状态

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（uptime + 当前运行任务 + 队列长度） |
| GET | `/api/summary` | 任务统计 + 队列总 ETA |

### 数据集管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/datasets` | 列出所有数据集（名称 + 视频数量） |
| POST | `/api/datasets` | 创建新数据集目录 `{"name":"xxx"}` |
| POST | `/api/datasets/merge` | 合并多个数据集 `{"sources":[...],"target":"xxx","move":false}` |
| POST | `/api/datasets/{name}/upload` | 上传视频文件到指定数据集（multipart） |
| DELETE | `/api/datasets/{name}` | 删除数据集目录及所有文件 |
| GET | `/api/datasets/{name}/videos` | 列出数据集中所有视频文件名和大小 |
| GET | `/api/datasets/{name}/videos/{filename}` | 流式获取单个视频（支持 Range/seek，可在线播放） |

### CSV 分析 & 下载

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/csv/analyze` | 分析 CSV，返回所有 `workflow_name` 及条数（不下载） |
| POST | `/api/download` | 提交下载任务（CSV + 类别配置），返回 `job_id` |

### 任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/jobs` | 列出所有任务（最新优先，支持 `?type=train&status=running` 过滤） |
| GET | `/api/jobs/{job_id}` | 获取任务详情（状态、进度、step_rate、eta_minutes） |
| GET | `/api/jobs/{job_id}/log` | 获取任务日志（`?tail=200`） |
| DELETE | `/api/jobs/{job_id}` | 取消运行中/排队中的任务 |
| DELETE | `/api/jobs/{job_id}/remove` | 彻底删除已完成任务记录（`?delete_outputs=true` 同时删除 checkpoint/采样视频，预处理缓存永远保留） |

### 训练

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/train` | 创建训练任务并加入队列 |
| POST | `/api/train/batch` | 批量创建多个训练任务（一次性排队） |
| GET | `/api/train/{job_id}/samples` | 列出采样视频 |
| GET | `/api/train/{job_id}/samples/{filename}` | 下载采样视频 |
| GET | `/api/train/{job_id}/checkpoints` | 列出 checkpoint 文件 |
| GET | `/api/train/{job_id}/checkpoints/{filename}` | 下载 checkpoint |
| GET | `/api/train/{job_id}/resume-info` | 查询续训信息（checkpoint 情况 + 预处理数据情况） |

#### 训练任务参数

```json
{
  "name": "任务名称",
  "data_dir": "数据集名或路径",
  "caption": "视频描述文本",
  "trigger": "触发词（可选）",
  "steps": 8000,
  "rank": 32,
  "with_audio": true,
  "fp8_quant": false,
  "high_capacity": false,
  "checkpoint_interval": 500,
  "validation_interval": 1000,
  "validation_prompt": null,
  "resume_from_job": null,
  "reuse_precomputed_from_job": null
}
```

- `fp8_quant`：开启 FP8 量化，可减少约 30-40% 显存占用，训练速度略降，显存不足时建议开启
- `high_capacity`：额外训练视频 feed-forward 层（`ff.net.0.proj` / `ff.net.2`），LoRA 表达能力更强，速度稍慢（`audio_ff` 层由 `with_audio` 单独控制）
- `resume_from_job`：从指定任务的最新 checkpoint 续训（LR scheduler 自动切换为 constant）
- `reuse_precomputed_from_job`：复用指定任务的预处理数据（跳过 VAE 编码步骤，适合训练被中断且尚无 checkpoint 时）

---

## 训练流程说明

提交训练任务后，自动按以下步骤执行：

```
1. 预处理（precomputing）
   ├─ 遍历数据集目录的所有 mp4/webm
   ├─ 用 VAE 编码视频为 latent
   └─ 用 Gemma-3-12b 编码 caption 为文本嵌入
   → 结果缓存到 .precomputed_autotrain_{job_id}_{name}/

2. 训练（training）
   ├─ 根据参数生成 YAML 配置
   ├─ 调用 scripts/train.py --disable-progress-bars
   ├─ 每 checkpoint_interval 步保存 .safetensors
   └─ 每 validation_interval 步生成采样视频
   → 输出到 outputs/autotrain_{job_id}_{name}/
```

多个任务自动排队，单任务串行执行（显存占用大，不支持并行）。

---

## Claude AI Skill

`ltx23_train_byapi_skill/` 目录包含一个 Claude Code Skill，让 Claude 能通过对话的方式引导用户完成整个训练流程：

```
ltx23_train_byapi_skill/
├── SKILL.md                  Skill 定义（对话原则 + API 速查）
├── scripts/
│   └── ltx_api.py            CLI 辅助脚本（封装常用 API 操作）
└── references/
    ├── api_reference.md      完整 API 文档
    ├── batch_example.json    批量训练配置示例
    └── caption_tips.md       Caption 写法指南
```

### Skill 支持两种运行模式

**对话辅助模式**（默认）：用户自然语言描述需求，逐步引导、每步确认再执行。

**批量自动化模式**：用户提供完整任务列表时，直接调用 `POST /api/train/batch` 一次性全部排队，无需逐个确认。

### Skill 能做什么

| 能力 | 说明 |
|------|------|
| 数据集管理 | 查看/创建/合并/删除数据集；上传本地视频文件 |
| 视频浏览分析 | 下载数据集中的视频进行视觉分析，自动拟写 caption |
| CSV 下载 | 分析 CSV 类别构成，选择类别和数量后批量下载 |
| 创建训练 | 引导填写参数（按需询问，不一次全问）；批量任务一次排队 |
| 实时监控 | 汇报步数/Loss/速度/预计剩余时间；队列总 ETA |
| 续训判断 | 自动调 `resume-info`：有 checkpoint 续训 / 有预处理数据跳过预处理重跑 |
| 失败分析 | 读日志分析 OOM / 数据空 / 预处理卡死 / 服务重启中断，给出处理建议 |
| 查看结果 | 列出采样视频和 checkpoint 文件，提供下载链接 |

### 典型对话

```
用户：我想训练一个抖臀 LoRA
Claude：[调 /api/datasets] 查到「对镜抖臀」有 80 个视频，可直接训练。
        Caption 你有想法吗？没有的话我可以先看几个视频帮你写。

用户：帮我写
Claude：[调 /api/datasets/对镜抖臀/videos，下载 2 个视频分析]
        建议：「镜前独舞，节奏感强，臀部律动自然流畅，
        富有感染力，背景音乐与动作配合默契。」
        触发词用什么？

用户：MirrorBooty，其他默认
Claude：确认提交：
        - 数据集：对镜抖臀（80个视频）
        - 触发词：MirrorBooty
        - Caption：MirrorBooty, 镜前独舞...
        - Steps 8000 / Rank 32 / 音频开启
        提交？

用户：训练到多少了
Claude：[调 /api/jobs?type=train&status=running]
        步数 3200/8000（40%），Loss 0.028，速度 2.3步/分，预计还需 42 分钟

用户：帮我批量训练这 3 个数据集（附参数表）
Claude：[调 /api/train/batch] 已一次性排队 3 个任务：
        任务A job_id=abc123 队列第1 / 任务B job_id=def456 队列第2 / ...
```

---

## 依赖与环境

- Python 环境：由 `ltx-trainer` 的 `uv` 管理（不在本目录单独维护）
- 模型文件：
  - LTX-2.3 基础模型：`/root/models/LTX-2.3/ltx-2.3-22b-dev.safetensors`
  - 文本编码器：`/root/lisiyuan/Models/gemma-3-12b-it-qat-q4_0-unquantized/`
- GPU：训练时约需 80GB 显存；开启 `fp8_quant` 后可降至约 48-55GB
- 公网访问：通过 FRP 内网穿透（配置在 `start_train_api.sh`）
