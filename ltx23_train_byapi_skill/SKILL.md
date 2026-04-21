---
name: ltx23-train-byapi
description: 引导并协助用户完成 LTX-2.3 视频 LoRA 模型的完整训练流程，包括从 CSV 下载视频数据集、创建训练任务、监控训练进度（步数/Loss）、查看采样视频、下载模型 checkpoint。当用户提到要训练 LTX 模型、下载训练数据、查看训练进度、下载 LoRA 模型等意图时触发本 skill。API 地址：http://train_ltx23.dev.ad2.cc（本机 http://localhost:8777）。
---

# LTX-2.3 训练助手 Skill

通过 HTTP API 完整引导用户完成 LTX-2.3 视频 LoRA 训练的每一个环节。

## API 基础信息

- **公网地址**：`http://train_ltx23.dev.ad2.cc`
- **本机地址**：`http://localhost:8777`
- **交互界面**：直接浏览器打开上述地址即可
- **API 文档**：`http://train_ltx23.dev.ad2.cc/docs`
- **辅助脚本**：`scripts/ltx_api.py`（封装所有 API 操作，免记 curl）

---

## 完整训练流程

### 阶段一：检查服务状态

首先确认 API 服务是否在线：

```bash
python3 scripts/ltx_api.py status
# 或
curl -s http://train_ltx23.dev.ad2.cc/api/jobs | python3 -m json.tool | head -5
```

若服务不在线，告知用户需要 SSH 到训练机执行：
```bash
nohup bash /root/lisiyuan/LTX-2/packages/ltx-trainer-server/start_train_api.sh > /var/log/ltx_train_api.log 2>&1 &
```

---

### 阶段二：下载训练数据（若用户已有数据可跳过）

**需要询问用户的问题：**
1. 是否有新的 CSV 文件？（CSV 来自 comfly 导出的签名 URL 文件）
2. 要下载哪些类别（workflow_name）？每个类别下载多少条（默认 80）？

**执行下载：**

```bash
# 方式一：使用辅助脚本
python3 scripts/ltx_api.py download \
  --csv /path/to/signed_urls.csv \
  --categories '健身房镜头亲吻:80,对镜抖臀:60'

# 方式二：curl
curl -s -X POST http://train_ltx23.dev.ad2.cc/api/download \
  -F "csv_file=@/path/to/signed_urls.csv" \
  -F 'categories=[{"name":"健身房镜头亲吻","limit":80}]'
```

**轮询直到下载完成：**

```bash
python3 scripts/ltx_api.py watch <job_id>
```

下载完成后，数据保存在：`datasets/autotraindata/<类别名>/`

---

### 阶段三：创建训练任务

**需要询问用户的问题（每个模板）：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `name` | 任务名称，便于识别 | 必填 |
| `data_dir` | 数据目录，填类别名即可 | 必填 |
| `caption` | 训练描述，对所有视频生效，也作为验证 prompt | 必填 |
| `trigger` | 触发词，训练时加在 caption 前 | 可选（留空） |
| `steps` | 训练步数 | 8000 |
| `rank` | LoRA rank | 32 |
| `with_audio` | 是否训练音频 | true |
| `checkpoint_interval` | 多少步保存一次模型 | 500 |
| `validation_interval` | 多少步生成一次采样视频 | 1000 |

**Caption 写法提示：**
- 好的 caption 应描述视频里的动作、场景、情绪、声音
- 若有触发词，caption 中不需要重复写，触发词会自动加在前面
- 例：触发词 `GymKiss`，caption：`在健身房镜头前，男女两人激情接吻，情绪热烈，有轻微的亲吻声和呼吸声。`

**创建任务：**

```bash
# 方式一：使用辅助脚本
python3 scripts/ltx_api.py train \
  --name "GymKiss LoRA" \
  --data-dir "健身房镜头亲吻" \
  --caption "在健身房镜头前，男女两人激情接吻，情绪热烈。" \
  --trigger "GymKiss" \
  --steps 8000 \
  --rank 32

# 方式二：curl
curl -s -X POST http://train_ltx23.dev.ad2.cc/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GymKiss LoRA",
    "data_dir": "健身房镜头亲吻",
    "caption": "在健身房镜头前，男女两人激情接吻，情绪热烈。",
    "trigger": "GymKiss",
    "steps": 8000,
    "rank": 32,
    "with_audio": true
  }'
```

**多个模板批量排队：**

```bash
python3 scripts/ltx_api.py batch-train --config /path/to/batch.json
```

batch.json 格式参见 `references/batch_example.json`。

---

### 阶段四：监控训练进度

```bash
# 查看所有任务
python3 scripts/ltx_api.py jobs

# 持续监控某个任务（每5秒刷新）
python3 scripts/ltx_api.py watch <job_id>

# 查看训练日志最后100行
python3 scripts/ltx_api.py log <job_id>
```

**状态说明：**

| status | phase | 含义 |
|--------|-------|------|
| `pending` | `pending` | 在队列中等待 |
| `running` | `precomputing` | 正在预处理视频（VAE 编码 + 文本嵌入） |
| `running` | `training` | 正在训练，可查看 `current_step`/`total_steps`/`loss` |
| `done` | `done` | 训练完成 |
| `failed` | — | 失败，查看 `error` 字段和日志 |

---

### 阶段五：查看采样视频

每隔 `validation_interval` 步会生成一个采样视频，可以评估训练效果：

```bash
# 列出所有采样视频
python3 scripts/ltx_api.py samples <job_id>

# 下载某个采样视频
python3 scripts/ltx_api.py download-sample <job_id> step_002000_1.mp4 ./output.mp4
```

或直接浏览器打开 HTML 界面查看内嵌视频播放。

---

### 阶段六：下载模型 Checkpoint

```bash
# 列出所有 checkpoint（含文件大小）
python3 scripts/ltx_api.py checkpoints <job_id>

# 下载最新或指定步数的 checkpoint
python3 scripts/ltx_api.py download-ckpt <job_id> lora_weights_step_08000.safetensors ./my_lora.safetensors
```

---

### 阶段七：从 Checkpoint 续训

当训练中断、或想在已有基础上继续训练更多步数时使用。

**重要限制：** 续训只恢复模型权重，不恢复 optimizer 状态和步数计数器（从 step 0 重新开始计数）。`steps` 应填写**剩余步数**而非总步数。

```bash
# 1. 先查询上次训练情况（自动计算剩余步数）
python3 scripts/ltx_api.py resume-info <old_job_id>

# 2. 从该任务最新 checkpoint 续训（推荐）
python3 scripts/ltx_api.py train \
  --name "GymKiss 续训" \
  --data-dir "健身房镜头亲吻" \
  --caption "..." \
  --steps 4000 \
  --resume-from-job <old_job_id>

# 3. 或手动指定 checkpoint 文件路径
python3 scripts/ltx_api.py train \
  --name "GymKiss 续训" \
  --data-dir "健身房镜头亲吻" \
  --caption "..." \
  --steps 4000 \
  --load-checkpoint /root/lisiyuan/LTX-2/packages/ltx-trainer/outputs/autotrain_xxx/checkpoints/lora_weights_step_04000.safetensors
```

续训时 API 自动将 `scheduler_type` 切换为 `constant`，避免学习率从高处重新 warmup。

HTML UI：训练表单底部「续训」区域，输入 job_id 点「查询」自动填充所有参数；或直接点任务卡片上的 ↩ **续训** 按钮。

---

### 阶段八：取消任务

```bash
python3 scripts/ltx_api.py cancel <job_id>
```

取消后，队列中下一个任务自动开始。

---

## 对话引导规则

1. **主动询问**：不要假设用户已知道参数，逐步引导：先问有没有 CSV，再问类别，再问 caption，再问参数。
2. **Caption 协助**：如果用户不知道如何写 caption，根据类别名帮他生成一段描述，让用户确认后再使用。
3. **批量友好**：如果用户要训练多个类别，列出所有参数让用户确认后，一次性批量创建。
4. **进度追踪**：创建任务后，主动告知用户如何查看进度，并解释各个阶段的含义。
5. **失败处理**：如果任务失败，帮用户查看日志并分析原因（常见：数据目录不存在、视频数量为 0、VRAM 不足）。
6. **不重复操作**：先 `python3 scripts/ltx_api.py jobs` 查看当前任务列表，避免重复创建相同任务。
7. **续训判断**：用户提到「继续训练」「从断点恢复」「训练中断」时，先调用 `resume-info <job_id>` 查询，确认剩余步数后再创建续训任务。

## 主要文件

- `scripts/ltx_api.py` — CLI 辅助脚本，封装所有 API 操作
- `references/api_reference.md` — 完整 API 端点文档
- `references/batch_example.json` — 批量训练配置示例
- `references/caption_tips.md` — Caption 和触发词写法指南
