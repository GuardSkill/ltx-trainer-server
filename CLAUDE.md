# CLAUDE.md — ltx-trainer-server

本项目为 LTX-2.3 视频 LoRA 训练的 HTTP API 服务层，配合 `../ltx-trainer` 使用。

## 关键文件

| 文件 | 说明 |
|------|------|
| `ltx_api_server.py` | FastAPI 主服务，所有 API 逻辑 |
| `static/index.html` | Web UI，单文件，无构建步骤 |
| `start_train_api.sh` | 启动脚本（含 FRP 内网穿透） |
| `ltx23_train_byapi_skill/SKILL.md` | Claude Code Skill 定义 |

## 架构要点

- **训练是单任务串行执行**（显存占用大，不支持并行）
- **任务状态 JSON** 存储在 `../ltx-trainer/autotrain_jobs/{job_id}.json`
- **预处理缓存** 存储在 `../ltx-trainer/.precomputed_autotrain_{job_id}_{name}/`
- **训练输出** 存储在 `../ltx-trainer/outputs/autotrain_{job_id}_{name}/`
- **数据集** 存储在 `../ltx-trainer/datasets/autotraindata/`
- 服务端口：8777；公网地址：`http://train_ltx23.dev.ad2.cc`

## 重要约束和行为

### Resolution Buckets（帧数）
- 格式：`WxHxF`，其中 F 必须满足 `8n+1`
- `_auto_resolution_buckets()` 用 ffprobe 采样数据集视频长度，取第 10 百分位，自动计算合适的 F
- **短视频（< bucket 最小帧数）会被静默跳过**，导致训练数据为空

### Trigger Word 嵌入
- Trigger 已预嵌入到每个 caption JSON 文件中（由 `process_captions.py` 处理）
- **不要**在 precompute 命令里传 `--lora-trigger`，否则触发词会被写入两次
- `--lora-trigger` 只用于 `train.py`（训练阶段），不用于 `process_dataset.py`（预处理阶段）

### 多数据集训练（data_sources）
- 使用 `data_sources` 数组替代单一的 `data_dir + caption`
- 每个 source 有独立的 `data_dir`、`caption`、`trigger`
- `validation_prompt` 留空时自动用第一个 source 的 `trigger + caption`
- `resume-info` 接口会在 `suggested_restart` 中返回 `data_sources`，供 UI 自动填回表单

### CSV 解析
- **完全在浏览器端（client-side）解析**，不依赖后端 `/api/csv/analyze` 接口
- 原因：大型 CSV（12MB+）会超过代理 body size 限制（502 Bad Gateway）
- 下载前自动过滤 CSV 只发送匹配所选类别的行

### 下载任务常见失败原因
- `HTTP Error 403 Forbidden`：CSV 中的签名 URL 已过期（有效期通常 24h），需重新导出 CSV
- `unknown url type: 'xxx/...'`：CSV 包含相对路径而非 HTTPS URL，需要带签名 URL 的 CSV

## 启动方式

```bash
# 必须在 ltx-trainer 目录运行（uv 虚拟环境在那里）
cd /root/lisiyuan/LTX-2/packages/ltx-trainer

# 后台启动
nohup bash ../ltx-trainer-server/start_train_api.sh > /var/log/ltx_train_api.log 2>&1 &

# 前台启动（方便调试）
uv run python ../ltx-trainer-server/ltx_api_server.py
```

## 修改指南

- 修改 API 逻辑：编辑 `ltx_api_server.py`，重启服务生效
- 修改 Web UI：编辑 `static/index.html`，刷新浏览器生效（无构建步骤）
- 修改 Skill：编辑 `ltx23_train_byapi_skill/SKILL.md`
- 服务器不需要重启即可使用新的 `static/index.html`（FastAPI 直接服务静态文件）
