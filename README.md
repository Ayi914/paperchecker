# 本地论文查阅修改助手

**简体中文** | [English](README_EN.md)

鉴于OpenAI被爆出据说剽窃数学家的成果当作自己的成果发布，于是我紧急编写了这个demo，一个面向Linux的轻量论文辅助CLI Agent。输入问题后，模型可以调用本地工具和论文检索；

默认模型为 Qwen3-14B-AWQ，权重约 9.4 GB。Agent 输入预算默认为 4096 token；模型最大输出由 `.env` 的 `LOCAL_AGENT_MAX_TOKENS` 控制，`.env.example` 中默认是 2048 token。

输入预算和系统 prompt 可以在 `config/agent.yaml` 中修改，因此用途不局限于论文助手。该文件中的 `tool_output_max_chars: 4000` 是工具结果的字符截断上限，不是模型答案的 token 上限。调整参数时需要考虑显存容量。

支持 MCP 协议，可通过工具注册机制接入 MCP 服务。

面向显卡多的没事干并且还极度希望保护自己隐私的用户（

## 使用说明

1. 从 GitHub 页面复制仓库地址，克隆项目：

   ```bash
   git clone https://github.com/Ayi914/paperchecker.git local-agent
   cd local-agent
   ```

   也可以选择 **Code → Download ZIP**，解压后在项目目录打开终端。
2. 在项目目录运行安装命令：

   ```bash
   bash scripts/install.sh
   ```
3. 安装完成后运行：

   ```bash
   bash scripts/start.sh
   ```
   等待模型加载，看到 `>>>` 后输入问题并回车。例如：

   ```text
   >>> 请解释残差连接解决了什么问题
   ```

以后只需运行 `bash scripts/start.sh`

## 用自己的论文

把文本型 PDF、Markdown 或 TXT 放入 `data/knowledge/`，然后建立索引：

```bash
bash scripts/start.sh rag-index
bash scripts/start.sh
```

第一次建索引会下载独立的检索模型，需要网络。示例问题：`根据本地论文总结 YOLOv1 的核心创新，列出来源。` 没有放入论文和建立索引时，回答来自模型本身；默认不启用联网搜索。扫描版 PDF 尚不支持 OCR。

## 常用操作

| 操作 | 输入或命令 |
| --- | --- |
| 查看对话命令 | `/help` |
| 开始新会话 | `/new` |
| 退出 | `/exit`、`/quit` 或 Ctrl+D |
| 单次提问 | `bash scripts/start.sh "总结注意力机制"` |
| 只输出答案 | `bash scripts/start.sh "问题" --quiet` |
| 不保存答案文件 | `bash scripts/start.sh "问题" --no-save` |
| 环境检查 | `bash scripts/doctor.sh` |
| 重新选择模型 | `bash scripts/setup.sh` |

每轮问题和答案默认保存在 `output/月/日/时间_微秒.md`。

当前默认会话上下文只保留在进程中，关闭 CLI 后不会恢复；Markdown 答案文件不会自动读回。需要跨窗口继续时，在 `config/agent.yaml` 将 `run.checkpoint_enabled` 改为 `true`，并始终使用同一会话 ID：

```bash
bash scripts/start.sh -i --thread-id my-paper
```

会话状态保存到 `data/checkpoints/langgraph.sqlite`。输入 `/new` 或不指定会话 ID 的新交互窗口会创建新会话。每次模型调用默认使用最近 4 轮并受 4096 token 输入预算限制。

## 遇到问题

先执行 `bash scripts/doctor.sh`。它检查环境、GPU 和模型配置，不会加载大模型或开始推理。

## 常见问题

- 没有 GPU：使用有 NVIDIA GPU 的 Linux 机器
- 模型目录不完整：运行 `bash scripts/setup.sh` 重新选择或继续下载。
- 下载失败：检查网络；可在有网络的机器下载完整模型目录后复制过来，再选择已有模型。
- 模型启动失败：查看终端指出的 `logs/cli-vllm-*.log`；确认显存没有被其他程序占用。


项目代码使用 [MIT 许可证](LICENSE)。
