# Local Paper Reading and Editing Assistant

[简体中文](README.md) | **English**

Reports alleging that OpenAI had presented mathematicians' work as its own prompted me to quickly put together this demo: a lightweight paper-assistance CLI agent for Linux. Given a question, the model can call local tools and search your papers.

The default model is Qwen3-14B-AWQ, with approximately 9.4 GB of weights. The agent's default input budget is 4,096 tokens. The maximum model output is controlled by `LOCAL_AGENT_MAX_TOKENS` in `.env`; the default in `.env.example` is 2,048 tokens.

You can change the input budget and system prompt in `config/agent.yaml`, so the agent is not limited to paper assistance. The `tool_output_max_chars: 4000` setting in that file truncates tool results by character count; it is not the token limit for the final answer. Keep GPU memory capacity in mind when adjusting these settings.

MCP is supported: MCP services can be connected through the tool registry.

For people with GPUs to spare, time on their hands, and a strong desire to protect their privacy :)

## Getting started

1. Copy the repository URL from GitHub and clone the project:

   ```bash
   git clone https://github.com/Ayi914/paperchecker.git local-agent
   cd local-agent
   ```

   Alternatively, select **Code → Download ZIP**, extract it, and open a terminal in the project directory.
2. Run the installer from the project directory:

   ```bash
   bash scripts/install.sh
   ```

3. Once installation is complete, start the agent:

   ```bash
   bash scripts/start.sh
   ```

   Wait for the model to load. When you see `>>>`, enter a question and press Enter:

   ```text
   >>> What problem do residual connections solve?
   ```

For subsequent sessions, just run `bash scripts/start.sh`.

## Using your own papers

Place text-based PDF, Markdown, or TXT files in `data/knowledge/`, then build the index:

```bash
bash scripts/start.sh rag-index
bash scripts/start.sh
```

The first indexing run downloads a separate embedding model and requires an internet connection. Example question: `Based on the local papers, summarize the main innovations in YOLOv1 and list the sources.` Without papers and an index, answers come from the model itself. Web search is disabled by default. Scanned PDFs are not yet supported by OCR.

## Common operations

| Action | Input or command |
| --- | --- |
| Show conversation commands | `/help` |
| Start a new conversation | `/new` |
| Exit | `/exit`, `/quit`, or Ctrl+D |
| Ask a single question | `bash scripts/start.sh "Summarize attention mechanisms"` |
| Print only the final answer | `bash scripts/start.sh "Question" --quiet` |
| Do not save the answer file | `bash scripts/start.sh "Question" --no-save` |
| Check the environment | `bash scripts/doctor.sh` |
| Select or configure a model | `bash scripts/setup.sh` |

Each question and final answer is saved by default to `output/month/day/time_microseconds.md`.

Conversation context is kept in memory by default and cannot be restored after exiting the CLI. Saved Markdown answers are not automatically loaded as context. To continue a conversation across windows or sessions, set `run.checkpoint_enabled` to `true` in `config/agent.yaml` and use the same conversation ID every time:

```bash
bash scripts/start.sh -i --thread-id my-paper
```

Conversation state is stored in `data/checkpoints/langgraph.sqlite`. Entering `/new`, or opening an interactive session without specifying a conversation ID, creates a new conversation. By default, each model call uses up to the latest four turns, subject to a 4,096-token input budget.

## Troubleshooting

Start with `bash scripts/doctor.sh`. It checks dependencies, GPU availability, and model configuration without loading the large model or running inference.

## Common issues

- **No GPU:** use a Linux machine with an NVIDIA GPU.
- **Incomplete model directory:** run `bash scripts/setup.sh` to select a complete model directory or resume downloading.
- **Download failure:** check your network connection. You can also download the complete model directory on another machine, copy it over, and select the existing model.
- **Model startup failure:** check the `logs/cli-vllm-*.log` file indicated in the terminal and confirm that other programs are not occupying GPU memory.

Project code is licensed under the [MIT License](LICENSE).
