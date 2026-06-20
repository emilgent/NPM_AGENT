# npm-agent

An autonomous AI Agent CLI tool for NPM development. It drives a local
[`llama.cpp`](https://github.com/ggerganov/llama.cpp) server (exposed through its
OpenAI-compatible REST API) using **native function calling**, and operates
directly on the codebase in your current working directory.

```
Thought -> Tool Selection -> Safety Check -> Execution -> Observation
```

## Requirements

- Node.js **v20 or higher**
- A `llama.cpp` server binary on your `PATH` (`llama-server`, `llama-cpp-server`
  or the legacy `server`). The agent can **start this for you automatically** on
  launch — see below. If you prefer to run it yourself:

  ```bash
  llama-server -m ./models/your-model.gguf -c 8192 --port 8080
  ```

  The agent talks to `http://localhost:8080/v1/chat/completions` by default
  (override with the `LLAMA_ENDPOINT` environment variable).

## Automatic server start & model selection

When you launch `npm-agent`, after choosing your session and context window it
will:

1. Check whether a llama.cpp server is already running on the endpoint and, if
   so, reuse it.
2. Otherwise offer to **start one automatically**. It locates the server binary
   (or `LLAMA_SERVER_BIN`), then asks **where your `.gguf` models live**,
   recursively scans that directory and lets you pick the model to load.
3. Spawn `llama-server -m <model> -c <contextWindow> --host 127.0.0.1 --port 8080`
   and wait until its HTTP API reports healthy before the agent loop begins.

The server we start is shut down automatically when you `/exit` or Ctrl+C. You
can always pick *“Skip auto-start”* to manage the server yourself.

## Installation

```bash
git clone https://github.com/emilgent/NPM_AGENT.git
cd NPM_AGENT
npm install
npm link          # makes the `npm-agent` command globally available
```

## Usage

Run the agent from inside the project you want it to work on:

```bash
cd /path/to/your/project
npm-agent
```

On startup you will be guided through an interactive menu:

1. **[Start New Session]** — describe a goal; a fresh session is created and the
   core system prompt is injected exactly once.
2. **[Load Existing Session]** — resume any session saved under
   `.agent_sessions/`.

You will also declare your model's **maximum context window** (4096 / 8192 /
16384 / custom).

Sessions are persisted to `.agent_sessions/` **only** when a task is reported
complete or on a clean exit (Ctrl+C). You can also force a save at any prompt
with `/save`, and quit with `/exit`.

## Tools

| Tool | Description |
| --- | --- |
| `listFiles()` | Recursive project tree (skips `node_modules`, `.git`, `.agent_sessions`). |
| `readFile(filePath)` | Reads a text file (max 500 KB; binary files rejected). |
| `writeFile(filePath, content)` | Writes a text file, creating parent dirs. |
| `executeCommand(command)` | Runs shell commands (1h timeout). Mutating commands need approval. |
| `webSearch(query)` | Web search via SerpAPI (`SERPAPI_KEY`). |
| `fetchWebpage(url)` | Fetches and cleans a web page to readable text. |

## Safety

- **Binary safeguard:** file tools refuse anything that isn't a recognised text
  format (`Error: Binary file manipulation is strictly prohibited.`).
- **Command interceptor:** read-only commands run automatically; state-mutating
  commands (`npm install`, `rm`, `git commit`, …) are shown in magenta and
  require an interactive `Y/N` confirmation.
- **Size wall:** files over 500 KB are rejected by `readFile`.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLAMA_ENDPOINT` | Chat-completions endpoint | `http://localhost:8080/v1/chat/completions` |
| `LLAMA_MODEL` | Model name advertised to the server | `local-model` |
| `LLAMA_SERVER_BIN` | Path/name of the llama.cpp server binary | auto-detected |
| `LLAMA_MODELS_DIR` | Default directory shown when picking a model | auto-detected |
| `LLAMA_PORT` | Port the auto-started server listens on | `8080` |
| `LLAMA_HOST` | Host the auto-started server binds to | `127.0.0.1` |
| `LLAMA_AUTO_START` | Set to `0`/`false` to default the auto-start prompt to “no” | `1` |
| `LLAMA_SERVER_EXTRA_ARGS` | Extra flags appended to the server (e.g. `-ngl 99`) | _(none)_ |
| `LLAMA_HEALTH_TIMEOUT_MS` | Max wait for the server to become healthy | `180000` |
| `SERPAPI_KEY` | API key for `webSearch` | _(none)_ |

## License

MIT
