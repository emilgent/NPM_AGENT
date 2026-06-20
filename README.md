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
- A running `llama.cpp` server with the OpenAI-compatible endpoint enabled, e.g.:

  ```bash
  ./server -m ./models/your-model.gguf -c 8192 --port 8080
  ```

  The agent talks to `http://localhost:8080/v1/chat/completions` by default
  (override with the `LLAMA_ENDPOINT` environment variable).

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
| `SERPAPI_KEY` | API key for `webSearch` | _(none)_ |

## License

MIT
