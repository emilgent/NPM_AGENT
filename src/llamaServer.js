/**
 * src/llamaServer.js
 * ----------------------------------------------------------------------------
 * Local llama.cpp server lifecycle manager.
 *
 * Responsibilities:
 *   - Locate a usable `llama-server` (or legacy `server`) binary.
 *   - Let the operator choose WHERE their GGUF models live and pick one
 *     interactively (recursive `*.gguf` scan).
 *   - Spawn the server with the chosen model / context window / port and wait
 *     until its OpenAI-compatible HTTP API reports healthy.
 *   - Reuse an already-running server if one is reachable, instead of starting
 *     a duplicate.
 *   - Cleanly terminate any server WE started on exit.
 *
 * The CLI consumes `prepareLlamaServer()` which returns the resolved endpoint
 * plus an optional handle (`stop()`) for the spawned process.
 * ----------------------------------------------------------------------------
 */

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';

import chalk from 'chalk';
import inquirer from 'inquirer';
import ora from 'ora';
import fetch from 'node-fetch';

const execFileAsync = promisify(execFile);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Candidate executable names for the llama.cpp HTTP server, in priority order. */
const SERVER_BINARY_CANDIDATES = ['llama-server', 'llama-cpp-server', 'server'];

/** Default port the server listens on (overridable via LLAMA_PORT). */
const DEFAULT_PORT = Number(process.env.LLAMA_PORT) || 8080;

/** Host the server binds to. Loopback by default for safety. */
const DEFAULT_HOST = process.env.LLAMA_HOST || '127.0.0.1';

/**
 * Maximum time (ms) to wait for the server to become healthy after spawning.
 * Large models can take a while to memory-map / load, hence the generous wait.
 */
const HEALTH_TIMEOUT_MS = Number(process.env.LLAMA_HEALTH_TIMEOUT_MS) || 180_000;

/** Polling interval (ms) while waiting for the health endpoint. */
const HEALTH_POLL_INTERVAL_MS = 1_000;

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

/**
 * Build the base URL (scheme + host + port) from a host/port pair.
 * @param {string} host
 * @param {number} port
 * @returns {string}
 */
export function buildBaseUrl(host, port) {
  // 0.0.0.0 is a bind address, not a connect address: talk to it via loopback.
  const connectHost = host === '0.0.0.0' || host === '::' ? '127.0.0.1' : host;
  return `http://${connectHost}:${port}`;
}

/**
 * Build the chat-completions endpoint from a base URL.
 * @param {string} baseUrl
 * @returns {string}
 */
export function buildEndpoint(baseUrl) {
  return `${baseUrl.replace(/\/+$/, '')}/v1/chat/completions`;
}

// ---------------------------------------------------------------------------
// Binary detection
// ---------------------------------------------------------------------------

/**
 * Resolve the llama.cpp server binary.
 *
 * Order of resolution:
 *   1. The LLAMA_SERVER_BIN environment variable (absolute path or PATH name).
 *   2. The first of SERVER_BINARY_CANDIDATES found on PATH.
 *
 * @returns {Promise<string|null>} The resolved binary command/path, or null.
 */
export async function detectServerBinary() {
  const explicit = process.env.LLAMA_SERVER_BIN;
  if (explicit && explicit.trim()) {
    return explicit.trim();
  }

  for (const candidate of SERVER_BINARY_CANDIDATES) {
    try {
      // `command -v` resolves both PATH executables and shell builtins reliably.
      const { stdout } = await execFileAsync('bash', ['-lc', `command -v ${candidate}`]);
      const resolved = stdout.trim();
      if (resolved) {
        return resolved;
      }
    } catch {
      /* not found — try the next candidate */
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

/**
 * Default directories to suggest when prompting for the models location.
 * @returns {string} A best-guess default models directory.
 */
export function defaultModelsDir() {
  if (process.env.LLAMA_MODELS_DIR && process.env.LLAMA_MODELS_DIR.trim()) {
    return process.env.LLAMA_MODELS_DIR.trim();
  }
  const home = os.homedir();
  const candidates = [
    path.join(process.cwd(), 'models'),
    path.join(home, 'models'),
    path.join(home, '.cache', 'llama.cpp'),
    path.join(home, '.cache', 'huggingface'),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(dir)) {
      return dir;
    }
  }
  return path.join(home, 'models');
}

/**
 * Recursively scan a directory for `*.gguf` model files.
 *
 * @param {string} dir - Directory to scan.
 * @param {number} [maxDepth=4] - Maximum recursion depth.
 * @returns {Promise<Array<{path: string, size: number}>>} Discovered models.
 */
export async function findGgufModels(dir, maxDepth = 4) {
  const results = [];

  async function walk(current, depth) {
    if (depth > maxDepth) return;
    let entries;
    try {
      entries = await fsp.readdir(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        // Skip noisy / irrelevant directories to keep the scan fast.
        if (['node_modules', '.git', '.cache'].includes(entry.name) && depth > 0) {
          // still allow .cache as a top-level root, but don't recurse blindly deep
        }
        await walk(full, depth + 1);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.gguf')) {
        let size = 0;
        try {
          size = (await fsp.stat(full)).size;
        } catch {
          /* ignore stat failures */
        }
        results.push({ path: full, size });
      }
    }
  }

  await walk(dir, 0);
  results.sort((a, b) => a.path.localeCompare(b.path));
  return results;
}

/**
 * Human-readable byte formatter (e.g. "4.1 GB").
 * @param {number} bytes
 * @returns {string}
 */
function formatBytes(bytes) {
  if (!bytes) return 'unknown size';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

// ---------------------------------------------------------------------------
// Health checks
// ---------------------------------------------------------------------------

/**
 * Probe whether a llama.cpp server is reachable and ready at a base URL.
 *
 * @param {string} baseUrl - e.g. http://127.0.0.1:8080
 * @param {number} [timeoutMs=1500] - Per-request timeout.
 * @returns {Promise<boolean>} True if the server responds healthy.
 */
export async function isServerHealthy(baseUrl, timeoutMs = 1_500) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: controller.signal });
    if (res.ok) return true;
    // Some builds expose /v1/models even before /health flips to 200.
    if (res.status === 503) return false;
  } catch {
    /* fall through to the /v1/models probe */
  } finally {
    clearTimeout(timer);
  }

  const controller2 = new AbortController();
  const timer2 = setTimeout(() => controller2.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/v1/models`, { signal: controller2.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer2);
  }
}

/**
 * Poll the health endpoint until the server is ready or the timeout elapses.
 *
 * @param {string} baseUrl
 * @param {number} timeoutMs
 * @param {() => boolean} [isAlive] - Optional liveness check for the child proc.
 * @returns {Promise<boolean>} True once healthy; false on timeout.
 */
export async function waitForHealth(baseUrl, timeoutMs, isAlive) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (isAlive && !isAlive()) {
      return false; // the process died while we were waiting
    }
    if (await isServerHealthy(baseUrl)) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_INTERVAL_MS));
  }
  return false;
}

// ---------------------------------------------------------------------------
// Server spawning
// ---------------------------------------------------------------------------

/**
 * Spawn a llama.cpp server process for the given model.
 *
 * @param {object} opts
 * @param {string} opts.binary - Server binary command/path.
 * @param {string} opts.modelPath - Absolute path to the .gguf model.
 * @param {number} opts.contextWindow - Context size (-c).
 * @param {number} [opts.port=DEFAULT_PORT]
 * @param {string} [opts.host=DEFAULT_HOST]
 * @returns {{process: import('node:child_process').ChildProcess, baseUrl: string, endpoint: string, stop: () => void}}
 */
export function spawnServer({ binary, modelPath, contextWindow, port = DEFAULT_PORT, host = DEFAULT_HOST }) {
  const args = [
    '-m', modelPath,
    '-c', String(contextWindow),
    '--host', host,
    '--port', String(port),
  ];

  // Allow advanced users to inject extra flags (e.g. "-ngl 99 --parallel 2").
  if (process.env.LLAMA_SERVER_EXTRA_ARGS && process.env.LLAMA_SERVER_EXTRA_ARGS.trim()) {
    args.push(...process.env.LLAMA_SERVER_EXTRA_ARGS.trim().split(/\s+/));
  }

  const child = spawn(binary, args, {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });

  // Stream a compact view of the server's own logs (dimmed) so the operator can
  // see load progress / errors without it drowning the agent output.
  const relayLog = (buf) => {
    const text = buf.toString();
    text
      .split('\n')
      .filter((line) => line.trim().length > 0)
      .forEach((line) => console.log(chalk.dim(`  [llama] ${line}`)));
  };
  if (child.stdout) child.stdout.on('data', relayLog);
  if (child.stderr) child.stderr.on('data', relayLog);

  const baseUrl = buildBaseUrl(host, port);

  let stopped = false;
  const stop = () => {
    if (stopped) return;
    stopped = true;
    try {
      child.kill('SIGTERM');
    } catch {
      /* already gone */
    }
  };

  return { process: child, baseUrl, endpoint: buildEndpoint(baseUrl), stop };
}

// ---------------------------------------------------------------------------
// Interactive orchestration
// ---------------------------------------------------------------------------

/**
 * Interactively choose a models directory and a specific .gguf file.
 *
 * Keeps re-prompting until the operator either selects a model, supplies an
 * explicit model path, or chooses to skip auto-start.
 *
 * @returns {Promise<string|null>} Absolute model path, or null to skip.
 */
async function chooseModelFile() {
  let modelsDir = defaultModelsDir();

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { dir } = await inquirer.prompt([
      {
        type: 'input',
        name: 'dir',
        message: 'Where are your AI models (.gguf) located?',
        default: modelsDir,
        validate: (value) => (value && value.trim().length > 0 ? true : 'Please enter a directory path.'),
      },
    ]);
    modelsDir = path.resolve(dir.trim().replace(/^~(?=$|\/)/, os.homedir()));

    const spinner = ora({ text: chalk.blue(`Scanning ${modelsDir} for .gguf models…`), color: 'blue' }).start();
    const models = await findGgufModels(modelsDir);
    spinner.stop();

    const choices = models.map((m) => ({
      name: `${path.relative(modelsDir, m.path) || path.basename(m.path)}  (${formatBytes(m.size)})`,
      value: m.path,
    }));

    if (choices.length === 0) {
      console.log(chalk.yellow(`No .gguf files found under ${modelsDir}.`));
    }

    choices.push(new inquirer.Separator());
    choices.push({ name: 'Enter a model file path manually…', value: '__manual__' });
    choices.push({ name: 'Choose a different directory…', value: '__rescan__' });
    choices.push({ name: 'Skip auto-start (I will run the server myself)', value: '__skip__' });

    const { selection } = await inquirer.prompt([
      {
        type: 'list',
        name: 'selection',
        message: 'Select the model to load:',
        choices,
        pageSize: 15,
      },
    ]);

    if (selection === '__skip__') return null;
    if (selection === '__rescan__') continue;
    if (selection === '__manual__') {
      const { manualPath } = await inquirer.prompt([
        {
          type: 'input',
          name: 'manualPath',
          message: 'Enter the absolute path to the .gguf model file:',
          validate: (value) => {
            const p = path.resolve(value.trim().replace(/^~(?=$|\/)/, os.homedir()));
            if (!fs.existsSync(p)) return `File not found: ${p}`;
            if (!p.toLowerCase().endsWith('.gguf')) return 'Expected a .gguf file.';
            return true;
          },
        },
      ]);
      return path.resolve(manualPath.trim().replace(/^~(?=$|\/)/, os.homedir()));
    }
    return selection; // a concrete model path
  }
}

/**
 * Full startup flow: optionally auto-start a local llama.cpp server after letting
 * the operator pick where their models live.
 *
 * Behaviour:
 *   - If a server is already healthy at the default endpoint, reuse it.
 *   - Otherwise offer to auto-start one; locate the binary, pick a model, spawn,
 *     and wait for health.
 *   - Always degrades gracefully: on any problem it returns the default endpoint
 *     so the user can still point the agent at an externally managed server.
 *
 * @param {object} opts
 * @param {number} opts.contextWindow - Declared context window (passed as -c).
 * @returns {Promise<{endpoint: string, handle: object|null}>}
 */
export async function prepareLlamaServer({ contextWindow }) {
  const port = DEFAULT_PORT;
  const host = DEFAULT_HOST;
  const baseUrl = buildBaseUrl(host, port);
  const defaultEndpoint = process.env.LLAMA_ENDPOINT || buildEndpoint(baseUrl);

  // 1) Reuse an already-running server if present.
  if (await isServerHealthy(baseUrl)) {
    console.log(chalk.green(`✔ Detected a running llama.cpp server at ${baseUrl} — reusing it.`));
    return { endpoint: defaultEndpoint, handle: null };
  }
  // If the user pinned a custom endpoint and it is alive, reuse that too.
  if (process.env.LLAMA_ENDPOINT) {
    const customBase = process.env.LLAMA_ENDPOINT.replace(/\/v1\/.*$/, '');
    if (customBase !== baseUrl && (await isServerHealthy(customBase))) {
      console.log(chalk.green(`✔ Detected a running server at ${customBase} — reusing it.`));
      return { endpoint: defaultEndpoint, handle: null };
    }
  }

  // 2) Offer to auto-start.
  const autoStartDefault = process.env.LLAMA_AUTO_START !== '0' && process.env.LLAMA_AUTO_START !== 'false';
  const { autoStart } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'autoStart',
      message: 'No local llama.cpp server detected. Start one automatically now?',
      default: autoStartDefault,
    },
  ]);

  if (!autoStart) {
    console.log(chalk.dim(`Continuing without auto-start. The agent will use ${defaultEndpoint}.`));
    return { endpoint: defaultEndpoint, handle: null };
  }

  // 3) Locate the server binary.
  const binary = await detectServerBinary();
  if (!binary) {
    console.log(
      chalk.yellow(
        'Could not find a llama.cpp server binary (looked for: ' +
          SERVER_BINARY_CANDIDATES.join(', ') +
          ').\n  Set LLAMA_SERVER_BIN to its full path, then restart. ' +
          `Falling back to ${defaultEndpoint}.`,
      ),
    );
    return { endpoint: defaultEndpoint, handle: null };
  }
  console.log(chalk.dim(`Using server binary: ${binary}`));

  // 4) Pick a model.
  const modelPath = await chooseModelFile();
  if (!modelPath) {
    console.log(chalk.dim(`Skipping auto-start. The agent will use ${defaultEndpoint}.`));
    return { endpoint: defaultEndpoint, handle: null };
  }

  // 5) Spawn + wait for health.
  console.log(
    chalk.blueBright(
      `\nStarting llama.cpp server:\n  ${binary} -m ${modelPath} -c ${contextWindow} --host ${host} --port ${port}\n`,
    ),
  );
  const handle = spawnServer({ binary, modelPath, contextWindow, port, host });

  const spinner = ora({ text: chalk.blue('Waiting for the server to become ready…'), color: 'blue' }).start();
  const alive = () => handle.process.exitCode === null && !handle.process.killed;
  const healthy = await waitForHealth(handle.baseUrl, HEALTH_TIMEOUT_MS, alive);

  if (!healthy) {
    spinner.fail(chalk.red('The llama.cpp server did not become healthy in time.'));
    handle.stop();
    console.log(
      chalk.yellow(
        `The server failed to start (check the [llama] logs above). Falling back to ${defaultEndpoint}.`,
      ),
    );
    return { endpoint: defaultEndpoint, handle: null };
  }

  spinner.succeed(chalk.green(`llama.cpp server is ready at ${handle.baseUrl}.`));
  return { endpoint: handle.endpoint, handle };
}
