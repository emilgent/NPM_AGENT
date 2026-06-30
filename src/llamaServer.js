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

import pc from 'picocolors';
import {
  select,
  text,
  confirm,
  isCancel,
  cancel,
  spinner,
  intro,
  outro,
  log,
} from '@clack/prompts';
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
 *   1. The LLAMA_SERVER_BIN environment variable.
 *   2. The first of SERVER_BINARY_CANDIDATES found on PATH.
 *   3. Common install locations for each candidate name.
 *
 * @returns {Promise<string|null>} The resolved binary command/path, or null.
 */
export async function detectServerBinary() {
  const explicit = process.env.LLAMA_SERVER_BIN;
  if (explicit && explicit.trim()) {
    const trimmed = explicit.trim();
    if (trimmed.includes(path.sep) || trimmed.startsWith('~')) {
      const resolved = resolveExistingBinaryPath(trimmed);
      if (resolved) return resolved;
    } else {
      return trimmed;
    }
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

  for (const candidate of SERVER_BINARY_CANDIDATES) {
    const resolved = resolveBinaryFromCommonLocations(candidate);
    if (resolved) {
      return resolved;
    }
  }

  return null;
}

/**
 * Expand a leading `~` in a path string.
 * @param {string} value
 * @returns {string}
 */
function expandHome(value) {
  return value.replace(/^~(?=$|\/)/, os.homedir());
}

/**
 * Resolve an existing binary file path.
 * @param {string} candidate
 * @returns {string|null}
 */
function resolveExistingBinaryPath(candidate) {
  const resolved = path.resolve(expandHome(candidate));
  if (!fs.existsSync(resolved)) {
    return null;
  }
  try {
    const stat = fs.statSync(resolved);
    return stat.isFile() ? resolved : null;
  } catch {
    return null;
  }
}

/**
 * Resolve a binary from common local install locations.
 * @param {string} candidate
 * @returns {string|null}
 */
function resolveBinaryFromCommonLocations(candidate) {
  const home = os.homedir();
  const locations = [
    path.join(process.cwd(), candidate),
    path.join(process.cwd(), 'build', 'bin', candidate),
    path.join(home, 'llama.cpp', 'build', 'bin', candidate),
    path.join(home, 'llama.cpp', candidate),
    path.join('/usr/local/bin', candidate),
    path.join('/opt/llama.cpp/bin', candidate),
  ];

  let firstFile = null;
  for (const location of locations) {
    if (!fs.existsSync(location)) {
      continue;
    }
    try {
      const stat = fs.statSync(location);
      if (!stat.isFile()) {
        continue;
      }
      if ((stat.mode & 0o111) !== 0) {
        return location;
      }
      if (!firstFile) {
        firstFile = location;
      }
    } catch {
      /* ignore unreadable paths and keep searching */
    }
  }

  return firstFile;
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

/**
 * Normalize a user-entered path, expanding a leading `~`.
 * @param {string} value
 * @returns {string}
 */
function normalizeInputPath(value) {
  return path.resolve(expandHome(value.trim()));
}

/**
 * Validate a user-entered GGUF file path.
 * @param {string} value
 * @param {boolean} allowEmpty
 * @returns {true|string}
 */
function validateGgufPath(value, allowEmpty = false) {
  const trimmed = value.trim();
  if (!trimmed) {
    return allowEmpty ? true : 'Please enter a path.';
  }

  const resolved = normalizeInputPath(trimmed);
  if (!resolved.toLowerCase().endsWith('.gguf')) {
    return 'Expected a .gguf file.';
  }

  if (!fs.existsSync(resolved)) {
    return `File not found: ${resolved}`;
  }

  try {
    const stat = fs.statSync(resolved);
    if (!stat.isFile()) {
      return `Not a file: ${resolved}`;
    }
  } catch {
    return `File not found: ${resolved}`;
  }

  return true;
}

/**
 * Validate a user-entered binary path.
 * @param {string} value
 * @param {boolean} allowEmpty
 * @returns {true|string}
 */
function validateBinaryPath(value, allowEmpty = false) {
  const trimmed = value.trim();
  if (!trimmed) {
    return allowEmpty ? true : 'Please enter a path.';
  }

  const resolved = normalizeInputPath(trimmed);
  if (!fs.existsSync(resolved)) {
    return `File not found: ${resolved}`;
  }

  try {
    const stat = fs.statSync(resolved);
    if (!stat.isFile()) {
      return `Not a file: ${resolved}`;
    }
  } catch {
    return `File not found: ${resolved}`;
  }

  return true;
}

/**
 * Prompt for a GGUF model path.
 * @param {object} opts
 * @param {string} opts.message
 * @param {boolean} [opts.allowEmpty=false]
 * @returns {Promise<string|null>}
 */
async function promptForGgufPath({ message, allowEmpty = false }) {
  const answer = await text({
    message,
    validate: (value) => validateGgufPath(value, allowEmpty),
  });

  if (isCancel(answer)) {
    cancel('Skipping auto-start.');
    return null;
  }

  const trimmed = answer.trim();
  if (!trimmed) {
    return null;
  }

  return normalizeInputPath(trimmed);
}

/**
 * Determine whether a file name looks like a Llama 3 model.
 * @param {string} fileName
 * @returns {boolean}
 */
function isLlama3Model(fileName) {
  const lower = fileName.toLowerCase();
  return lower.includes('llama') && (lower.includes('llama3') || lower.includes('llama-3') || lower.includes('llama_3') || /\b3\b/.test(lower));
}

/**
 * Determine whether a file name looks like a Phi-3 model.
 * @param {string} fileName
 * @returns {boolean}
 */
function isPhi3Model(fileName) {
  const lower = fileName.toLowerCase();
  return lower.includes('phi') && (lower.includes('phi3') || lower.includes('phi-3') || lower.includes('phi_3') || /\b3\b/.test(lower));
}

/**
 * Scan a set of directories for GGUF files and filter them by model family.
 * @param {string[]} roots
 * @param {(fileName: string) => boolean} matcher
 * @returns {Promise<Array<{path: string, size: number}>>}
 */
async function findMatchingModels(roots, matcher) {
  const matches = [];
  const seen = new Set();

  for (const root of roots) {
    const models = await findGgufModels(root);
    for (const model of models) {
      if (seen.has(model.path)) {
        continue;
      }
      seen.add(model.path);
      if (matcher(path.basename(model.path))) {
        matches.push(model);
      }
    }
  }

  matches.sort((a, b) => a.path.localeCompare(b.path));
  return matches;
}

/**
 * Prompt for a model from a preset family or a custom path.
 * @param {object} opts
 * @param {string} opts.family
 * @param {(fileName: string) => boolean} opts.matcher
 * @returns {Promise<string|null>}
 */
async function choosePresetModel({ family, matcher }) {
  const roots = [...new Set([
    defaultModelsDir(),
    process.cwd(),
    path.join(process.cwd(), 'models'),
  ])];

  const scanSpinner = spinner();
  scanSpinner.start(`Scanning for ${family} .gguf models…`);
  const matches = await findMatchingModels(roots, matcher);
  scanSpinner.stop();

  if (matches.length === 0) {
    log.warn(`No ${family} .gguf files were found in: ${roots.join(', ')}`);
    return promptForGgufPath({
      message: `Enter the absolute path to your ${family} .gguf file:`,
    });
  }

  if (matches.length === 1) {
    const only = matches[0];
    const relative = path.relative(process.cwd(), only.path) || path.basename(only.path);
    const useIt = await confirm({
      message: `Use ${relative} (${formatBytes(only.size)})?`,
      initialValue: true,
    });

    if (isCancel(useIt)) {
      cancel('Skipping auto-start.');
      return null;
    }

    if (useIt) {
      return only.path;
    }

    return promptForGgufPath({
      message: `Enter the absolute path to your ${family} .gguf file:`,
    });
  }

  const selection = await select({
    message: `Select a ${family} model to load:`,
    options: [
      ...matches.map((model) => ({
        label: `${path.relative(process.cwd(), model.path) || path.basename(model.path)} (${formatBytes(model.size)})`,
        value: model.path,
      })),
      { label: 'None of these — enter a path manually…', value: '__manual__' },
    ],
  });

  if (isCancel(selection)) {
    cancel('Skipping auto-start.');
    return null;
  }

  if (selection === '__manual__') {
    return promptForGgufPath({
      message: `Enter the absolute path to your ${family} .gguf file:`,
    });
  }

  return selection;
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
      .forEach((line) => console.log(pc.dim(`  [llama] ${line}`)));
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
  const choice = await select({
    message: 'Choose how to select a model:',
    options: [
      { label: 'Llama 3', value: 'llama3' },
      { label: 'Phi-3', value: 'phi3' },
      { label: 'Custom path…', value: 'custom' },
      { label: "Skip auto-start (I'll run the server myself)", value: 'skip' },
    ],
  });

  if (isCancel(choice)) {
    cancel('Skipping auto-start.');
    return null;
  }

  if (choice === 'skip') {
    return null;
  }

  if (choice === 'custom') {
    return promptForGgufPath({
      message: 'Enter the absolute path to your .gguf model file:',
    });
  }

  if (choice === 'llama3') {
    return choosePresetModel({ family: 'Llama 3', matcher: isLlama3Model });
  }

  if (choice === 'phi3') {
    return choosePresetModel({ family: 'Phi-3', matcher: isPhi3Model });
  }

  return null;
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
    console.log(pc.green(`✔ Detected a running llama.cpp server at ${baseUrl} — reusing it.`));
    return { endpoint: defaultEndpoint, handle: null };
  }
  // If the user pinned a custom endpoint and it is alive, reuse that too.
  if (process.env.LLAMA_ENDPOINT) {
    const customBase = process.env.LLAMA_ENDPOINT.replace(/\/v1\/.*$/, '');
    if (customBase !== baseUrl && (await isServerHealthy(customBase))) {
      console.log(pc.green(`✔ Detected a running server at ${customBase} — reusing it.`));
      return { endpoint: defaultEndpoint, handle: null };
    }
  }

  intro(pc.bold('llama.cpp server setup'));

  // 2) Pick a model.
  const modelPath = await chooseModelFile();
  if (!modelPath) {
    outro(pc.dim(`Using ${defaultEndpoint}.`));
    return { endpoint: defaultEndpoint, handle: null };
  }

  // 3) Locate the server binary.
  let binary = await detectServerBinary();
  if (!binary) {
    const binaryPath = await text({
      message: 'Could not find a llama.cpp server binary. Enter its path, or press Enter to skip auto-start:',
      validate: (value) => validateBinaryPath(value, true),
    });

    if (isCancel(binaryPath)) {
      cancel('Skipping auto-start.');
      outro(pc.dim(`Using ${defaultEndpoint}.`));
      return { endpoint: defaultEndpoint, handle: null };
    }

    const trimmed = binaryPath.trim();
    if (!trimmed) {
      log.info(`Skipping auto-start. The agent will use ${defaultEndpoint}.`);
      outro(pc.dim(`Using ${defaultEndpoint}.`));
      return { endpoint: defaultEndpoint, handle: null };
    }

    const resolvedBinary = normalizeInputPath(trimmed);
    if (!fs.existsSync(resolvedBinary)) {
      log.warn(`Binary not found at ${resolvedBinary}. Falling back to ${defaultEndpoint}.`);
      outro(pc.dim(`Using ${defaultEndpoint}.`));
      return { endpoint: defaultEndpoint, handle: null };
    }

    binary = resolvedBinary;
    log.info(`Using ${binary}. Set LLAMA_SERVER_BIN to skip this prompt next time.`);
  } else {
    log.info(`Using server binary: ${binary}`);
  }

  // 4) Spawn + wait for health.
  console.log(pc.blueBright(`\nStarting llama.cpp server:\n  ${binary} -m ${modelPath} -c ${contextWindow} --host ${host} --port ${port}\n`));
  const handle = spawnServer({ binary, modelPath, contextWindow, port, host });

  const waitSpinner = spinner();
  waitSpinner.start('Waiting for the server to become ready…');
  const alive = () => handle.process.exitCode === null && !handle.process.killed;
  const healthy = await waitForHealth(handle.baseUrl, HEALTH_TIMEOUT_MS, alive);

  if (!healthy) {
    waitSpinner.stop(pc.red('The llama.cpp server did not become healthy in time.'));
    handle.stop();
    log.warn(`The server failed to start (check the [llama] logs above). Falling back to ${defaultEndpoint}.`);
    outro(pc.dim(`Using ${defaultEndpoint}.`));
    return { endpoint: defaultEndpoint, handle: null };
  }

  waitSpinner.stop(pc.green(`llama.cpp server is ready at ${handle.baseUrl}.`));
  outro(pc.green(`Using ${handle.baseUrl}.`));
  return { endpoint: handle.endpoint, handle };
}
