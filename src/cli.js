#!/usr/bin/env node
/**
 * src/cli.js
 * ----------------------------------------------------------------------------
 * Terminal entrypoint for the npm-agent autonomous CLI tool.
 *
 * Responsibilities:
 *   - Scan for / create the `.agent_sessions/` directory in the current working
 *     directory.
 *   - Render the interactive Inquirer main menu:
 *         [Start New Session]  -> prompt for a goal description.
 *         [Load Existing Session] -> pick a saved session JSON file.
 *   - Ask the operator to declare the model's maximum context window.
 *   - Initialise the conversation message history (injecting the system prompt
 *     exactly ONCE for new sessions; never re-injecting it on load).
 *   - Drive the agent loop turn-by-turn, prompting the user for follow-up input
 *     between turns.
 *   - Persist the session to disk ONLY when the task is reported complete or on
 *     a clean exit signal (Ctrl+C / SIGINT / SIGTERM).
 * ----------------------------------------------------------------------------
 */

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

import chalk from 'chalk';
import inquirer from 'inquirer';

import {
  buildSystemPrompt,
  runTurn,
  createTokenAccumulator,
  resolveConfig,
} from './agent.js';
import { prepareLlamaServer } from './llamaServer.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Name of the directory used to persist chat sessions. */
const SESSIONS_DIR_NAME = '.agent_sessions';

/** Absolute path to the sessions directory (relative to the invocation cwd). */
const SESSIONS_DIR = path.resolve(process.cwd(), SESSIONS_DIR_NAME);

// ---------------------------------------------------------------------------
// Mutable application state used by the exit handler for final persistence.
// ---------------------------------------------------------------------------

const appState = {
  /** @type {Array<object>|null} The active message history. */
  messages: null,
  /** @type {string|null} Absolute path of the session file to persist. */
  sessionFilePath: null,
  /** @type {boolean} Whether the session has already been saved this run. */
  saved: false,
  /** @type {{stop: () => void}|null} Handle to a llama.cpp server we started. */
  serverHandle: null,
};

// ---------------------------------------------------------------------------
// Filesystem / session helpers
// ---------------------------------------------------------------------------

/**
 * Ensure the `.agent_sessions/` directory exists, creating it if necessary.
 * @returns {Promise<void>}
 */
async function ensureSessionsDirectory() {
  try {
    await fsp.mkdir(SESSIONS_DIR, { recursive: true });
  } catch (err) {
    console.error(chalk.red(`Failed to create ${SESSIONS_DIR_NAME}/: ${err.message}`));
    throw err;
  }
}

/**
 * List the JSON session files currently present in `.agent_sessions/`.
 * @returns {Promise<string[]>} File names (not full paths), newest first.
 */
async function listSessionFiles() {
  let entries;
  try {
    entries = await fsp.readdir(SESSIONS_DIR, { withFileTypes: true });
  } catch {
    return [];
  }

  const files = entries
    .filter((e) => e.isFile() && e.name.endsWith('.json'))
    .map((e) => e.name);

  // Sort by modification time, newest first, for convenience.
  const withStats = await Promise.all(
    files.map(async (name) => {
      const stat = await fsp.stat(path.join(SESSIONS_DIR, name));
      return { name, mtime: stat.mtimeMs };
    }),
  );
  withStats.sort((a, b) => b.mtime - a.mtime);
  return withStats.map((f) => f.name);
}

/**
 * Generate a unique, timestamped session file name (e.g.
 * session_2026-06-20T14-52-03.json) to avoid clobbering same-day sessions.
 * @returns {string} The generated file name.
 */
function generateSessionFileName() {
  const now = new Date();
  const stamp = now.toISOString().replace(/:/g, '-').replace(/\..+$/, '');
  return `session_${stamp}.json`;
}

/**
 * Persist the active session's message history to disk. Only the `messages`
 * array is written, matching the mandated schema.
 *
 * @param {Array<object>} messages - The conversation history.
 * @param {string} filePath - Absolute destination path.
 * @returns {Promise<void>}
 */
async function saveSession(messages, filePath) {
  const payload = { messages };
  await fsp.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf8');
}

/**
 * Load a session's message history from disk.
 * @param {string} fileName - File name within `.agent_sessions/`.
 * @returns {Promise<Array<object>>} The restored messages array.
 */
async function loadSession(fileName) {
  const fullPath = path.join(SESSIONS_DIR, fileName);
  const raw = await fsp.readFile(fullPath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || !Array.isArray(parsed.messages)) {
    throw new Error(`Session file "${fileName}" is missing a valid "messages" array.`);
  }
  return parsed.messages;
}

// ---------------------------------------------------------------------------
// Interactive menus
// ---------------------------------------------------------------------------

/**
 * Render the main menu and gather all startup configuration.
 *
 * @returns {Promise<{messages: Array<object>, sessionFilePath: string, contextWindow: number}>}
 */
async function runMainMenu() {
  console.log(chalk.bold.blueBright('\n=== npm-agent :: Autonomous NPM Development Agent ===\n'));

  const existingSessions = await listSessionFiles();

  const { action } = await inquirer.prompt([
    {
      type: 'list',
      name: 'action',
      message: 'What would you like to do?',
      choices: [
        { name: '[Start New Session]', value: 'new' },
        {
          name: `[Load Existing Session]${existingSessions.length === 0 ? ' (none found)' : ''}`,
          value: 'load',
          disabled: existingSessions.length === 0 ? 'no saved sessions' : false,
        },
      ],
    },
  ]);

  // --- Context window declaration (asked for both new and loaded sessions) --
  const { contextChoice } = await inquirer.prompt([
    {
      type: 'list',
      name: 'contextChoice',
      message: "Declare your model's maximum context window (tokens):",
      choices: [
        { name: '4096', value: 4096 },
        { name: '8192', value: 8192 },
        { name: '16384', value: 16384 },
        { name: 'Custom…', value: 'custom' },
      ],
      default: 8192,
    },
  ]);

  let contextWindow = contextChoice;
  if (contextChoice === 'custom') {
    const { customContext } = await inquirer.prompt([
      {
        type: 'number',
        name: 'customContext',
        message: 'Enter the custom context window size (tokens):',
        default: 8192,
        validate: (value) =>
          Number.isInteger(value) && value > 0 ? true : 'Please enter a positive integer.',
      },
    ]);
    contextWindow = customContext;
  }

  // --- Branch: new vs. load -----------------------------------------------
  if (action === 'new') {
    const { goal } = await inquirer.prompt([
      {
        type: 'input',
        name: 'goal',
        message: 'Describe the goal for this session:',
        validate: (value) => (value.trim().length > 0 ? true : 'Please enter a goal.'),
      },
    ]);

    // System prompt is injected exactly ONCE, as the very first element.
    const messages = [
      { role: 'system', content: buildSystemPrompt() },
      { role: 'user', content: goal.trim() },
    ];

    const sessionFilePath = path.join(SESSIONS_DIR, generateSessionFileName());
    console.log(chalk.dim(`\nNew session will be saved to: ${sessionFilePath}\n`));
    return { messages, sessionFilePath, contextWindow };
  }

  // action === 'load'
  const { sessionFile } = await inquirer.prompt([
    {
      type: 'list',
      name: 'sessionFile',
      message: 'Select a session to load:',
      choices: existingSessions,
    },
  ]);

  const messages = await loadSession(sessionFile);
  const sessionFilePath = path.join(SESSIONS_DIR, sessionFile);
  console.log(chalk.dim(`\nLoaded ${messages.length} messages from ${sessionFile}.`));
  console.log(chalk.dim('(System prompt is NOT re-injected for loaded sessions.)\n'));

  // If the user loaded a session, give them the chance to add a new instruction.
  const { followUp } = await inquirer.prompt([
    {
      type: 'input',
      name: 'followUp',
      message: 'Enter a new instruction to continue this session (or leave blank to resume):',
    },
  ]);
  if (followUp && followUp.trim().length > 0) {
    messages.push({ role: 'user', content: followUp.trim() });
  }

  return { messages, sessionFilePath, contextWindow };
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

/**
 * Application entrypoint.
 * @returns {Promise<void>}
 */
async function main() {
  await ensureSessionsDirectory();

  const { messages, sessionFilePath, contextWindow } = await runMainMenu();

  // Wire up shared state for the clean-exit persistence handler.
  appState.messages = messages;
  appState.sessionFilePath = sessionFilePath;

  // Auto-start a local llama.cpp server (and let the user pick where their
  // models live) before the agent loop begins. Falls back to the default
  // endpoint if the user skips or no server can be started.
  const { endpoint, handle } = await prepareLlamaServer({ contextWindow });
  appState.serverHandle = handle;

  const config = resolveConfig({ contextWindow, endpoint });
  const tokens = createTokenAccumulator();

  console.log(
    chalk.dim(
      `Endpoint: ${config.endpoint} | Model: ${config.model} | Context window: ${config.contextWindow}\n`,
    ),
  );

  // ----- Turn-by-turn agent loop ------------------------------------------
  // The loop runs one "turn" (which itself may execute many tool calls), then
  // either finishes (task complete) or hands control back to the user.
  let running = true;
  while (running) {
    let result;
    try {
      result = await runTurn(messages, config, tokens);
    } catch (err) {
      console.error(chalk.red.bold(`\n[FATAL] ${err.message}`));
      console.log(chalk.yellow('Saving session before exiting due to the error above…'));
      await persistOnce(true);
      stopServer();
      process.exitCode = 1;
      return;
    }

    if (result.status === 'complete') {
      // Persistence trigger: task fully accomplished.
      await persistOnce(true);
      console.log(chalk.greenBright(`\nSession saved to ${sessionFilePath}. Goodbye!`));
      running = false;
      break;
    }

    // status === 'awaiting_user': prompt for the next instruction. This inner
    // loop keeps re-prompting the human (without re-invoking the model) until a
    // real message is entered or an exit is requested. Non-message commands
    // such as /save and empty input fall through to another prompt rather than
    // triggering a wasteful extra LLM round-trip.
    let advanceModel = false;
    while (!advanceModel && running) {
      const { nextInput } = await inquirer.prompt([
        {
          type: 'input',
          name: 'nextInput',
          message: chalk.whiteBright('Your reply (type /exit to quit, /save to persist now):'),
        },
      ]);

      const trimmed = (nextInput || '').trim();

      if (trimmed === '/exit' || trimmed === '/quit') {
        await persistOnce(true);
        console.log(chalk.greenBright(`\nSession saved to ${sessionFilePath}. Goodbye!`));
        running = false;
        break;
      }

      if (trimmed === '/save') {
        await persistOnce(true);
        console.log(chalk.green(`Session manually saved to ${sessionFilePath}.`));
        continue; // re-prompt the user without advancing the model
      }

      if (trimmed.length === 0) {
        console.log(chalk.dim('(empty input ignored — type a message, /save or /exit)'));
        continue; // re-prompt the user without advancing the model
      }

      // A real instruction was provided: record it and let the outer loop run
      // the next model turn. New content invalidates any prior save.
      messages.push({ role: 'user', content: trimmed });
      appState.saved = false;
      advanceModel = true;
    }
  }

  // Normal exit (task complete or /exit): tear down any server we started.
  stopServer();
}

/**
 * Persist the active session exactly once per logical save point. Pass
 * `force = true` to bypass the one-shot guard (used by the explicit /save).
 *
 * @param {boolean} [force=false]
 * @returns {Promise<void>}
 */
async function persistOnce(force = false) {
  if (!appState.messages || !appState.sessionFilePath) return;
  if (appState.saved && !force) return;
  try {
    await saveSession(appState.messages, appState.sessionFilePath);
    appState.saved = true;
  } catch (err) {
    console.error(chalk.red(`Failed to persist session: ${err.message}`));
  }
}

/**
 * Terminate any llama.cpp server this process started so it doesn't outlive the
 * CLI. No-op when the server was externally managed / reused.
 */
function stopServer() {
  if (appState.serverHandle && typeof appState.serverHandle.stop === 'function') {
    console.log(chalk.dim('Shutting down the llama.cpp server we started…'));
    appState.serverHandle.stop();
    appState.serverHandle = null;
  }
}

// ---------------------------------------------------------------------------
// Clean-exit signal handling
// ---------------------------------------------------------------------------

/**
 * Install handlers so that a Ctrl+C / SIGTERM triggers a final, clean session
 * write before the process terminates.
 */
function installSignalHandlers() {
  const handler = async (signal) => {
    console.log(chalk.yellow(`\nReceived ${signal} — persisting session before exit…`));
    await persistOnce(true);
    if (appState.sessionFilePath) {
      console.log(chalk.greenBright(`Session saved to ${appState.sessionFilePath}.`));
    }
    stopServer();
    process.exit(0);
  };

  process.on('SIGINT', () => handler('SIGINT'));
  process.on('SIGTERM', () => handler('SIGTERM'));
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

installSignalHandlers();

main().catch((err) => {
  console.error(chalk.red.bold(`\nUnexpected fatal error: ${err && err.stack ? err.stack : err}`));
  process.exitCode = 1;
});
