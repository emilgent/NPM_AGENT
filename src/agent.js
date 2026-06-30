/**
 * src/agent.js
 * ----------------------------------------------------------------------------
 * The Core Asynchronous Scheduler.
 *
 * Responsibilities:
 *   - Hold the canonical system prompt that defines the agent's behaviour.
 *   - Drive the recursive "agentic" execution loop:
 *
 *         Thought -> Tool Selection -> Safety Check -> Execution -> Observation
 *
 *   - Talk to a local llama.cpp instance over its OpenAI-compatible
 *     /v1/chat/completions REST endpoint, passing the native `tools` array and
 *     handling the returned `tool_calls`.
 *   - Print highly detailed, colour-coded terminal logs (THOUGHT / ACTION /
 *     USER APPROVAL / OBSERVATION / TOKEN ENGINE).
 *   - Track and report per-turn and cumulative token usage.
 *   - Detect task completion so the CLI layer knows when to persist the session.
 *
 * The loop is deliberately resilient: any tool error or non-zero shell exit is
 * captured and fed straight back to the model as a `tool` role observation so
 * the LLM can autonomously debug and retry.
 * ----------------------------------------------------------------------------
 */

import chalk from 'chalk';
import ora from 'ora';
import fetch from 'node-fetch';

import { toolSchemas, dispatchTool, writeFile, toolImplementations } from './tools.js';

/** Known tool names, used to detect text-format tool calls. */
const KNOWN_TOOL_NAMES = new Set(Object.keys(toolImplementations));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Default llama.cpp OpenAI-compatible chat-completions endpoint. Overridable
 * via the LLAMA_ENDPOINT environment variable.
 */
const DEFAULT_ENDPOINT =
  process.env.LLAMA_ENDPOINT || 'http://localhost:8080/v1/chat/completions';

/**
 * The model name advertised to the server. llama.cpp ignores this for routing
 * (it serves whatever model is loaded) but the OpenAI schema requires the
 * field. Overridable via LLAMA_MODEL.
 */
const DEFAULT_MODEL = process.env.LLAMA_MODEL || 'local-model';

/**
 * Sentinel marker the model is instructed to emit (as the start of a plain text
 * assistant message, with no tool calls) once the overall task is fully done.
 */
export const COMPLETION_MARKER = 'TASK_COMPLETE';

/**
 * Hard ceiling on the number of automatic loop iterations within a single user
 * turn. Prevents runaway loops if the model keeps calling tools indefinitely.
 */
const MAX_ITERATIONS_PER_TURN = 50;

// ---------------------------------------------------------------------------
// Auto-write interceptor — code-block extraction + filename inference
// ---------------------------------------------------------------------------

/**
 * Language hint (from a fenced code block) → file extension mapping.
 * Used as a fallback when the surrounding text contains no filename.
 */
const LANG_EXT_MAP = new Map([
  ['python', '.py'], ['py', '.py'],
  ['javascript', '.js'], ['js', '.js'], ['jsx', '.jsx'],
  ['typescript', '.ts'], ['ts', '.ts'], ['tsx', '.tsx'],
  ['html', '.html'], ['htm', '.html'],
  ['css', '.css'], ['scss', '.scss'], ['sass', '.sass'], ['less', '.less'],
  ['json', '.json'], ['jsonc', '.json'],
  ['yaml', '.yaml'], ['yml', '.yaml'],
  ['bash', '.sh'], ['sh', '.sh'], ['shell', '.sh'], ['zsh', '.sh'],
  ['markdown', '.md'], ['md', '.md'],
  ['xml', '.xml'], ['svg', '.svg'],
  ['sql', '.sql'],
  ['ruby', '.rb'], ['rb', '.rb'],
  ['rust', '.rs'], ['rs', '.rs'],
  ['go', '.go'],
  ['java', '.java'],
  ['c', '.c'], ['cpp', '.cpp'], ['cxx', '.cpp'],
  ['toml', '.toml'],
  ['ini', '.ini'],
  ['txt', '.txt'],
  ['csv', '.csv'],
  ['env', '.env'],
  ['cfg', '.cfg'],
]);

/**
 * Allowed file extensions for auto-write (text-based formats only).
 * Must stay in sync with the text-extension allowlist in tools.js.
 */
const AUTOWRITE_EXTS = new Set([
  '.py', '.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx',
  '.html', '.htm', '.css', '.scss', '.sass', '.less',
  '.json', '.jsonc', '.yaml', '.yml',
  '.sh', '.md', '.markdown', '.xml', '.svg', '.sql',
  '.rb', '.rs', '.go', '.java', '.c', '.cpp', '.cxx',
  '.toml', '.ini', '.txt', '.csv', '.env', '.cfg',
]);

/** Language hints that typically represent terminal commands, not file content. */
const SHELL_LANGS = new Set(['sh', 'bash', 'shell', 'zsh', 'cmd', 'powershell', 'bat', 'ps1', 'console', 'terminal']);

// ---------------------------------------------------------------------------
// Text-format tool call interceptor
// ---------------------------------------------------------------------------

/**
 * Many small / local models do not use the native function-calling protocol.
 * Instead they output a fenced JSON block such as:
 *
 *   ```json
 *   { "name": "writeFile", "arguments": { "filePath": "hello.py", ... } }
 *   ```
 *
 * This function detects those blocks and returns them as structured tool calls
 * so the agent loop can dispatch them through the normal `dispatchTool` path.
 *
 * @param {string} content - The full assistant message text.
 * @returns {Array<{name: string, args: object}>}
 */
export function extractTextToolCalls(content) {
  const codeBlockRe = /```(?:json)?\n([\s\S]*?)```/g;
  const calls = [];
  let match;
  while ((match = codeBlockRe.exec(content)) !== null) {
    const raw = match[1].trim();
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      continue;
    }
    if (
      parsed &&
      typeof parsed === 'object' &&
      typeof parsed.name === 'string' &&
      KNOWN_TOOL_NAMES.has(parsed.name) &&
      parsed.arguments !== undefined &&
      typeof parsed.arguments === 'object'
    ) {
      calls.push({ name: parsed.name, args: parsed.arguments });
    }
  }
  return calls;
}

/**
 * Check whether a code block body is a JSON-format tool call that will be
 * handled by `extractTextToolCalls` instead of auto-write.
 *
 * @param {string} code - The code block body.
 * @param {string} lang - The fenced language hint.
 * @returns {boolean}
 */
function isTextToolCall(code, lang) {
  if (lang !== '' && lang !== 'json' && lang !== 'jsonc') return false;
  try {
    const parsed = JSON.parse(code.trim());
    return (
      parsed &&
      typeof parsed === 'object' &&
      typeof parsed.name === 'string' &&
      KNOWN_TOOL_NAMES.has(parsed.name) &&
      parsed.arguments !== undefined
    );
  } catch {
    return false;
  }
}

/**
 * Try to find a plausible filename inside a text fragment by looking for
 * backtick-quoted or double-quoted paths with known text-file extensions.
 *
 * @param {string} text - The text to search.
 * @returns {string|null} The best candidate filename, or null.
 */
function findFilenameInText(text) {
  // Backtick-quoted filenames (strongest signal).
  const backtickRe = /`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]{1,8})`/g;
  const candidates = [];
  let m;
  while ((m = backtickRe.exec(text)) !== null) {
    const ext = '.' + m[1].split('.').pop().toLowerCase();
    if (AUTOWRITE_EXTS.has(ext)) candidates.push(m[1]);
  }
  if (candidates.length > 0) return candidates[candidates.length - 1];

  // Double-quoted filenames.
  const quotedRe = /"([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]{1,8})"/g;
  while ((m = quotedRe.exec(text)) !== null) {
    const ext = '.' + m[1].split('.').pop().toLowerCase();
    if (AUTOWRITE_EXTS.has(ext)) candidates.push(m[1]);
  }
  if (candidates.length > 0) return candidates[candidates.length - 1];

  // Bare filenames preceded by common phrases.
  const phraseRe = /(?:named|called|save (?:it )?(?:as|to)|file\s+)\s*[`"']?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]{1,8})[`"']?/gi;
  while ((m = phraseRe.exec(text)) !== null) {
    const ext = '.' + m[1].split('.').pop().toLowerCase();
    if (AUTOWRITE_EXTS.has(ext)) return m[1];
  }

  return null;
}

/**
 * Detect whether a code block looks like a terminal command rather than file
 * content. Terminal commands typically start with common CLI tool names.
 *
 * @param {string} code - The code block body.
 * @returns {boolean}
 */
function looksLikeCommand(code) {
  const first = code.trim().split('\n')[0].replace(/^\$\s*/, '').trim();
  return /^(python|python3|node|npm|npx|pip|cd|ls|cat|echo|mkdir|git|curl|wget|.\/)/.test(first);
}

/**
 * Parse the assistant's plain-text response for fenced code blocks and
 * surrounding filename hints. Returns a list of files that should be
 * auto-written to disk.
 *
 * @param {string} content - The full assistant message text.
 * @returns {Array<{filePath: string, code: string}>}
 */
export function extractFileWrites(content) {
  const results = [];
  const codeBlockRe = /```(\w*)\n([\s\S]*?)```/g;
  const blocks = [];
  let match;
  while ((match = codeBlockRe.exec(content)) !== null) {
    blocks.push({
      lang: (match[1] || '').toLowerCase(),
      code: match[2],
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  if (blocks.length === 0) return results;

  const usedPaths = new Set();

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];

    // Skip shell / command blocks.
    if (SHELL_LANGS.has(block.lang) || looksLikeCommand(block.code)) continue;

    // Skip JSON blocks that are text-format tool calls (handled separately).
    if (isTextToolCall(block.code, block.lang)) continue;

    // Skip trivially short snippets.
    if (block.code.trim().length < 10) continue;

    // --- Filename resolution -------------------------------------------------
    // Look at the text segment between the previous block (or start) and this
    // block, then fall back to the segment after this block.
    const segBefore = content.slice(i === 0 ? 0 : blocks[i - 1].end, block.start);
    const segAfter = content.slice(
      block.end,
      i < blocks.length - 1 ? blocks[i + 1].start : content.length,
    );

    let filePath = findFilenameInText(segBefore) || findFilenameInText(segAfter);

    // Fallback: generate a default name from the language hint.
    if (!filePath && block.lang) {
      const ext = LANG_EXT_MAP.get(block.lang);
      if (ext) {
        // Use a descriptive default; append a counter if collisions arise.
        const base = `output${ext}`;
        filePath = usedPaths.has(base) ? `output_${i}${ext}` : base;
      }
    }

    if (!filePath) continue;
    if (usedPaths.has(filePath)) {
      // Disambiguate duplicate targets.
      const ext = '.' + filePath.split('.').pop();
      const stem = filePath.slice(0, filePath.length - ext.length);
      filePath = `${stem}_${i}${ext}`;
    }
    usedPaths.add(filePath);
    results.push({ filePath, code: block.code });
  }

  return results;
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/**
 * Build the immutable system prompt that is injected exactly once, as the very
 * first message, when a brand-new session is created.
 *
 * @returns {string} The system prompt text.
 */
export function buildSystemPrompt() {
  return [
    'You are NPM-Agent, an elite autonomous software engineering agent operating directly',
    'inside the user\'s current working directory (their live codebase). You specialise in',
    'Node.js / NPM package development.',
    '',
    'You operate in a strict reasoning loop:',
    '  Thought -> Tool Selection -> Safety Check -> Execution -> Observation',
    '',
    'You have access to the following tools (invoke them via native function calling):',
    '  - listFiles(): recursively list the project files (node_modules/.git/.agent_sessions skipped).',
    '  - readFile(filePath): read a text file (max 500 KB, binary files are forbidden).',
    '  - writeFile(filePath, content): write/overwrite a text file (parent dirs auto-created).',
    '  - executeCommand(command): run a shell command (1h timeout). Mutating commands need user approval.',
    '  - webSearch(query): search the web (SerpAPI) for documentation and answers.',
    '  - fetchWebpage(url): download and clean a web page for reading.',
    '',
    'Operating rules:',
    '  1. Always inspect the project (listFiles / readFile) before making changes.',
    '  2. Make small, verifiable edits. After editing code, run the relevant tests or build',
    '     command via executeCommand and read the output to confirm success.',
    '  3. If a tool returns an error or a command fails, analyse the error and fix it yourself',
    '     in the next step. Do not give up after a single failure.',
    '  4. Never attempt to read or write binary files.',
    '  5. State-mutating shell commands (npm install, rm, git commit, ...) will pause for human',
    '     approval. If a command is rejected, find a different, safe approach.',
    '  6. CRITICAL — always write files to disk: whenever you produce content that belongs in a',
    '     file (code, configuration, documentation, data, etc.) you MUST call writeFile() to',
    '     persist it to the correct path. NEVER just display file content in your response',
    '     without also writing it. If you generate or modify code, always use writeFile() so',
    '     the result actually exists on disk. Summarise what you wrote AFTER the writeFile call.',
    '',
    'Completion protocol:',
    `  - When (and only when) the user's goal is FULLY accomplished and verified, reply with a`,
    `    plain text message (no tool calls) whose first line is exactly "${COMPLETION_MARKER}",`,
    '    followed by a concise summary of everything you did.',
    '  - If you need clarification or additional input from the user, reply with a plain text',
    '    message (no tool calls) asking your question.',
    '',
    'Be precise, methodical and production-grade. Write complete code with no placeholders.',
  ].join('\n');
}

// ---------------------------------------------------------------------------
// Logging helpers (Chalk colour-coded)
// ---------------------------------------------------------------------------

/** Print the model's reasoning string. */
function logThought(text) {
  if (!text || !text.trim()) return;
  console.log(chalk.cyan('[THOUGHT] ') + chalk.cyan(text.trim()));
}

/** Print a tool invocation with its structured arguments. */
function logAction(name, args) {
  const argString = JSON.stringify(args ?? {}, null, 0);
  console.log(chalk.yellow('[ACTION] ') + chalk.yellow(`${name}(${argString})`));
}

/** Print a truncated/summarised tool observation. */
function logObservation(text) {
  const flat = String(text ?? '');
  const MAX = 1200;
  const shown = flat.length > MAX ? `${flat.slice(0, MAX)}\n  [...truncated ${flat.length - MAX} chars...]` : flat;
  const indented = shown
    .split('\n')
    .map((line) => `  ${line}`)
    .join('\n');
  console.log(chalk.gray('[OBSERVATION]'));
  console.log(chalk.dim(indented));
}

/** Print clean per-turn and cumulative token metrics. */
function logTokenUsage(usage, cumulative, contextWindow) {
  if (!usage) {
    console.log(chalk.green('[TOKEN ENGINE] ') + chalk.green('(no usage metadata returned by server)'));
    return;
  }
  const prompt = usage.prompt_tokens ?? 0;
  const completion = usage.completion_tokens ?? 0;
  const total = usage.total_tokens ?? prompt + completion;

  const pct = contextWindow
    ? ` | context: ${cumulative.lastTotal}/${contextWindow} (${((cumulative.lastTotal / contextWindow) * 100).toFixed(1)}%)`
    : '';

  console.log(
    chalk.green('[TOKEN ENGINE] ') +
      chalk.green(
        `turn → prompt: ${prompt}, completion: ${completion}, total: ${total} | ` +
          `session total: ${cumulative.total}${pct}`,
      ),
  );
}

// ---------------------------------------------------------------------------
// llama.cpp REST interaction
// ---------------------------------------------------------------------------

/**
 * Send the current message history (plus tool schemas) to the llama.cpp
 * OpenAI-compatible endpoint and return the parsed response.
 *
 * @param {Array<object>} messages - The OpenAI-format message history.
 * @param {object} config - Active runtime configuration.
 * @param {string} config.endpoint - Chat-completions endpoint URL.
 * @param {string} config.model - Model name to advertise.
 * @param {number} config.contextWindow - Declared max context window.
 * @returns {Promise<object>} The parsed JSON response body.
 * @throws {Error} On network failure or non-2xx HTTP responses.
 */
async function callLlama(messages, config) {
  const body = {
    model: config.model,
    messages,
    tools: toolSchemas,
    tool_choice: 'auto',
    temperature: 0.2,
    stream: false,
  };

  let response;
  try {
    response = await fetch(config.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(
      `Network error contacting llama.cpp at ${config.endpoint}: ${err.message}. ` +
        'Is the server running (e.g. ./server -m model.gguf -c 8192 --port 8080)?',
    );
  }

  if (!response.ok) {
    const errText = await response.text().catch(() => '');
    throw new Error(
      `llama.cpp returned HTTP ${response.status} ${response.statusText}: ${errText.slice(0, 500)}`,
    );
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Tool-call argument parsing
// ---------------------------------------------------------------------------

/**
 * Safely parse the JSON arguments string attached to a tool call. llama.cpp
 * occasionally emits empty strings or malformed JSON; we degrade gracefully.
 *
 * @param {string} rawArgs - The raw `arguments` string from the tool call.
 * @returns {object} The parsed arguments object (empty object on failure).
 */
function parseToolArguments(rawArgs) {
  if (rawArgs === undefined || rawArgs === null || rawArgs === '') {
    return {};
  }
  if (typeof rawArgs === 'object') {
    return rawArgs; // already parsed
  }
  try {
    return JSON.parse(rawArgs);
  } catch {
    // Some models wrap arguments oddly; try a forgiving extraction of the first
    // JSON object substring before giving up.
    const match = String(rawArgs).match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch {
        /* fall through */
      }
    }
    return { __parse_error__: true, raw: String(rawArgs) };
  }
}

// ---------------------------------------------------------------------------
// Core agentic loop
// ---------------------------------------------------------------------------

/**
 * Run the autonomous agent loop for a single user turn until either:
 *   - the model emits a plain text response (no tool calls), or
 *   - the per-turn iteration cap is reached.
 *
 * The `messages` array is mutated in place to accumulate the full conversation
 * (assistant + tool messages), keeping it consistent with the OpenAI schema for
 * later persistence.
 *
 * @param {Array<object>} messages - The mutable conversation history.
 * @param {object} config - Runtime configuration.
 * @param {object} cumulative - Token accumulator { total, lastTotal }.
 * @returns {Promise<{status: 'complete'|'awaiting_user', content: string}>}
 */
export async function runTurn(messages, config, cumulative) {
  for (let iteration = 0; iteration < MAX_ITERATIONS_PER_TURN; iteration += 1) {
    const spinner = ora({
      text: chalk.blue('Contacting llama.cpp…'),
      color: 'blue',
    }).start();

    let data;
    try {
      data = await callLlama(messages, config);
      spinner.stop();
    } catch (err) {
      spinner.fail(chalk.red('LLM request failed.'));
      throw err;
    }

    // --- Token tracking --------------------------------------------------
    const usage = data.usage;
    if (usage) {
      cumulative.total += usage.total_tokens ?? 0;
      cumulative.lastTotal = usage.total_tokens ?? cumulative.lastTotal;
    }
    logTokenUsage(usage, cumulative, config.contextWindow);

    const choice = Array.isArray(data.choices) ? data.choices[0] : undefined;
    if (!choice || !choice.message) {
      throw new Error('llama.cpp response contained no choices/message.');
    }

    const assistantMessage = choice.message;

    // Normalise the assistant message into strict OpenAI schema before storing.
    const storedAssistant = {
      role: 'assistant',
      content: assistantMessage.content ?? null,
    };
    if (Array.isArray(assistantMessage.tool_calls) && assistantMessage.tool_calls.length > 0) {
      storedAssistant.tool_calls = assistantMessage.tool_calls;
    }
    messages.push(storedAssistant);

    // Surface any reasoning text the model included.
    if (assistantMessage.content) {
      logThought(assistantMessage.content);
    }

    const toolCalls = assistantMessage.tool_calls;

    // --- No tool calls => natural turn boundary --------------------------
    if (!Array.isArray(toolCalls) || toolCalls.length === 0) {
      const content = (assistantMessage.content ?? '').trim();

      // Detect explicit completion signal.
      if (content.startsWith(COMPLETION_MARKER)) {
        const summary = content.slice(COMPLETION_MARKER.length).trim();
        console.log(chalk.greenBright.bold('\n[TASK COMPLETE] ') + chalk.green(summary || '(no summary provided)'));
        return { status: 'complete', content: summary };
      }

      // --- Text-format tool call interceptor --------------------------------
      // Some local models output tool invocations as ```json {"name":"writeFile",
      // "arguments":{...}} ``` instead of using the native function-calling
      // protocol.  Detect these, dispatch them through the normal tool pipeline,
      // and feed the results back to the model.
      const textCalls = extractTextToolCalls(content);
      if (textCalls.length > 0) {
        const resultLines = [];
        for (const { name, args } of textCalls) {
          logAction(name, args);
          let observation;
          try {
            observation = await dispatchTool(name, args);
          } catch (err) {
            observation = `Error while executing tool "${name}": ${err && err.message ? err.message : String(err)}`;
          }
          logObservation(observation);
          const obsStr = typeof observation === 'string' ? observation : JSON.stringify(observation);
          resultLines.push(`[${name}] ${obsStr}`);
        }

        console.log(
          chalk.blueBright(
            `\n[TEXT-TOOL-CALL] Executed ${textCalls.length} tool call(s) from model text output.`,
          ),
        );

        messages.push({
          role: 'user',
          content:
            `[System] Your tool calls were executed successfully. Results:\n\n` +
            resultLines.join('\n\n') +
            `\n\nContinue with the task. When fully done, reply with TASK_COMPLETE followed by a summary.`,
        });
        continue; // re-enter the agent loop
      }

      // --- Auto-write interceptor -------------------------------------------
      // Many local models fail to use function-calling and instead output code
      // blocks as plain text.  Detect these, write them to disk automatically,
      // and re-enter the loop so the model sees the files were created.
      const pendingWrites = extractFileWrites(content);
      if (pendingWrites.length > 0) {
        const written = [];
        for (const { filePath, code } of pendingWrites) {
          logAction('writeFile (auto)', { filePath });
          try {
            const obs = await writeFile(filePath, code);
            logObservation(obs);
            written.push(filePath);
          } catch (err) {
            logObservation(`Auto-write failed for "${filePath}": ${err.message}`);
          }
        }

        if (written.length > 0) {
          console.log(
            chalk.blueBright(
              `\n[AUTO-WRITE] Wrote ${written.length} file(s) to disk: ${written.join(', ')}`,
            ),
          );

          // Inject a synthetic user message so the model sees the result and can
          // verify / continue / emit TASK_COMPLETE on the next iteration.
          messages.push({
            role: 'user',
            content:
              `[System] The following files were automatically written to disk based on your response:\n` +
              written.map((f) => `  - ${f}`).join('\n') +
              `\nPlease verify them (e.g. readFile / executeCommand) and then complete the task.`,
          });
          continue; // re-enter the agent loop
        }
      }

      // Otherwise the model is talking to the user (question / status).
      console.log(chalk.whiteBright('\n[AGENT] ') + chalk.white(content || '(empty response)'));
      return { status: 'awaiting_user', content };
    }

    // --- Execute each requested tool call --------------------------------
    for (const call of toolCalls) {
      const fn = call.function || {};
      const name = fn.name;
      const args = parseToolArguments(fn.arguments);

      logAction(name, args);

      let observation;
      try {
        // dispatchTool resolves the implementation and applies its own safety
        // boundaries (binary guard, command approval, size limits, ...).
        observation = await dispatchTool(name, args);
      } catch (err) {
        // Exception resilience: capture stack/message and feed it back so the
        // LLM can debug autonomously on its next turn.
        observation = `Error while executing tool "${name}": ${err && err.message ? err.message : String(err)}`;
        if (err && err.stack) {
          observation += `\nStack:\n${err.stack}`;
        }
      }

      logObservation(observation);

      // Append the tool result as a `tool` role message keyed by tool_call_id.
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: typeof observation === 'string' ? observation : JSON.stringify(observation),
      });
    }

    // Loop continues: the accumulated tool observations are sent back to the
    // model on the next iteration so it can decide the next action.
  }

  // Safety valve: too many iterations without a natural stop.
  console.log(
    chalk.red.bold(
      `\n[SCHEDULER] Reached the per-turn iteration cap (${MAX_ITERATIONS_PER_TURN}). Pausing for user input.`,
    ),
  );
  return {
    status: 'awaiting_user',
    content: `Reached the maximum of ${MAX_ITERATIONS_PER_TURN} automatic steps for this turn.`,
  };
}

/**
 * Factory for a fresh cumulative-token accumulator.
 * @returns {{total: number, lastTotal: number}}
 */
export function createTokenAccumulator() {
  return { total: 0, lastTotal: 0 };
}

/**
 * Resolve the active runtime configuration from explicit overrides + env vars.
 *
 * @param {object} overrides - Partial config (e.g. contextWindow from the menu).
 * @returns {{endpoint: string, model: string, contextWindow: number}}
 */
export function resolveConfig(overrides = {}) {
  return {
    endpoint: overrides.endpoint || DEFAULT_ENDPOINT,
    model: overrides.model || DEFAULT_MODEL,
    contextWindow: overrides.contextWindow || 8192,
  };
}
